"""
MCP 客户端模块。

这个文件负责连接基于 stdio 的 MCP Server，完成工具发现，并把 Agent 发起的工具调用
转发到对应的 MCP Server。

实现方式：
- 不依赖 MCP SDK，直接通过标准输入/标准输出传输 JSON-RPC 消息。
- 每个 MCP Server 都由一个子进程承载。
- 每个 MCP 工具都会被包装成 `mcp__serverName__toolName` 形式，避免和本地工具重名。

配置来源：
- 全局配置：`~/.bear/settings.json`
- 项目配置：`.bear/settings.json`
- Claude Code 约定配置：`.mcp.json`

配置格式示例：
{
    "mcpServers": {
        "name": {
            "command": "...",
            "args": [...],
            "env": {...}
        }
    }
}
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import signal
from pathlib import Path
from typing import Any, Awaitable, Callable

from .ui import print_error, print_info

MCP_TOOL_TIMEOUT_SECONDS = 60.0
MCP_MAX_MESSAGE_BYTES = 8 * 1024 * 1024


class McpToolError(RuntimeError):
    """A valid MCP response explicitly reported a failed tool operation."""

    def __init__(self, result: dict):
        self.result = result
        content = result.get("content") or []
        text = "\n".join(str(item.get("text", "")) for item in content
                         if isinstance(item, dict) and item.get("type") == "text")
        super().__init__("MCP tool failed: " + (text or json.dumps(result, ensure_ascii=False)))


# ─── 单个 MCP 连接：一个 McpConnection 对应一个 MCP Server 子进程 ──────────────────


class BorrowedMcpManager:
    """A child may call its inherited tools, but never owns parent processes."""

    def __init__(self, owner: McpManager, allowed_tools: set[str]):
        self._owner = owner
        self._allowed_tools = frozenset(allowed_tools)

    def is_mcp_tool(self, name: str) -> bool:
        return name in self._allowed_tools

    async def call_tool(self, name: str, args: dict) -> str:
        if name not in self._allowed_tools:
            raise ValueError(f"MCP tool not available in this child: {name}")
        return await self._owner.call_tool(name, args)

    async def disconnect_all(self) -> None:
        # The parent's lifetime encloses all child calls and owns cleanup.
        pass


class McpConnection:
    """管理某一个特定的 MCP Server 子进程，以及和它之间的 JSON-RPC 通信。"""

    def __init__(self, server_name: str, command: str, args: list[str] | None = None,
                 env: dict[str, str] | None = None):
        # MCP Server 在配置中的名称，用于后续生成工具名前缀和路由工具调用。
        self.server_name = server_name
        # 启动 MCP Server 的命令，例如 `node`、`python`、某个可执行文件路径等。
        self.command = command
        # 启动命令附带的参数。
        self.args = args or []
        # 额外环境变量。连接时会和当前进程环境变量合并。
        self.env = env or {}
        # MCP Server 子进程对象。连接成功前为空。
        self._process: asyncio.subprocess.Process | None = None
        # JSON-RPC 请求 id 自增计数器，用于把请求和响应对应起来。
        self._next_id = 1
        # 正在等待响应的请求。key 是 JSON-RPC id，value 是 Future。
        self._pending: dict[int, asyncio.Future] = {}
        # 后台读取 stdout 的任务。MCP Server 的响应会从 stdout 持续读出。
        self._reader_task: asyncio.Task | None = None
        # 后台消费 stderr 的任务，避免 stderr 缓冲区写满阻塞 server。
        self._stderr_task: asyncio.Task | None = None
        self._reader_error: Exception | None = None

    async def connect(self) -> None:
        """启动一个 MCP Server 子进程，并安排两个后台任务，分别读取它的标准输出和错误输出。这里的连接，只是做了拉起进程、建立 stdio 管道、准备好 JSON-RPC 的读写流的工作
        关于协议里面的握手和工具发现，是在后续的 initialize() 和 list_tools() 中完成的。"""
        # 子进程环境变量 = 当前进程环境变量 + 配置里声明的额外变量。
        # 相同名称的变量，以后面的 self.env 为准。
        merged_env = {**os.environ, **self.env}
        self._reader_error = None
        # create_subprocess_exec() 用来启动外部程序(本质就是执行command args的指令尝试连接外部服务器)，并返回一个异步子进程对象，保存在 self._process 中
        # await等待子进程初始化完成
        self._process = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            # 使用 stdio 模式启动 MCP Server：
            # - stdin：客户端向 Server 写 JSON-RPC 请求。
            # - stdout：Server 向客户端返回 JSON-RPC 响应。
            # - stderr：保留错误输出管道，避免 Server 继承当前终端输出。
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
            # 在非 Windows 系统上创建新会话，使子进程与当前终端会话分离，方便后续按进程组管理
            start_new_session=os.name != "nt",
            # 设置异步输出读取流的缓冲限制为 MCP_MAX_MESSAGE_BYTES，避免单条消息过大导致阻塞。
            limit=MCP_MAX_MESSAGE_BYTES,
        )
        # 后台持续读取 stdout。这里不阻塞 connect()，否则后续无法继续初始化。
        # 设置两个事件循环任务：
        # 1. _read_loop()：持续读取 MCP Server stdout 中按行分隔的 JSON-RPC 响应。
        # 2. _drain_stderr()：持续消费 stderr，避免子进程输出过多后阻塞。
        # 这里使用asyncio.create_task()创建两个异步任务是避免循环阻塞后续的步骤执行，这种不会进行等待所以不会阻塞
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _read_loop(self) -> None:
        """持续读取 MCP Server stdout 中按行分隔的 JSON-RPC 响应。"""
        # 确保子进程已经启动，并且 stdout 管道可用。
        assert self._process and self._process.stdout
        try:
            while True:
                # MCP stdio 通常是一行一个 JSON-RPC 消息。
                # 持续从process进程的stdout管道中读取一行数据，
                # 暂时没有数据还是会异步等待的
                line = await self._process.stdout.readline()
                if not line:
                    # 读不到内容说明子进程 stdout 已关闭，通常表示进程退出。
                    break
                if len(line) > MCP_MAX_MESSAGE_BYTES:
                    raise ValueError(f"MCP message exceeds {MCP_MAX_MESSAGE_BYTES} bytes")
                try:
                    # 反序列化为 Python 对象。MCP Server 的响应通常是一个字典，包含 id
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # 非法 JSON 直接忽略，避免单条脏输出打断整个连接。
                    continue

                # JSON-RPC 响应会带 id；通知类消息通常没有 id。
                if not isinstance(msg, dict):
                    continue
                msg_id = msg.get("id")
                # 这表示这个任务的id，已经从mcp server的stdout管道中拿到了结果，
                # 相应的pending中这个id对应的future应该对他进行赋值
                if msg_id is not None and msg_id in self._pending:
                    fut = self._pending.pop(msg_id)
                    if "error" in msg:
                        # Server 返回 JSON-RPC error 时，把等待中的 Future 标记为异常。
                        e = msg["error"]
                        # 赋值前还是检查是否done了避免出现意外
                        if not fut.done():
                            fut.set_exception(
                                RuntimeError(f"MCP error {e.get('code')}: {e.get('message')}")
                            )
                    elif not fut.done():
                        # 正常响应只把 result 部分交给调用方。
                        fut.set_result(msg.get("result"))
        # 对异常捕获进行分类处理
        # 这里的所有意外都会导致所有等待的pending任务都会被标记为异常，避免调用方无限等待
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._reader_error = RuntimeError(f"MCP reader failed (message limit {MCP_MAX_MESSAGE_BYTES} bytes): {exc}")
            self._fail_pending(self._reader_error)
        else:
            self._reader_error = RuntimeError(f"MCP server '{self.server_name}' closed")
            self._fail_pending(self._reader_error)

    async def _drain_stderr(self) -> None:
        """持续消费 stderr，避免子进程输出过多后阻塞。(但是他对这个错误信息并没有特定的用处)"""
        # 确保子进程已经启动，并且 stderr 管道可用。
        assert self._process and self._process.stderr
        try:
            while True:
                # 每次最多读取通道中的 65536 字节
                chunk = await self._process.stderr.read(65536)
                if not chunk:
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    def _fail_pending(self, error: Exception) -> None:
        """任务关闭(可能是异常/也可能是正常退出)时，所有还没收到响应的请求都不可能再完成，需要显式置为异常。"""
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(error)
        self._pending.clear()

    async def _send_request(self, method: str, params: dict | None = None) -> Any:
        """发送一条 JSON-RPC request，并等待对应 id 的 response。"""

        # 首先先判断当前的mcp连接到现在位置是否已经有异常了，如果有异常就直接抛出异常，避免继续发送请求
        if self._reader_error is not None:
            raise RuntimeError(str(self._reader_error))
        assert self._process and self._process.stdin
        # 为本次请求分配唯一 id。读循环会用这个 id 找到对应 Future。
        req_id = self._next_id
        self._next_id += 1

        # JSON-RPC 2.0 请求格式：
        # {
        #   "jsonrpc": "2.0",
        #   "id": 1,
        #   "method": "...",
        #   "params": {...}
        # }
        msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
        # 获取当前线程的Event loop对象
        loop = asyncio.get_running_loop()
        # 定义一个Future对象绑定在当前线程上
        fut: asyncio.Future = loop.create_future()
        # 将这次的unique req_id跟fut匹配，表示这个线程执行这个request时，等待的结果会被放在这个fut上
        self._pending[req_id] = fut
        # stdio 传输使用换行作为消息边界。
        try:
            # 把请求数据交给输出缓冲区
            self._process.stdin.write((msg + "\n").encode())
            # drain()的作用是提供流量控制：
            # - 缓冲区没有触发暂停写入时，通常很快返回。
            # - 缓冲区积压较多、触发暂停写入时，等待数据被消耗到允许继续写入的程度
            await self._process.stdin.drain()
            # 等待这个请求的响应结果，fut会在_read_loop()中被赋值
            return await fut
        # 如果出现异常，说明这个请求无法继续等待结果了。如果同时fut还不是一个确定值，那么用cancel()取消对fut求值的过程
        except asyncio.CancelledError:
            self._pending.pop(req_id, None)
            if not fut.done():
                fut.cancel()
            raise
        except Exception:
            self._pending.pop(req_id, None)
            if not fut.done():
                fut.cancel()
            raise

    def _send_notification(self, method: str, params: dict | None = None) -> None:
        """发送一条 JSON-RPC notification。notification 没有 id，也不等待响应。所以不需要用_send_request()这一套逻辑，直接写入 stdin 即可。"""
        if not self._process or not self._process.stdin:
            return
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}})
        self._process.stdin.write((msg + "\n").encode())

    async def initialize(self) -> None:
        """执行 MCP 初始化握手。
        1、先进行 initialize 请求，协商协议版本和客户端信息。
        2、再发送 initialized 通知，告诉 MCP Server 初始化完成。
        """
        # initialize 是 MCP 连接建立后的第一步，用于协商协议版本和客户端信息。
        await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mini-claude", "version": "1.0.0"},
        })
        # 初始化请求成功后，按 MCP 协议发送 initialized 通知。这条通知只是单纯告知 MCP Server，客户端已经完成初始化，不需要等待响应。
        self._send_notification("notifications/initialized")

    async def list_tools(self) -> list[dict]:
        """从当前 MCP Server 查询可用工具列表。
        核心就是发送一个 tools/list 请求，然后把返回的工具列表进行包装，附加 serverName 方便上层路由。"""
        result = await self._send_request("tools/list")
        if not result or not isinstance(result.get("tools"), list):
            return []
        # 这里保留 MCP 原始 inputSchema，同时附加 serverName，方便上层做工具名加前缀和路由。
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": t.get("inputSchema"),
                "serverName": self.server_name,
            }
            for t in result["tools"]
        ]

    async def call_tool(self, name: str, args: dict) -> str:
        """调用当前 MCP Server 上的某个工具，并把结果转换成字符串返回。
        本质就是调用 tools/call 请求，param中插入name和args，然后把返回的 content block 列表拼接成字符串。"""
        result = await self._send_request("tools/call", {"name": name, "arguments": args})
        # 判断返回的tools/call调用是否是一个错误，如果是错误就抛出异常
        if isinstance(result, dict) and result.get("isError") is True:
            raise McpToolError(result)
        if isinstance(result, dict) and isinstance(result.get("content"), list):
            # MCP 工具结果通常是 content block 列表。当前 Agent 只消费 text block。
            text = "\n".join(
                c["text"] for c in result["content"] if c.get("type") == "text"
            )
            # text增加structuredContent结构化内容的json字符串
            if result.get("structuredContent") is not None:
                text += "\n" + json.dumps(result["structuredContent"], ensure_ascii=False)
            # 避免前面的content/structuredContent丢失信息，如果没有标准content list，就退回为 JSON 字符串
            return text or json.dumps(result, ensure_ascii=False)
        # 如果不是标准 content list，就退回为 JSON 字符串，避免丢失信息。
        return json.dumps(result)

    def close(self) -> None:
        """关闭 MCP Server 子进程，并让所有等待中的请求失败。就是把剩下在_pending中的future都标记为异常，避免调用方无限等待"""

        if self._reader_task:
            # 停止 stdout 后台读取任务。
            self._reader_task.cancel()
            self._reader_task = None
        if self._stderr_task:
            # 停止stderr后台读取任务。
            self._stderr_task.cancel()
            self._stderr_task = None
        if self._process:
            try:
                # 直接杀掉子进程，确保外部 MCP Server 不继续残留。
                if getattr(self._process, "stdin", None):
                    # 清除掉stdin管道，避免子进程继续阻塞等待输入
                    self._process.stdin.close()
                if hasattr(self._process, "kill"):
                    if os.name != "nt":
                        # 非 Windows 系统上，子进程会成为新会话和新进程组的首进程，其进程组编号通常就是它的 PID。
                        # 因此这里可以用它的 PID 定位进程组，终止 Server 以及仍属于该组的所有子进程。
                        os.killpg(self._process.pid, signal.SIGKILL)
                    else:
                        # windows直接把子进程杀死
                        self._process.kill()
            except ProcessLookupError:
                # 进程已经退出时 kill 可能抛出该异常，忽略即可。
                pass
            self._process = None
        # 连接关闭后，所有还没收到响应的请求都不可能再完成，需要显式置为异常。
        self._fail_pending(RuntimeError(f"MCP server '{self.server_name}' closed"))

    async def aclose(self) -> None:
        process = self._process
        # 这里先保存进程对象和两个后台任务的引用，因为接下来的 self.close() 会把对应实例属性设为 None。
        # 属性被清空后，局部变量仍然持有原来的对象，可以继续等待它们。
        tasks = [task for task in (self._reader_task, self._stderr_task) if task]
        self.close()
        # self.close()会下发这两个任务的cancel指令
        # 这行代码是等待两个读取任务都结束。因为cancel指令下发并不代表立刻会停止结束
        await asyncio.gather(*tasks, return_exceptions=True)
        if process:
            # 在MCP尝试连接生成的子进程尸体后，需要等待子进程退出，避免僵尸进程残留
            # 这里设置了5秒的超时，如果子进程在5秒内没有退出，就会抛出异常，避免无限等待
            await asyncio.wait_for(process.wait(), timeout=5)


# ─── MCP 管理器：统一管理多个 MCP Server 连接和工具路由 ─────────────────────────────


class McpManager:
    """
    管理所有 MCP Server 连接。

    使用方式：
    1. 调用 load_and_connect(authorize_start=...) 授权启动、连接 Server、发现工具。
    2. 调用 get_tool_definitions() 把 MCP 工具暴露给模型。
    3. 当模型调用 mcp__server__tool 形式的工具时，用 call_tool() 路由到对应 Server。
    """

    def __init__(self):
        # 已连接的 MCP Server。key 是 server name，value 是对应连接对象。
        self._connections: dict[str, McpConnection] = {}
        # 所有 MCP Server 发现出来的工具定义，保持接近 MCP 原始格式。
        self._tools: list[dict] = []
        # 防止重复连接。load_and_connect() 只应真正执行一次。
        self._connected = False

    async def load_and_connect(
        self, *, authorize_start: Callable[[str, dict], Awaitable[bool]] | None = None
    ) -> None:
        """
        在连接和发现工具之前，调用 authorize_start 回调授权启动每个 MCP Server。
        Authorize each configured process before connecting and discovering tools."""
        # 如果已经连接过，就不再重复连接。
        if self._connected:
            return
        self._connected = True

        # 合并全局、项目和 .mcp.json 配置。后读取的配置会覆盖同名 Server。
        # 本质就是读取当前所有的mcp的服务信息
        configs = self._load_configs()
        if not configs:
            return

        # 每个 Server 的初始化和工具发现最多等待 15 秒，避免坏配置长期卡住启动。
        timeout = 15.0

        for name, cfg in configs.items():
            conn = None
            try:
                # Configuration discovery must never implicitly execute project commands.
                # 统一格式化所有的cfg的配置信息，这样子后续便于统一尝试交互
                startup = {
                    "server_name": name,
                    # shlex: 一个用于简单类shell语法的词法分析器类。
                    # 提取command和args的组合，方便后续打印和授权
                    # 最后结果类似于：command: "python mcp_server.py --port 8080"
                    "command": shlex.join([cfg["command"], *(cfg.get("args") or [])]),
                    "cwd": str(Path.cwd()),
                }
                # 利用定义好的authorize_start回调函数，去看每一个配置的config是否处于allowed状态，如果不允许启动，就直接跳过这个配置
                if authorize_start is None or not await authorize_start("mcp_start", startup):
                    print_info(f"MCP startup denied: {name}")
                    continue
                # 初始化McpConnection对象，传入服务名、命令、参数和环境变量
                conn = McpConnection(name, cfg["command"], cfg.get("args"), cfg.get("env"))

                # 连接子进程 -> MCP 初始化握手 -> 查询工具列表。
                # conn.conect() 只是启动子进程和建立沟通管道
                await conn.connect()
                # 然后conn连接上server后进行握手，等待握手完成，确认对话双方的协议模式等更具体的实现信息
                await asyncio.wait_for(conn.initialize(), timeout=timeout)
                # 展示该mcp项下所有的tool_list
                server_tools = await asyncio.wait_for(conn.list_tools(), timeout=timeout)
                # 只有初始化和工具发现都成功，才登记为可用连接。
                # 把mcp名字和对应的连接对象存入_connections字典中，方便后续路由工具调用
                self._connections[name] = conn
                # 存储所有mcp的工具定义，方便后续获取工具定义和路由工具调用
                self._tools.extend(server_tools)
                print_info(f"MCP connected: {name} ({len(server_tools)} tools)")
            except asyncio.CancelledError:
                if conn is not None:
                    await conn.aclose()
                raise
            except Exception as e:
                # 单个 Server 失败不影响其他 Server。失败连接需要关闭以清理子进程。
                print_error(f"MCP failed to connect: {name}: {e}")
                if conn is not None:
                    await conn.aclose()

    def get_tool_definitions(self) -> list[dict]:
        """返回 Agent/Anthropic 可直接使用的工具定义，并给 MCP 工具名加前缀。
        为了区分 MCP 工具和内置工具，MCP 工具名会加上前缀 `mcp__serverName__toolName`。"""
        return [
            {
                # 前缀格式：mcp__服务名__工具名。
                # 这样可以避免 MCP 工具和内置工具重名，也方便 call_tool() 反向解析路由。
                "name": f"mcp__{t['serverName']}__{t['name']}",
                "description": t.get("description") or f"MCP tool {t['name']} from {t['serverName']}",
                # Anthropic 工具字段叫 input_schema；MCP 原始工具字段一般叫 inputSchema。
                "input_schema": t.get("inputSchema") or {"type": "object", "properties": {}},
            }
            for t in self._tools
        ]

    def is_mcp_tool(self, name: str) -> bool:
        """判断一个工具名是否是 MCP 工具名。所哟的mcp工具都会被设定为mcp__开头"""
        return name.startswith("mcp__")

    async def call_tool(self, prefixed_name: str, args: dict) -> str:
        """把带前缀的 MCP 工具调用路由到正确的 MCP Server。"""
        # 工具名格式为 mcp__serverName__toolName。
        # 获取真正的mcp对应的服务名+工具名
        parts = prefixed_name.split("__")
        if len(parts) < 3:
            raise ValueError(f"Invalid MCP tool name: {prefixed_name}")
        server_name = parts[1]
        # 工具名本身也可能包含 "__"，所以第 3 段之后要重新拼回去。
        tool_name = "__".join(parts[2:])  # tool name might contain __
        # 找当这个server name对应的mcp连接对象，如果没有连接就抛出异常
        conn = self._connections.get(server_name)
        if not conn:
            raise RuntimeError(f"MCP server '{server_name}' not connected")
        try:
            # 执行工具调用，设置超时时间为MCP_TOOL_TIMEOUT_SECONDS秒，如果超时就抛出异常
            return await asyncio.wait_for(
                conn.call_tool(tool_name, args), timeout=MCP_TOOL_TIMEOUT_SECONDS
            )
        # 如果超时，则抛出异常错误
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"MCP tool '{prefixed_name}' timed out after {MCP_TOOL_TIMEOUT_SECONDS:g} seconds"
            ) from exc

    async def disconnect_all(self) -> None:
        """断开所有 MCP Server 连接，并清空工具缓存。"""
        # 等待所有连接的关闭操作完成，确保子进程和后台任务都被清理干净
        await asyncio.gather(*(conn.aclose() for conn in self._connections.values()))
        self._connections.clear()
        self._tools.clear()
        self._connected = False

    # ─── 配置加载 ──────────────────────────────────────

    def _load_configs(self) -> dict[str, dict]:
        """
        按优先级加载并合并 MCP Server 配置。"""
        merged: dict[str, dict] = {}

        # 1. 全局配置：~/.bear/settings.json
        global_path = Path.home() / ".bear" / "settings.json"
        self._merge_config_file(global_path, merged)

        # 2. 当前项目配置：<cwd>/.bear/settings.json
        project_path = Path.cwd() / ".bear" / "settings.json"
        self._merge_config_file(project_path, merged)

        # 3. Claude Code 约定配置：<cwd>/.mcp.json
        mcp_json_path = Path.cwd() / ".mcp.json"
        self._merge_config_file(mcp_json_path, merged)

        return merged

    def _merge_config_file(self, path: Path, target: dict[str, dict]) -> None:
        """
        把单个配置文件里的 mcpServers 合并进 target。"""
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            # 支持两种格式：
            # 1. {"mcpServers": {"name": {...}}}
            # 2. {"name": {...}}

            # 如果找不到mcpServers的项，直接就返回raw本身，方便Claude Code约定的配置文件直接写成{"name": {...}}。
            servers = raw.get("mcpServers", raw)
            for name, config in servers.items():
                # 只接收包含 command 的对象；无效配置直接忽略。
                if isinstance(config, dict) and "command" in config:
                    target[name] = config
        except Exception:
            # 配置文件格式错误时跳过，避免一个坏配置导致整个 Agent 启动失败。
            pass
