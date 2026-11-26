#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import os
import re
import signal
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from agents.memory import get_memory_dir
from agents.tool_schemas import ToolDef, tool_definitions
from .tool_status import ToolResult

# 这里定义了具体的类型别名，类似于定义了只有一个属性的class（区别不大）
ToolInput = dict[str, Any]
SyncToolHandler = Callable[[ToolInput], str]
PermissionMode = Literal["default", "plan", "acceptEdits", "bypassPermissions", "dontAsk"]

READ_TOOLS = {"read_file", "list_files", "grep_search", "compact_context"}
EDIT_TOOLS = {"write_file", "edit_file", "skill_evolve", "skill_create", "skill_prune", "skill_restore", "state_cleanup"}


#并发安全的工具可以并行运行（只读，无副作用）
CONCURRENCY_SAFE_TOOLS = {"read_file", "list_files", "grep_search"}


#----------------------工具调用----------------------------

def _resolve_tool_path(raw_path: str) -> Path:
    """Preserve the requested destination, including missing absolute paths."""
    return Path(raw_path)


@dataclass(frozen=True)
class FileVersion:
    """这个类用来保存文件在某一时刻的状态快照"""
    device: int
    inode: int
    size: int
    modified_ns: int
    # Unix: metadata change time. Windows: creation time, shared by stat/fstat.
    changed_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> FileVersion:
        # Windows stat() can expose creation time as ctime while fstat() exposes
        # change time. Use birthtime for comparable snapshots on Python 3.12+;
        # older Windows versions of Python expose creation time through ctime.
        changed_ns = (getattr(value, "st_birthtime_ns", value.st_ctime_ns)
                      if os.name == "nt" else value.st_ctime_ns)
        return cls(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, changed_ns)


class _FileResult(ToolResult):
    """Attach the version of the actual open file without changing tool output."""
    def __new__(cls, text: str, path: Path, version: FileVersion):
        result = super().__new__(cls, text)
        result.path = str(path)
        result.version = version
        return result


MAX_READ_BYTES = 24 * 1024
MAX_READ_LINES = 1000
_UNTRACKED = object()


def _file_version(path: Path) -> FileVersion | None:
    try:
        return FileVersion.from_stat(path.stat())
    except FileNotFoundError:
        return None


def _read_page(handle, offset: int, limit: int) -> str:
    """Read a line window with bounded buffers, including a final empty line."""
    for _ in range(offset - 1):
        piece = handle.readline(MAX_READ_BYTES + 1)
        if not piece:
            return f"No lines found at offset {offset}."
        while not piece.endswith("\n"):
            piece = handle.readline(MAX_READ_BYTES + 1)
            if not piece:
                return f"No lines found at offset {offset}."

    numbered: list[str] = []
    remaining = MAX_READ_BYTES
    for number in range(offset, offset + limit):
        piece = handle.readline(MAX_READ_BYTES + 1)
        line = piece[:-1] if piece.endswith("\n") else piece
        encoded = line.encode("utf-8")
        prefix = f"{number:4d} | "
        cost = len(prefix.encode("utf-8")) + len(encoded) + 1
        if cost > remaining:
            if numbered:
                notice = f"Byte limit reached. Continue with read_file(offset={number}, limit={limit})."
            else:
                beginning = encoded[:max(0, remaining - len(prefix.encode("utf-8")) - 1)].decode("utf-8", errors="ignore")
                numbered.append(prefix + beginning)
                notice = (f"Line {number} exceeds the {MAX_READ_BYTES}-byte limit; only its beginning is shown. "
                          f"Use run_shell for a narrower character range within this line; "
                          f"read_file(offset={number + 1}, limit={limit}) continues with the next line.")
            return ("\n".join(numbered) + f"\n\n[Truncated: {notice} "
                    "Partial file: prefer edit_file to preserve unseen content.]")
        numbered.append(prefix + line)
        remaining -= cost
        if not piece.endswith("\n"):
            result = "\n".join(numbered)
            if offset > 1:
                result += "\n\n[Partial file: earlier lines were omitted. Prefer edit_file to preserve unseen content.]"
            return result
    return ("\n".join(numbered) +
            f"\n\n[Truncated: Line limit reached. Continue with read_file(offset={offset + limit}, limit={limit}). "
            "Partial file: prefer edit_file to preserve unseen content.]")


def _read_file(inp: ToolInput) -> str:
    """读取文件内容，返回一个str，包含文件内容和行号(有读取前后的状态校验,避免中间被修改)"""
    try:
        path = _resolve_tool_path(inp["file_path"]).resolve()
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            before_stat = os.fstat(handle.fileno())
            before = FileVersion.from_stat(before_stat)
            result = _read_page(handle, inp.get("offset", 1), inp.get("limit", MAX_READ_LINES))
            after_stat = os.fstat(handle.fileno())
            after = FileVersion.from_stat(after_stat)
        # Raw ctime remains comparable between two fstat() calls on one handle,
        # so retain that change detector alongside the portable path comparison.
        if (before != after or before_stat.st_ctime_ns != after_stat.st_ctime_ns
                or _file_version(path) != before):
            return "Error reading file: file changed during the read. Please read_file again."
        return _FileResult(result, path, before)
    except Exception as e:
        return f"Error reading file: {e}"


def _replace_file(path: Path, content: str, expected_version=_UNTRACKED) -> FileVersion:
    """
    他的本质涉及就是它和 atomic_write_text() 都采用“先写临时文件，再发布为目标文件”的思路。
    Stage content, then verify the observed version immediately before commit.
    检查目标文件版本
        ↓
    在同目录创建临时文件
            ↓
    写入完整新内容
            ↓
    再次检查目标文件版本
            ↓
    发布临时文件：新建或替换目标
            ↓
    返回新版本，清理临时文件
    """
    # 获取当前path下的文件状态信息，返回一个FileVersion对象，用于跟已有的expected_version进行比较，判断文件是否被外部修改过
    current = _file_version(path)
    if expected_version is _UNTRACKED:
        expected_version = current
    if current != expected_version:
        raise ValueError("file was modified externally; please read_file again before writing")

    path.parent.mkdir(parents=True, exist_ok=True)
    # 在path的同目录下创建一个临时文件，文件名为.{原文件名}.{uuid}.tmp，确保不会覆盖已有文件
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    # O_WRONLY	只写
    # O_CREAT	创建文件
    # O_EXCL	配合 O_CREAT，要求文件此前不存在
    # 0o666	创建时请求的权限；实际效果受系统及权限设置影响
    # 返回的 descriptor 是操作系统的文件描述符。
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            if current is not None:
                # 取得原文件的权限位，然后应用到临时文件
                mode = stat.S_IMODE(path.stat().st_mode)
                # os.fchmod是根据文件描述符来修改权限。这里优先使用os.fchmod，如果不支持则使用os.chmod
                if hasattr(os, "fchmod"):
                    os.fchmod(handle.fileno(), mode)
                else:
                    # 而os.chmod是根据路径来修改权限。
                    os.chmod(temporary, mode)
            handle.write(content)
            handle.flush()
            # 获取临时文件的文件状态(从文件描述符中获取的)
            staged_version = FileVersion.from_stat(os.fstat(handle.fileno()))
            if os.name == "nt":
                # Windows does not generally permit renaming an open file.
                # Its stat ctime is creation time and survives rename/link, so
                # retain the observed staging version instead of restatting
                # the destination and accidentally adopting an external edit.
                # Windows 分支先关闭临时文件
                # Windows 下，为了让后面的链接或替换操作顺利进行，这份实现先关闭临时文件句柄。
                # 关闭后，不能再通过这个 handle 读取它的状态
                handle.close()
            # 写好临时文件后还要而却到底检查目标文件的状态，确保在写入期间没有被外部修改过
            if _file_version(path) != expected_version:
                raise ValueError("file was modified externally; please read_file again before writing")
            # 如果没有预期版本，说明目标文件不存在，就直接创建一个硬链接到临时文件，并删除临时文件；否则就替换目标文件为临时文件
            if expected_version is None:
                # Link rather than replace so a file created since the check is
                # never silently overwritten. The staged file is in the same directory.
                # 直接把临时文件的内容硬链接到目标地址上
                # 临时文件名 .new.py.abc.tmp  ──┐
                #                             ├──→ 同一份新内容
                # 目标文件名 new.py   ──────────┘
                os.link(temporary, path)
                # 再删除临时文件，避免残留
                temporary.unlink()
            else:
                # 相反，直接替换目标文件为临时文件，确保原子性
                os.replace(temporary, path)
            # 最终获取目标文件的状态信息，返回一个FileVersion对象，本质就是获取原本临时文件内容句柄，然后再根据句柄获取的状态
            return staged_version if handle.closed else FileVersion.from_stat(os.fstat(handle.fileno()))
    finally:
        # 最终接触该地址连接
        temporary.unlink(missing_ok=True)


def _write_file(inp: ToolInput, expected_version=_UNTRACKED) -> str:
    """Write content to a file, creating it if necessary, and return the new version."""
    try:
        # 获取inp下面的绝对路径
        path = _resolve_tool_path(inp["file_path"]).resolve()
        # 利用路径/待修改内容/当前文件的状态信息进行文件写入操作，返回新的文件版本信息
        version = _replace_file(path, inp["content"], expected_version)
        # 如果更新的地址是memory相关的，就自动更新memory索引文件MEMORY.md，确保索引文件始终包含最新的memory信息
        _auto_update_memory_index(str(path))
        # 过去inp中的content内容切分后展示
        lines = inp["content"].split("\n")
        line_count = len(lines)
        preview = "\n".join(f"{i + 1:4d} | {l}" for i, l in enumerate(lines[:30]))
        trunc = f"\n  ... ({line_count} lines total)" if line_count > 30 else ""
        # 返回一个str，带着对应的path和文件状态信息
        return _FileResult(f"Successfully wrote to {inp['file_path']} ({line_count} lines)\n\n{preview}{trunc}", path, version)
    except Exception as e:
        return f"Error writing file: {e}"

def _auto_update_memory_index(file_path:str) -> None:
    """如果file_path是memory下的一个md文件，就自动更新memory索引文件MEMORY.md，确保索引文件始终包含最新的memory信息"""
    try:
        mem_dir = str(get_memory_dir())
        if file_path.startswith(mem_dir) and file_path.endswith(".md") and not file_path.endswith("MEMORY.md"):
            mem_path = Path(mem_dir)
            lines = ["# Memory Index", ""]
            for f in sorted(mem_path.glob("*.md")):
                if f.name == "MEMORY.md":
                    continue
                try:
                    # 获取每一个memory文件的内容，解析出name、type、description等信息，并添加到lines中
                    raw = f.read_text()
                    name_match = re.search(r"^name:\s*(.+)$", raw, re.MULTILINE)
                    type_match = re.search(r"^type:\s*(.+)$", raw, re.MULTILINE)
                    desc_match = re.search(r"^description:\s*(.+)$", raw, re.MULTILINE)
                    if name_match and type_match:
                        n = name_match.group(1).strip()
                        t = type_match.group(1).strip()
                        d = desc_match.group(1).strip()
                        lines.append(f"- **[{n}]({f.name})** ({t}) — {d}")
                except Exception:
                    pass
            (mem_path / "MEMORY.md").write_text("\n".join(lines))
    except Exception:
        pass

#-------------------------编辑助手，符号规范化+差异化------------------------

#将各种Unicode引号字符统一转换为标准的ASCII直引号。
def _normalize_quotes(s: str) -> str:
    """将各种Unicode引号字符统一转换为标准的ASCII直引号。"""
    s = re.sub("[\u2018\u2019\u2032]", "'", s)
    s = re.sub('[\u201c\u201d\u2033]', '"', s)
    return s

#查询匹配到的字符串
def _find_actual_string(file_content: str, search_string: str) -> str | None:
    """查询file_content中是否存在search_string"""
    # 首先先简单寻找，寻找不到再尝试归一化操作
    if search_string in file_content:
        return search_string
    norm_search = _normalize_quotes(search_string)
    norm_file = _normalize_quotes(file_content)
    idx = norm_file.find(norm_search)
    # 最后找到还是从原始的file_content中返回对应的字符串，确保返回的字符串是原始的，而不是归一化后的
    if idx != -1:
        return file_content[idx:idx + len(search_string)]
    return None

#生成一个简单的文本差异格式
def _generate_diff(old_content: str, old_string: str, new_string: str) -> str:
    """生成一个新旧文本修改的差异化展示"""
    before_change = old_content.split(old_string)[0]
    line_num = before_change.count("\n") + 1
    old_lines = old_string.split("\n")
    new_lines = new_string.split("\n")

    parts = [f"@@ -{line_num},{len(old_lines)} +{line_num},{len(new_lines)} @@"]
    for l in old_lines:
        parts.append(f"- {l}")
    for l in new_lines:
        parts.append(f"+ {l}")
    return "\n".join(parts)

#编辑文件
def _edit_file(inp: ToolInput, expected_version=_UNTRACKED) -> str:
    """编辑文件"""
    try:
        path = _resolve_tool_path(inp["file_path"]).resolve()
        if expected_version is _UNTRACKED:
            expected_version = _file_version(path)
        content = path.read_text(errors="replace")

        # 查找实际匹配的字符串(并非是归一化后的结果)，如果找不到就返回错误信息
        actual = _find_actual_string(content, inp["old_string"])
        if not actual:
            return f"Error: old_string not found in {inp['file_path']}"

        # Normalized matches must be unique in the same representation used to
        # find them, including mixed straight/curly quote spellings.
        # 这里的修改文本需要指定唯一性
        # 模型给出的旧文本不足以唯一确定修改位置。 此时应提供更多前后文，让 old_string 只匹配一处。
        occurrences = (content.count(actual) if actual == inp["old_string"] else
                       _normalize_quotes(content).count(_normalize_quotes(inp["old_string"])))
        if occurrences > 1:
            return f"Error: old_string found {occurrences} times in {inp['file_path']}. Must be unique."

        # 跟换后得到的新的内容
        new_content = content.replace(actual, inp["new_string"], 1)
        # 进行替换
        version = _replace_file(path, new_content, expected_version)
        # 生成一个简单的文本差异格式，展示修改前后的差异
        diff = _generate_diff(content, actual, inp["new_string"])

        quote_note = " (matched via quote normalization)" if actual != inp["old_string"] else ""

        return _FileResult(f"Successfully edited {inp['file_path']}{quote_note}\n\n{diff}", path, version)
    except Exception as e:
        return f"Error editing file: {e}"


def _list_files(inp: ToolInput, path_allowed: Callable[[Path], bool] | None = None) -> str:
    """查找inp["path"]下所有匹配inp["pattern"]的文件，返回一个str，包含所有匹配到的文件路径"""
    try:
        # 获取查找的基础路径，如果没有指定就默认是当前目录
        base = _resolve_tool_path(inp.get("path") or ".")
        # 判断该路径是否被允许访问，如果不允许就直接返回一个错误信息
        if path_allowed is not None and not path_allowed(base):
            return "No files found matching the pattern."
        # 获取查找的文件匹配模式
        pattern = inp["pattern"]
        files = []
        for p in base.glob(pattern):
            # 每个候选项都需要判断是否被允许访问，如果不允许就跳过
            if p.is_file() and (path_allowed is None or path_allowed(p)):
                # 获取相对于base的路径，如果base是当前目录就直接返回p，否则返回p相对于base的路径
                rel = str(p.relative_to(base) if base != Path(".") else p)

                # 过滤掉node_modules和.git目录下的文件，避免返回过多无关文件
                if "node_modules" in rel or ".git" in rel.split(os.sep):
                    continue
                files.append(rel)
                if len(files) >= 200:
                    break
        if not files:
            return "No files found matching the pattern."
        result = "\n".join(files[:200])

        if len(files) > 200:
            result += f"\n...and {len(files) - 200} more files are found ..."
        return result
    except Exception as e:
        return f"Error listing files: {e}"


def _grep_search(inp: ToolInput, path_allowed: Callable[[Path], bool] | None = None) -> str:
    """在inp["path"]下查找所有匹配inp["include"]的文件，返回一个str，包含所有匹配到的文件路径和行号"""
    # One regex implementation avoids platform-dependent grep syntax and lets
    # us reject denied files before opening them, including glob discoveries.
    return _grep_python(inp["pattern"], str(_resolve_tool_path(inp.get("path") or ".")),
                        inp.get("include"), path_allowed)


def _grep_python(pattern: str, directory: str, include: str | None,
                 path_allowed: Callable[[Path], bool] | None = None) -> str:
    """在directory下查找所有匹配include的文件，返回一个str，包含所有匹配到的文件路径和行号
    params:
    - pattern: 搜索文件内容的正则表达式
    - directory: 搜索的根目录
    - include: 文件名匹配模式，支持通配符
    - path_allowed: 一个函数，判断当前路径是否被允许访问，如果不允许就跳过
    """
    try:
        # 编译正则表达式，如果有错误就返回一个错误信息
        regex = re.compile(pattern)
    except re.error as exc:
        return f"Error searching files: invalid regular expression: {exc}"

    matches: list[str] = []

    def search_file(path: Path) -> None:
        # 如果指定了include参数，就判断当前文件名是否匹配include模式，如果不匹配就直接返回
        if include and not fnmatch.fnmatch(path.name, include):
            return
        # 如果指定了path_allowed参数，就判断当前文件是否被允许访问，如果不允许就直接返回
        if path_allowed is not None and not path_allowed(path):
            return
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.rstrip("\n")
                    # 用regex搜索当前行，如果匹配就把路径、行号和行内容加入matches列表，如果matches长度超过200就停止搜索
                    if regex.search(line):
                        matches.append(f"{path}:{line_number}:{line}")
                        if len(matches) >= 200:
                            break
        except OSError:
            return

    root = Path(directory)
    # 如果path_allowed不为None，就判断root是否被允许访问，如果不允许就直接返回一个错误信息
    if path_allowed is not None and not path_allowed(root):
        return "No matches found."
    # 如果是文件,直接开始搜寻
    if root.is_file():
        search_file(root)
    else:
        # Prune excluded directories before visiting them. Directory symlinks
        # are not followed because they may point back to an ancestor.
        # os.walk()会遍历目录树，返回一个三元组(parent, directories, files)
        # parent       当前遍历到的目录
        # directories 当前目录下的子目录名称列表
        # files        当前目录下的文件名称列表
        for parent, directories, files in os.walk(root, followlinks=False):
            # 过滤掉以.开头的隐藏目录和node_modules目录，以及不被允许访问的目录
            directories[:] = [name for name in directories if not name.startswith(".") and name != "node_modules"
                              and (path_allowed is None or path_allowed(Path(parent) / name))]
            # 再对当前目录下的文件进行搜索
            for name in files:
                if not name.startswith(".") and name != "node_modules":
                    search_file(Path(parent) / name)
                if len(matches) >= 200:
                    break
            if len(matches) >= 200:
                break
    if not matches:
        return "No matches found."
    output = "\n".join(matches[:100])
    if len(matches) > 100:
        output += f"\n... and {len(matches) - 100} more matches"
    return output


#--------截断过长的工具调用结果

MAX_RESULT_CHARS = 50000


def _truncate_result(result:str) -> str:
    """截断过长的工具调用结果"""
    if len(result) <= MAX_RESULT_CHARS:
        return result
    keep_each = (MAX_RESULT_CHARS - 60) // 2
    return (
        result[:keep_each]
        +f"\n\n[... truncated {len(result) - keep_each * 2} chars ...]\n\n"
        +result[-keep_each:]
    )


#----------延迟工具激活---------------------------
_activated_tools: set[str] = set()


def reset_activated_tools() -> None:
    _activated_tools.clear()

def get_active_tool_definitions(all_tools: list[ToolDef] | None = None) -> list[ToolDef]:
    """
    获取当前所有活跃的ToolDef
    Return active API schemas without internal deferred-tool metadata."""
    tools = all_tools if all_tools is not None else tool_definitions
    active: list[ToolDef] = []
    for tool in tools:
        # 表示该工具是延迟激活的，且未被激活
        if tool.get("deferred") and tool["name"] not in _activated_tools:
            continue
        # 进行一个拷贝，避免索引导致修改
        schema = tool.copy()
        # 消除deferred标记，避免模型在系统提示词中看到该标记
        schema.pop("deferred", None)
        active.append(schema)
    return active

def get_deferred_tool_names(all_tools: list[ToolDef] | None = None) -> list[str]:
    """返回所有deferred tool的名称列表，供模型在系统提示词中使用。"""
    tools = all_tools if all_tools is not None else tool_definitions
    return [t["name"] for t in tools if t.get("deferred") and t["name"] not in _activated_tools]

#执行shell命令
async def _run_shell(inp: ToolInput) -> str:
    """异步执行一条 Shell 命令，收集标准输出和错误输出，判断是否成功；
    如果超时或任务被取消，就尝试终止命令及其子进程，并完成清理。

    检查超时时间
          ↓
    启动 Shell 子进程
          ↓
    等待命令结束，同时收集输出
          ↓
    根据退出码组织结果
          ↓
    执行 finally 清理，再返回"""
    # 启动后的 Shell 子进程对象。
    process = None
    # 读取输出、等待进程结束的异步任务。
    output_task = None
    try:
        # 获取指令的timeout参数，默认值为30000ms（30秒），并将其转换为秒
        timeout_ms = inp.get("timeout", 30000)
        # 转换成秒s
        timeout_s = timeout_ms / 1000
        if timeout_s <= 0:
            return "Error: timeout must be positive"
        # 创建一个子进程来执行shell命令，捕获其标准输出和标准错误输出
        process = await asyncio.create_subprocess_shell(
            inp["command"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # 在非 Windows 平台，这有助于将命令及其衍生进程组织到独立的进程组中，方便后面使用 os.killpg() 终止整组进程。
            start_new_session=os.name != "nt",
        )
        # process.communicate() 负责：
        # 1. 读取标准输出。
        # 2. 读取标准错误。
        # 3. 等待进程结束。
        # 4. 返回两个字节串：
        # 创建一个创建协程对象，并将其包装为一个任务，以便我们可以在后续使用 asyncio.wait_for() 来设置超时。
        output_task = asyncio.create_task(process.communicate())
        # 将communicate中读取到的标准输出和标准错误字节串解码为字符串，并处理可能的解码错误。
        stdout_bytes, stderr_bytes = await asyncio.wait_for(asyncio.shield(output_task), timeout_s)
        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        output = stdout or ""
        # 查看子进程的退出码,本质就是看inp["command"]到底有没有顺利被执行完成，如果不为0，表示命令执行失败，返回错误信息。
        if process.returncode != 0:
            stderr = f"\nStderr: {stderr}" if stderr else ""
            stdout = f"\nStdout: {stdout}" if stdout else ""
            return ToolResult(f"Command failed (exit code {process.returncode}){stdout}{stderr}", failed=True)
        # 如果顺利完成，但是还存在报错，就连带到stdout中一起返回
        if stderr:
            output += ("\n" if output else "") + "Stderr:\n" + stderr
        return ToolResult(output or "(no output)")
    except asyncio.TimeoutError:
        return ToolResult(f"Command timed out after {inp.get('timeout', 30000)}ms", failed=True)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return f"Error: {e}"
    finally:
        # 如果子进程存在，并且输出任务也存在，并且输出任务还没有完成，就尝试终止子进程及其子进程组。
        if process is not None and output_task is not None and not output_task.done():
            try:
                if os.name == "nt":
                    # 创建一个新的子进程来执行 taskkill 命令，强制终止指定 PID 的进程及其子进程。
                    # 参数含义：
                    # /PID	指定要终止的进程
                    # /T	同时终止其子进程
                    # /F	强制终止
                    # DEVNULL	丢弃 taskkill 自己的输出
                    killer = await asyncio.create_subprocess_exec(
                        "taskkill", "/PID", str(process.pid), "/T", "/F",
                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                    # 等待这个 taskkill 命令完成，确保子进程及其子进程组被终止。
                    await killer.wait()
                else:
                    # 向对应进程组发送强制终止信号，尝试同时结束命令衍生的进程。
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await output_task
            await process.wait()


#危险命令检测模式列表

DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s"),
    re.compile(r"\bgit\s+(push|reset|clean|checkout\s+\.)"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s"),
    re.compile(r">\s*/dev/"),
    re.compile(r"\bkill\b"),
    re.compile(r"\bpkill\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\bdel\s", re.IGNORECASE),
    re.compile(r"\brmdir\s", re.IGNORECASE),
    re.compile(r"\bformat\s", re.IGNORECASE),
    re.compile(r"\btaskkill\s", re.IGNORECASE),
    re.compile(r"\bRemove-Item\s", re.IGNORECASE),
    re.compile(r"\bStop-Process\s", re.IGNORECASE),
]
def is_dangerous(command: str) -> bool:
    return any(p.search(command) for p in DANGEROUS_PATTERNS)

#权限规则
def _parse_rule(rule: str) -> dict:
    """
    权限规则：每次捕获类似于 tool_name(pattern) 的形式，返回一个字典，包含tool和pattern两个键值对。
    如果没有匹配，则世界返回一个字典，tool为rule本身，pattern为None。
    """
    m = re.match(r"^([A-Za-z0-9_-]+)\((.+)\)$", rule)
    if m:
        return {"tool": m.group(1), "pattern": m.group(2)}
    return {"tool": rule, "pattern": None}

def _load_settings(file_path: Path) -> dict | None:
    if not file_path.exists():
        return None
    try:
        return json.loads(file_path.read_text())
    except Exception:
        return None

_cached_rules: dict | None = None
_cached_rules_signature = None


def load_permission_rules() -> dict:
    """
    获取cwd下面的allow/deny的权限规则，需要复用user层面的权限规则，返回一个字典，包含allow和deny两个键值对，分别对应允许和拒绝的规则列表。
    """
    global _cached_rules, _cached_rules_signature
    from .project_trust import is_project_trusted, trust_file
    paths = [Path.home() / ".bear" / "settings.json", Path.cwd() / ".bear" / "settings.json",
             Path.cwd() / ".mcp.json", trust_file()]
    def signature(path):
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None
    current_signature = (str(Path.cwd().resolve()), *(signature(path) for path in paths))
    if _cached_rules is not None and _cached_rules_signature == current_signature:
        return _cached_rules

    allow: list[dict] = []
    deny: list[dict] = []

    user_settings = _load_settings(Path.home() / ".bear" / "settings.json")
    project_settings = _load_settings(Path.cwd() / ".bear" / "settings.json")

    # 从全局和当前项目设置中找到权限规则，并根据项目是否被信任来决定是否允许规则
    for settings, trusted in [(user_settings, True), (project_settings, is_project_trusted())]:
        if not settings or "permissions" not in settings:
            continue
        perms = settings["permissions"]
        if trusted:
            for r in perms.get("allow", []):
                allow.append(_parse_rule(r))
        # 收紧deny权限规则，保护项目安全
        for r in perms.get("deny", []):
            deny.append(_parse_rule(r))

    _cached_rules = {"allow": allow, "deny": deny}
    _cached_rules_signature = current_signature
    return _cached_rules


def _permission_paths(value: str) -> set[str]:
    """_permission_paths对value路径进行一系列转化，并统一写入返回值
    1、绝对化 + 纯字符串折叠 ./..（不碰 symlink）
    2、解开 symlink 后的绝对路径
    3、前缀（祖先）换成解析后的真实位置，后缀（祖先之下的部分）保留原样字面

    PS：需要第三点操作是因为 只 resolve 前缀、保留后缀字面 → 得到 /real/private/link，保留了中间链接路线，使基于"别名前缀"写的规则也能命中"""
    lexical = os.path.normcase(os.path.abspath(value))
    paths = {lexical}
    try:
        # Resolve the original path before collapsing '..': alias/../secret
        # follows the symlink first, which can differ from abspath(value).
        path = Path(value)
        if not path.is_absolute():
            path = Path.cwd() / path
        paths.add(os.path.normcase(str(path.resolve())))
        # Preserve intermediate symlink routes: private/link -> public.txt
        # must still match private/* when private itself is reached by an alias.
        for parent in path.parents:
            resolved = parent.resolve()
            if resolved != parent:
                paths.add(os.path.normcase(os.path.abspath(resolved / path.relative_to(parent))))
    except (OSError, RuntimeError):
        pass
    return paths


def _permission_patterns(pattern: str) -> tuple[set[str], bool]:
    """
    _permission_patterns:主要是为了实现查询pattern地址下的所有“标准化”的地址（通过_permission_paths）
    只有pattern以"/*"结尾时才会直接对标准化的路径添加分割符，避免后续最后的路径出现部分匹配的意外情况
    """
    prefix = pattern.endswith("*")
    raw = pattern[:-1] if prefix else pattern
    # 检验当前的raw是否以分隔符(sep)结尾
    trailing_separator = raw.endswith((os.sep, os.altsep or os.sep))
    paths = _permission_paths(raw)
    # 当且仅当 模式既是"末尾 * 前缀" 且 去 * 后以分隔符结尾时，给集合里每个路径统一补上结尾 /（若还没有）
    # 这是为了保证后续跟pattern比较的话可以直接用startswith来判断是否匹配上，不加分隔符可能最后的路径时部分匹配，不符合预期
    if prefix and trailing_separator:
        paths = {path if path.endswith(os.sep) else path + os.sep for path in paths}
    return paths, prefix


def _path_matches_pattern(value: str, pattern: str) -> bool:
    """
    判断"要操作的文件路径"value是否符合"规则里写的路径模式"pattern。具体而言就是分别对value和pattern进行_permission_paths处理（本质将路径展开成几种等价写法），除此之外pattern还需要更新为路径最后补上分隔符（如果没有的话）
    按照以下逻辑判断：
    1、如果pattern是"*"，表示规则容忍所有路径，value写什么都可以，返回True
    2、如果pattern是以"*"结尾，此时只要存在一个value的表示形式的前缀是以pattern中去掉"*"的部分开头的表现形式中之一，就代表value被pattern包含，返回True
    3、如果pattern是精确路径，那么就需要value的表示形式中有一个跟pattern的某一个表示形式完全相同，才能表示副歌pattern要求，返回True
    4、如果以上都不满足，则返回False"""
    if pattern == "*":
        return True
    patterns, prefix = _permission_patterns(pattern)
    return any(candidate.startswith(normalized) if prefix else candidate == normalized
               for candidate in _permission_paths(value) for normalized in patterns)


def _discovery_path_filter(tool_name: str, permission_mode: PermissionMode) -> Callable[[Path], bool]:
    """为搜索、列文件等“发现文件”的操作，生成一个路径过滤函数，用来判断某个文件或目录是否可以访问。
    在项目中，它主要用于 grep_search 和 list_files，避免从一个允许访问的目录开始搜索，却把其中被禁止的文件也搜出来。"""
    if permission_mode == "bypassPermissions":
        return lambda path: True
    # 收集deny规则中名称为tool_name或read_file的规则，生成一个列表
    # 这里增加read_file是因为如果一个文件禁止读取，却允许 grep_search 搜索其正文，就可能通过搜索结果获取里面的内容。
    rules = [rule for rule in load_permission_rules()["deny"]
             if rule["tool"] in {tool_name, "read_file"}]
    # 规则rules中只要有一个参数是None/*，就直接返回一个结果为False的函数，这是因为None/*表示不允许访问任何路径，直接返回False即可
    if any(rule["pattern"] in {None, "*"} for rule in rules):
        return lambda path: False
    # 为所有的pattern参数路径进行转化
    # 收集到了所有规则的路径表示+到底是前缀匹配还是精确匹配
    # 后续给到allow函数使用
    patterns = [_permission_patterns(rule["pattern"]) for rule in rules]
    # 如果patterns为空，表示规则没有封锁任何一个地址，那么所有地址都可以通过，直接返回Ture即可
    if not patterns:
        return lambda path: True

    def allowed(path: Path) -> bool:
        # Checking lexical and resolved parents covers a denied directory
        # discovered from an allowed ancestor and symlinks crossing its boundary.
        # 收集多样化的path，包括：
        # 1、绝对化、规范化的路径。
        # 2、解析符号链接后的真实路径。
        # 3、保留部分中间链接路线的路径表示。
        # 4、收集路径自身及全部父目录。
        # 这些都获取是因为后续如果是精确比较的话，就需要我知道每一层的真实目录到底长什么样子才能保证能够比较出来
        candidates = {str(parent) for value in _permission_paths(str(path))
                      for parent in (Path(value), *Path(value).parents)}
        # 把patterns里面的每个路径和candidates中的每个路径，按照rule规则中的prefix要求来进行比较，如果有一个比对到了，说明这个path不应该被找到，返回false
        return not any(candidate.startswith(pattern) if prefix else candidate == pattern
                       for values, prefix in patterns for pattern in values for candidate in candidates)

    return allowed


def _matches_rule(rule: dict, tool_name: str, inp: dict) -> bool:
    """
    _matches_rule 根据rule中的tool是否跟tool_name匹配然后再根据rule["pattern"]中的信息是否能再inp中的特定信息匹配上，有返回True。
    匹配规则具体见函数详情。

    rule的基本格式为：{{"tool": "tool_name", "pattern": "pattern"},.....}  最开始原本的记录类似于tool_name(pattern)的形式，经过_parse_rule函数解析后变成了字典形式。

    demo：
    rule = {"tool": "run_shell", "pattern": "git*"}
    _matches_rule(rule, "run_shell", {"command": "git status"})   # True  (前缀 git)
    """
    if rule["tool"] != tool_name:
        return False
    # 如果没有pattern参数，直接放行
    if rule["pattern"] is None:
        return True

    value = ""
    if tool_name == "run_shell":
        value = inp.get("command", "")
    elif tool_name == "mcp_start":
        value = inp.get("server_name", "")
    elif tool_name in {"skill_create", "skill_evolve", "skill_restore", "skill_prune"}:
        value = inp.get("skill_name") or inp.get("name", "")
    elif "file_path" in inp:
        return _path_matches_pattern(inp["file_path"], rule["pattern"])
    elif tool_name in {"grep_search", "list_files"}:
        return _path_matches_pattern(inp.get("path") or ".", rule["pattern"])
    elif tool_name.startswith("mcp__") and isinstance(inp.get("command"), str):
        value = inp["command"]
    else:
        return False

    # 这里的本质就是希望pattern使用了跟rule["pattern"]一样的前置指令
    # 如果有额外参数，则先过滤掉；反之则需要完全匹配才能证明input命中了对应的rule
    pattern = rule["pattern"]
    if pattern.endswith("*"):
        return value.startswith(pattern[:-1])
    return value == pattern


def _check_permission_rules(tool_name: str, inp: dict) -> str | None:
    """检查当前tool_name在当前输入inp下是否被允许或拒绝，返回"allow"、"deny"或None"""
    # rule格式： {“deny”：{{“tool_name”: "XXX","pattern": "XXX"},...}, "allow": {{“tool_name”: "XXX","pattern": "XXX"},...}}
    rules = load_permission_rules()
    for rule in rules["deny"]:
        if _matches_rule(rule, tool_name, inp):
            return "deny"
    for rule in rules["allow"]:
        if _matches_rule(rule, tool_name, inp):
            return "allow"
    return None


def check_permission(
    tool_name: str,
    inp: dict,
    mode: PermissionMode = "default",
    plan_file_path: str | None = None,
) -> dict:
    """
    根据三个方面来判断时当前tool_name在inp下的权限如何：
    1、在不同mode情况下（）
    2、在tool_name和inp参数是否符合rule权限规则，是否在deny列表中，是否在allow列表中且inp符合要求
    3、是否在一串需要confirm的列表中

    Returns {"action": "allow"|"deny"|"confirm", "message": explain the reason if it is "deny"/show "confirm" information}"""
    if mode == "bypassPermissions":
        return {"action": "allow"}

    # 检查这个tool_name和当前的输入inp是否被权限规则允许或拒绝，如果被拒绝则直接返回deny
    rule_result = _check_permission_rules(tool_name, inp)
    if rule_result == "deny":
        return {"action": "deny", "message": f"Denied by permission rule for {tool_name}"}

    # 在plan模式下，本质上其实就是只能对文件进行编辑，所以可以看到，除了EDIT_TOOLS中的{"write_file", "edit_file"}的工具都会被拒绝，甚至连shell命令都不允许执行
    if mode == "plan":
        if tool_name == "mcp_start" or tool_name.startswith("mcp__"):
            return {"action": "deny", "message": "MCP tools are blocked in plan mode"}
        if tool_name in EDIT_TOOLS:
            file_path = inp.get("file_path") or inp.get("path")
            # 只有在tool_name符合条件，plan_file_path存在，且file_path包含于plan_file_path的情况下，才允许对文件进行编辑
            if (tool_name in {"write_file", "edit_file"} and plan_file_path
                    and _path_matches_pattern(file_path, plan_file_path)):
                return {"action": "allow"}
            return {"action": "deny", "message": f"Blocked in plan mode: {tool_name}"}
        if tool_name == "run_shell":
            return {"action": "deny", "message": "Shell commands blocked in plan mode"}

    if rule_result == "allow":
        return {"action": "allow"}

    if tool_name in READ_TOOLS:
        return {"action": "allow"}

    if tool_name in ("enter_plan_mode", "exit_plan_mode"):
        return {"action": "allow"}

    if tool_name.startswith("mcp__"):
        # dontAsk自动拒绝模式
        if mode == "dontAsk":
            return {"action": "deny", "message": "Auto-denied (dontAsk mode): MCP tool call"}
        # 其他模式下，MCP工具调用需要确认
        return {"action": "confirm", "message": f"MCP tool call: {tool_name}\nArguments: {json.dumps(inp, ensure_ascii=False, sort_keys=True)}"}

    # "acceptEdits"模式下，EDIT_TOOLS中的工具调用会被直接允许
    if mode == "acceptEdits" and tool_name in EDIT_TOOLS:
        return {"action": "allow"}

    needs_confirm = False
    confirm_message = ""

    # A shell command can write files through an interpreter, pipe, or subprocess.
    # A blacklist cannot reliably distinguish read-only commands from writes.
    # 这一段是需要confirm的工具列表，都需要手动进行确认
    if tool_name == "run_shell":
        needs_confirm = True
        confirm_message = inp.get("command", "")
    elif tool_name == "mcp_start":
        needs_confirm = True
        confirm_message = f"start MCP server {inp.get('server_name', '')}: {inp.get('command', '')} (cwd: {inp.get('cwd', '')})"
    elif tool_name == "write_file":
        needs_confirm = True
        confirm_message = f"write file: {inp.get('file_path', '')}"
    elif tool_name == "edit_file":
        needs_confirm = True
        confirm_message = f"edit file: {inp.get('file_path', '')}"
    elif tool_name == "skill_evolve":
        needs_confirm = True
        confirm_message = f"evolve skill: {inp.get('skill_name', '')}"
    elif tool_name == "skill_create":
        needs_confirm = True
        confirm_message = f"create skill: {inp.get('name', '')}"
    elif tool_name in {"skill_prune", "skill_restore"}:
        needs_confirm = True
        confirm_message = f"{tool_name}: {inp.get('skill_name', '')}"
    elif tool_name == "state_cleanup":
        needs_confirm = True
        confirm_message = f"delete old disposable artifacts: {json.dumps(inp, ensure_ascii=False)}"

    if needs_confirm:
        if mode == "dontAsk":
            return {"action": "deny", "message": f"Auto-denied (dontAsk mode): {confirm_message}"}
        return {"action": "confirm", "message": confirm_message}

    return {"action": "allow"}


#----------------执行工具调用-----------------------
# 'agent' 和 'skill' 这两个工具在 agent.py 中处理，以避免循环依赖。"

def _tool_search(inp: ToolInput) -> str:
    # 获取当前输入参数中的query(工具查询的关键词)，并将其转换为小写
    query = (inp.get("query") or "").lower()
    # 查询所有tool_definitions中deferred为True的工具，并且工具名称或描述中包含query关键词的工具
    matches = [
        tool for tool in tool_definitions
        if tool.get("deferred")
        and (query in tool["name"].lower() or query in (tool.get("description") or "").lower())
    ]
    if not matches:
        return "No matching deferred tools found."

    # 放到_activated_tools中，表示这些工具已经被激活了
    _activated_tools.update(tool["name"] for tool in matches)
    # 格式化输出结果为JSON字符串，包含工具的名称、描述和输入参数的schema
    return json.dumps(
        [{"name": tool["name"], "description": tool.get("description", ""), "input_schema": tool["input_schema"]}
         for tool in matches],
        indent=2,
    )


def _skill_evolve(inp: ToolInput) -> str:
    """进化技能，返回进化后的技能信息。"""
    from .skills import evolve_skill

    result = evolve_skill(
        skill_name=inp.get("skill_name", ""),
        lesson=inp.get("lesson", ""),
        rationale=inp.get("rationale", ""),
        target=inp.get("target", "active"),
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def _skill_create(inp: ToolInput) -> str:
    """创建技能，返回创建后的技能信息。"""
    from .skills import create_skill

    result = create_skill(
        name=inp.get("name", ""),
        description=inp.get("description", ""),
        instructions=inp.get("instructions", ""),
        when_to_use=inp.get("when_to_use", "") or inp.get("when-to-use", ""),
        target=inp.get("target", "project"),
        context=inp.get("context", "inline"),
        user_invocable=bool(inp.get("user_invocable", False)),
        allowed_tools=inp.get("allowed_tools"),
        evidence=inp.get("evidence", ""),
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def _check_file_write_state(name: str, inp: ToolInput, read_file_state: dict[str, FileVersion]) -> str | None:
    """
    检查ToolInput参数中的当前文件是否可以被写入/编辑。主要判断当前文件是否先前被读取过(修改前必须先进行读取)，以及当前文件的状态是否与读取时的状态一致(防止外部修改)。
    Require a fresh read before replacing an existing file."""

    # 获取文件的绝对路径
    abs_path = str(_resolve_tool_path(inp["file_path"]).resolve())
    try:
        # 获取当前绝对路径下文件的状态对象version(FileVersion类型)
        version = _file_version(Path(abs_path))
    except OSError as exc:
        return f"Error checking file before {name}: {exc}"

    if version is None:
        return None
    verb = "writing" if name == "write_file" else "editing"
    # 如果这个文件信息都没有记录到read_file_state中，表明从未读取过。那么就无法进行写入操作了
    if abs_path not in read_file_state:
        return f"Error: You must read this file before {verb}. Use read_file first to see its current contents."
    # 如果当前asb_path下的文件状态于记录在read_file_state[abs_path]中的状态不一致，说明文件在外部被修改过了，那么就无法进行写入操作了
    if version != read_file_state[abs_path]:
        return ToolResult(f"Warning: {inp['file_path']} was modified externally since your last read. Please read_file again before {verb}.", failed=True)
    return None


async def execute_tool(
    name: str, inp: ToolInput, read_file_state: dict[str, FileVersion] | None = None, *,
    truncate_result: bool = True, permission_mode: PermissionMode = "default",
) -> str:
    """进行tool调用，返回结果字符串。对于read_file/write_file/edit_file工具，read_file_state用于跟踪文件版本以防止覆盖外部修改。

    params:
    name: 工具名称
    inp: 工具输入参数
    read_file_state: 涉及到文件的的相关状态信息
    truncate_result: 是否截断结果
    permission_mode: 权限模式，默认"default"，可选"plan"、"acceptEdits"、"dontAsk"、"bypassPermissions"
    """
    from .validation import validate_json

    # 判断当前工具调用是否在tool_definitions中
    definition = next((tool for tool in tool_definitions if tool["name"] == name), None)
    if definition is not None:
        # 校验输入参数是否符合工具定义的input_schema
        error = validate_json(inp, definition["input_schema"])
        if error:
            return f"Error: invalid arguments for {name}: {error}"

    # 如果是编辑/写文件，前提是需要掌握文件的状态信息
    if name in {"write_file", "edit_file"} and read_file_state is not None:
        # 检查当前文件是否可以被写入/编辑，主要判断当前文件是否先前被读取过，以及当前文件的状态是否与读取时的状态一致
        # 检查不通过则报错
        error = _check_file_write_state(name, inp, read_file_state)
        if error:
            return error

    # 工具搜索
    if name == "tool_search":
        # Return the complete schema payload even when ordinary output is capped.
        return _tool_search(inp)

    # Shell命令执行
    if name == "run_shell":
        # Shell execution must remain awaitable so cancellation kills descendants.
        result = await _run_shell(inp)
    # 执行写入/编辑文件
    elif name in {"write_file", "edit_file"} and read_file_state is not None:
        # 根据工具名称选择相应的处理函数
        path = str(_resolve_tool_path(inp["file_path"]).resolve())
        handler = _write_file if name == "write_file" else _edit_file
        result = handler(inp, expected_version=read_file_state.get(path))
    # 批量获取文件/搜索
    elif name in {"list_files", "grep_search"}:
        # 获取路径过滤器，判断是否被rules允许访问
        path_allowed = _discovery_path_filter(name, permission_mode)
        handler = _list_files if name == "list_files" else _grep_search
        # 启动一个新线程执行handler(inp, path_allowed)，并等待其完成，返回结果
        result = await asyncio.to_thread(handler, inp, path_allowed)
    else:
        # 根据工具名称选择相应的处理函数
        handlers: dict[str, SyncToolHandler] = {
            "read_file": _read_file,
            "write_file": _write_file,
            "edit_file": _edit_file,
            "list_files": _list_files,
            "grep_search": _grep_search,
            "skill_evolve": _skill_evolve,
            "skill_create": _skill_create,
        }
        handler = handlers.get(name)
        if handler is None:
            return f"Unknown tool: {name}"
        if name in CONCURRENCY_SAFE_TOOLS:
            # 因为是并发安全的,所以可以并发形式的在一个新线程中执行handler(inp)，并等待其完成，返回结果
            result = await asyncio.to_thread(handler, inp)
        else:
            # Do not leave a file mutation running in a worker after cancellation.
            # 反之只能在当前线程中执行handler(inp),返回结果
            result = handler(inp)

    if name in {"read_file", "write_file", "edit_file"} and read_file_state is not None:
        # 如果不清楚可以去看一下这三个函数,最终正常的都是会返回一个底层是str的_FileResult类
        if isinstance(result, _FileResult):
            # 因为都涉及到了文件,所以需要将当前文件的状态信息记录到read_file_state中,以便后续进行写入/编辑操作时进行检查
            read_file_state[result.path] = result.version
        elif name == "read_file":
            # 如果read_file没有正常返回_FileResult对象,表明读文件中出现了问题,那么这个对应的文件状态信息就不应该被记录到read_file_state中,所以需要将其从read_file_state中删除
            read_file_state.pop(str(_resolve_tool_path(inp["file_path"]).resolve()), None)
    # 按照要求对工具执行结果进行截断处理
    return _truncate_result(result) if truncate_result else result


def reset_permission_cache() -> None:
    global _cached_rules
    _cached_rules = None


