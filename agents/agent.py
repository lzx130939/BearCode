#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Awaitable, Any, TypedDict

import anthropic
import openai

from agents.mcp_client import BorrowedMcpManager, McpManager
from .background_jobs import BackgroundSkillTasks
from .tool_runtime import ToolExecution
from .context_runtime import ContextRuntime
from .streaming import StreamAttempts
from .checkpoints import ToolCheckpoints
from agents.memory import MemoryPrefetch
from agents.prompt import build_system_prompt
from agents.session_memory import (
    FOLD_COMPONENT_SYSTEMS,
    FOLD_SESSION_MEMORY_SYSTEM,
    build_anthropic_transcript,
    build_folding_user_prompt,
    build_openai_transcript,
    format_folded_memory,
    parse_folded_memory,
    parse_fold_component,
)
from agents.session import save_folded_session_memory, save_session, session_title, title_from_text
from .tool_status import tool_failed
from agents.subagent import get_sub_agent_config
from agents.tools import (
    FileVersion, ToolDef, tool_definitions, execute_tool, CONCURRENCY_SAFE_TOOLS,
    check_permission, get_active_tool_definitions,
)
from agents.ui import (
    print_info, print_divider, print_assistant_text, print_sub_agent_start, print_sub_agent_end,
    start_spinner, stop_spinner, print_cost, print_tool_result, print_confirmation, print_retry,
    print_error,
)


# 指数退避重试


def _is_retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status in (429, 503, 529):
        return True
    msg = str(error)
    if "overloaded" in msg or "ECONNRESET" in msg or "ETIMEDOUT" in msg:
        return True
    return False


def _safe_utf8_text(value: object) -> str:
    return str(value).encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_for_utf8(value: Any) -> Any:
    """用utf-8的格式去递归解析对象"""
    if isinstance(value, str):
        return _safe_utf8_text(value)
    if isinstance(value, list):
        return [_sanitize_for_utf8(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_utf8(item) for item in value)
    if isinstance(value, dict):
        return {
            _sanitize_for_utf8(key): _sanitize_for_utf8(item)
            for key, item in value.items()
        }
    return value


async def _with_retry(fn, max_retries: int = 3):
    """本质就是重复至多max_retries次执行fn函数"""
    for attempt in range(max_retries + 1):
        try:
            # 这里_with_retry的fn参数是attempt
            #         async def attempt():
            #             state = self._capture_stream_output()
            #             try:
            #                 return await request()
            #             except BaseException:
            #                 self._discard_stream_output(state)
            #                 raise
            return await fn()
        except Exception as error:
            if attempt >= max_retries or not _is_retryable(error):
                raise
            delay = min(1000 * (2 ** attempt), 30000) / 1000 + (hash(str(time.time())) % 1000) / 1000
            status = getattr(error, "status_code", None) or getattr(error, "status", None)
            reason = f"HTTP {status}" if status else (getattr(error, "code", None) or "network error")
            print_retry(attempt + 1, max_retries, reason)
            await asyncio.sleep(delay)

MODEL_CONTEXT = {
    "claude-opus-4-6": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-haiku-4-5-20251001": 200000,
    "claude-opus-4-20250514": 200000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "deepseek-chat":200000
}

def _get_context_windows(model:str)->int:
    return MODEL_CONTEXT.get(model, 32000)


AUTO_COMPACT_THRESHOLD = 0.70


def _get_max_output_tokens(model: str) -> int:
    m = model.lower()
    if "opus-4-6" in m:
        return 64000
    if "sonnet-4-6" in m:
        return 32000
    if any(x in m for x in ("opus-4", "sonnet-4", "haiku-4")):
        return 32000
    return 16384

#转换tool的形式到openai
def _to_openai_tools(tools: list[ToolDef]) -> list[dict]:
    """转换tool的形式到openai"""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


class TokenUsage(TypedDict):
    input: int
    output: int


class RunResult(TypedDict):
    text: str
    tokens: TokenUsage


class Agent(BackgroundSkillTasks, ToolExecution, ContextRuntime, StreamAttempts, ToolCheckpoints):
    def __init__(self,
                 *,
                 permission_mode:str="default",
                 model:str="deepseek-chat",
                 api_base: str | None=None,
                 anthropic_base_url: str | None=None,
                 api_key: str | None=None,
                 api_protocol: str | None=None,
                 thinking: bool=False,
                 max_cost_usd: float | None=None,
                 max_turns: int | None=None,
                 max_output_tokens: int | None=None,
                 input_price_per_million: float=3.0,
                 output_price_per_million: float=15.0,
                 shutdown_timeout: float=5.0,
                 context_window: int | None=None,
                 background_skill_concurrency: int=2,
                 background_skill_queue_limit: int=16,
                 confirm_fn:Callable[[str], Awaitable[bool]] | None=None,
                 custom_system_prompt: str | None=None,
                 custom_tools: list[ToolDef] | None=None,
                 is_sub_agent: bool=False,
                 enable_folding: bool=True,
                 max_fold_limit: int | None=None,
                 fold_memory_mode: str="single_json",
                 auto_compact_threshold: float=AUTO_COMPACT_THRESHOLD,
                 enable_memory_prefetch: bool=True,
                 memory_recall_timeout: float=2.0,
                 enable_mcp: bool=True,
                 enable_online_evolution: bool=True,
                 fold_observer: Callable[[dict[str, Any]], None] | None=None,):
        if not 0 < auto_compact_threshold < 1:
            raise ValueError("auto_compact_threshold must be between 0 and 1")
        if max_fold_limit is not None and max_fold_limit < 0:
            raise ValueError("max_fold_limit must be non-negative")
        if fold_memory_mode not in {"single_json", "three_parallel"}:
            raise ValueError("fold_memory_mode must be single_json or three_parallel")
        if max_output_tokens is not None and max_output_tokens < 2:
            raise ValueError("max_output_tokens must be at least 2")
        if api_protocol not in {None, "openai", "anthropic"}:
            raise ValueError("api_protocol must be openai or anthropic")
        if any(not math.isfinite(price) or price < 0 for price in (input_price_per_million, output_price_per_million)):
            raise ValueError("token prices must be finite and non-negative")
        if not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be finite and positive")
        if max_cost_usd is not None and (not math.isfinite(max_cost_usd) or max_cost_usd < 0):
            raise ValueError("max_cost_usd must be finite and non-negative")
        if context_window is not None and (type(context_window) is not int or context_window < 1024):
            raise ValueError("context_window must be an integer of at least 1024 tokens")
        if max_turns is not None and (type(max_turns) is not int or max_turns < 0):
            raise ValueError("max_turns must be a non-negative integer")
        if min(background_skill_concurrency, background_skill_queue_limit) < 1:
            raise ValueError("background Skill limits must be positive")
        self.permission_mode = permission_mode
        """当前Agent的模式"""
        self.thinking = thinking
        """是否支持思考"""
        self.model = model
        self.use_openai = api_protocol == "openai" if api_protocol else bool(api_base)
        self.is_sub_agent = is_sub_agent
        """是否是子Agent"""
        self._api_base = api_base
        """openAI 专用的url"""
        self._anthropic_base_url = anthropic_base_url
        """claude 专用的url"""
        self._api_key = api_key
        self.enable_folding = enable_folding
        """是否压缩会话"""
        self.max_fold_limit = max_fold_limit
        """最大折叠次数"""
        self.fold_memory_mode = fold_memory_mode
        """折叠模式，"single_json"表示整个进行折叠，"three_parallel"表示并行折叠"""
        self.auto_compact_threshold = auto_compact_threshold
        """自动压缩阈值系数(0-1.0之间)"""
        if not math.isfinite(memory_recall_timeout) or memory_recall_timeout <= 0:
            raise ValueError("memory_recall_timeout must be finite and positive")
        self.memory_recall_timeout = memory_recall_timeout
        """表示记忆召回的超时时间"""
        self.enable_memory_prefetch = enable_memory_prefetch
        """表示是否开启了记忆预取功能，即在开始对话前先召回相关记忆"""
        self.enable_mcp = enable_mcp
        """是否启用MCP"""
        self.enable_online_evolution = enable_online_evolution
        """是否启用在线技能进化"""
        self.fold_observer = fold_observer
        available_tools = tool_definitions if custom_tools is None else custom_tools
        self.tools = [tool for tool in available_tools if enable_folding or tool["name"] != "compact_context"]
        self.max_cost_usd = max_cost_usd
        self.max_turns = max_turns
        """Agent 设置的最大轮数"""
        self.max_output_tokens = max_output_tokens
        """Agent 设置的最大输出token数"""
        self.input_price_per_million = input_price_per_million
        self.output_price_per_million = output_price_per_million
        self.shutdown_timeout = shutdown_timeout
        self.context_window = context_window or _get_context_windows(model)
        """上下文窗口大小"""
        self.background_skill_concurrency = background_skill_concurrency
        """定义后台技能任务执行的并发量"""
        self.background_skill_queue_limit = background_skill_queue_limit
        """后台技能队列数量限制"""
        self._background_semaphore = None
        """后台任务调度的信号量对象"""
        self._background_job_ids: set[str] = set()
        """后台任务的对应的文件名集合"""
        self._budget_parent: Agent | None = None
        """当前Agent的预算父级,用来指引找到根Agent(None表示自己就是根Agent)"""
        self._active_children: list[Agent] = []
        """当前激活的子代理列表"""
        self._memory_tasks: set[asyncio.Task] = set()
        """表示当前Agent中关于memory召回的任务集合"""
        self.confirm_fn = confirm_fn
        self._custom_system_prompt = custom_system_prompt
        """自定义系统提示词"""
        self.effective_window=self.context_window - min(20000, max(512, self.context_window // 4))
        """有效上下文窗口大小, 可以理解为一个阈值"""
        self.session_id = uuid.uuid4().hex[:8]
        """当前Agent的会话ID(猜测：这个项目每一次的启动都只能绑定唯一的session_id, 想要切换应该得重启一个新的进程指定对应的session_id)"""
        self._session_title = ""
        self._conversation_id = self.session_id
        self.session_start_time= time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())

        self.total_input_tokens = 0
        """总输入 Token 数"""
        self.total_output_tokens = 0
        """总输出 Token 数"""
        self.side_input_tokens = 0
        """侧边输入 Token 数(非主流程的辅助调用消耗，如 Skill 匹配、意图分类、摘要压缩、记忆检索等“幕后” LLM 调用)"""
        self.side_output_tokens = 0
        """侧边输出 Token 数"""
        self.last_input_token_count = 0
        """最近一次模型请求实际消耗的输入 token 数(这是llm返回的精确结果)"""
        self._last_request_bytes = None
        """上次发送前，程序计算的上下文大小(这是自己估计的模糊的)"""
        self.last_finish_reason: str | None = None
        """上一次对话的结束原因"""
        self.current_turns = 0
        """当前 Agent 对话的轮数计数器"""
        self.last_api_call_time = 0
        """上一次调用大模型的时间戳"""


        self._aborted = False
        """本轮对话的主动中止的标志。"""
        #存储异步任务
        self._current_task:asyncio.Task | None = None
        """存储的异步当前执行任务"""
        #权限白名单
        self._confirmed_paths: set[str] = set()


        # 计划模式”（Plan Mode）状态的变量
        self._pre_plan_mode: str | None=None
        self._plan_file_path: str | None=None
        """计划模式下唯一允许写的文件地址"""
        self._plan_approval_fn : Callable[[str], Awaitable[dict[str, Any]]] | None=None
        """计划审批同意函数(输入是str，输出是dict[str,Any]的异步函数)"""
        self._context_cleared : bool=False
        """计划模式下，是否已经清理了上下文"""

        #思考模式
        self._thinking_mode = self._resolve_thinking_mode()

        #子agent的输出缓存
        self._output_buffer: list[str] | None=None
        """用来接管输出，把原本要打印的文本收集起来"""
        self._turn_output_buffer: list[str] | None = None
        """本轮输出的累积缓冲（流式 token / 事件 / 工具调用记录等），用于本轮结束后的统一处理（如保存、重放、评测比对）"""

        # 编辑前读取
        self._read_file_state: dict[str, FileVersion] = {}
        """用于跟踪文件的版本信息，以防止在编辑或写入文件时覆盖外部修改。"""

        #MCP集成
        self._mcp_manager = McpManager()
        """mcp的管理器对象"""
        self._mcp_initialized = False

        # 本会话已经注入的 Memory 路径；随 Session 保存，用于恢复后的召回去重。
        self._already_surfaced_memories: set[str] = set()
        """本会话已呈现的 Memory 文件路径集合，并非 Memory 正文。"""
        self._session_memory_bytes = 0
        """本会话累计注入的 Memory 正文字节数，用于召回预算；不是进程内存或上下文总大小。"""

        #区分message的历史消息
        self._anthropic_messages: list[dict[str, Any]] = []
        self._openai_messages: list[dict[str, Any]] = []
        self._last_retrieved_skill_reference: dict[str, Any] | None = None
        """上一次检索到的技能引用"""
        self._last_retrieved_skill_hits: list[dict[str, Any]] = []
        """上一次检索到的技能命中记录"""
        self._pending_skill_extraction_window: dict[str, Any] | None = None
        """这里存储着还没有得到最终回复的窗口记录信息"""
        self._skill_eval_dialog: list[dict[str, str]] = []
        """技能评测的对话记录,存储技能评估参考的近期对话上下文"""
        self._background_skill_tasks: set[asyncio.Task] = set()
        """后台待执行的技能任务集合"""
        self._skill_feedback_outbox: list[dict] = []
        """待处理的技能反馈列表，主要暂存skill执行完之后已经补齐了feedback后的对话记录"""
        self._tool_checkpoints: dict[str, dict] = {}
        """工具检查点的字典，主要用记录工具的执行记录，为后续可能的回滚和恢复提供支持"""
        self._folded_session_memories: list[dict[str, Any]] = []
        """上下文折叠后保留的会话记忆组成一个历史列表"""
        self._fold_last_time: float = 0.0
        """上一次上下文折叠时间"""
        self._fold_count: int = 0
        """上下文折叠次数"""
        self._tool_error_streak: int = 0
        """连续工具调用失败次数"""
        self._same_tool_repeat_count: int = 0
        """同一工具重复调用次数"""
        self._last_tool_name: str = ""
        """上一次调用的工具名称"""

        #构建系统提示词
        self._base_system_prompt = custom_system_prompt or build_system_prompt()
        """Agent系构建系统提示词"""

        if self.permission_mode == "plan":
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
        else:
            self._system_prompt = self._base_system_prompt

        #初始化大模型客户端
        if self.use_openai:
            self._openai_client = openai.AsyncOpenAI(base_url=api_base, api_key=api_key)
            self._anthropic_client = None
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        else:
            kwargs : dict[str,Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if anthropic_base_url:
                kwargs["base_url"] = anthropic_base_url
            self._anthropic_client = anthropic.AsyncAnthropic(**kwargs)
            self._openai_client = None

        self._refresh_runtime_system_prompt()

    #判断返回模型的思考模式
    def _resolve_thinking_mode(self) -> str:
        if not self.thinking:
            return "disabled"
        if not self._model_supports_thinking():
            return "disabled"

        if self._model_supports_adaptive_thinking():
            return "adaptive"
        return "enabled"

    def _model_supports_thinking(self) -> bool:
        m = self.model.lower()
        if "claude-3-" in m or "3-5-" in m or "3-7-" in m:
            return False
        if "claude" in m and any(x in m for x in ("opus", "sonnet", "haiku")):
            return True
        return False
    def _model_supports_adaptive_thinking(self) -> bool:
        m = self.model.lower()
        return "opus-4-6" in m or "sonnet-4-6" in m

    #生成一个用于保存 AI 计划（Plan）的 Markdown 文件的绝对路径。
    def _generate_plan_file_path(self) -> str:
        """生成一个用于保存 AI 计划（Plan）的 Markdown 文件的绝对路径,每一个session都会生成一个唯一的plan文件"""
        d = Path.home() / ".bear" / "plans"
        d.mkdir(parents=True, exist_ok=True)
        return str(d / f"plan-{self.session_id}.md")

    def _build_plan_mode_prompt(self) -> str:
        return f"""

    # Plan Mode Active

    Plan mode is active. You MUST NOT make any edits (except the plan file below), run non-readonly tools, or make any changes to the system.

    ## Plan File: {self._plan_file_path}
    Write your plan incrementally to this file using write_file or edit_file. This is the ONLY file you are allowed to edit.

    ## Workflow
    1. **Explore**: Read code to understand the task. Use read_file, list_files, grep_search.
    2. **Design**: Design your implementation approach. Use the agent tool with type="plan" if the task is complex.
    3. **Write Plan**: Write a structured plan to the plan file including:
       - **Context**: Why this change is needed
       - **Steps**: Implementation steps with critical file paths
       - **Verification**: How to test the changes
    4. **Exit**: Call exit_plan_mode when your plan is ready for user review.

    IMPORTANT: When your plan is complete, you MUST call exit_plan_mode. Do NOT ask the user to approve — exit_plan_mode handles that."""

    #判断当前的任务所有的任务是否完成
    @property
    def is_processing(self)->bool:
        return self._current_task is not None and not self._current_task.done()

    # 大模型调用的工厂方法,构建一个用于记忆召回（memory recall）的 sideQuery 可调用对象，兼容anthropic, openai。
    def _build_side_query(self, *, max_tokens: int = 256, require_complete: bool = False, model_override: str | None = None):
        """
        大模型调用的工厂方法,构建一个用于记忆召回（memory recall）的 sideQuery 可调用对象，兼容anthropic, openai。
        该返回对象被调用后，会返回最大可能性的内容文本

        max_tokens:设置每次模型请求的输出 token 上限
        require_complete:检查模型是否以代码认可的正常结束状态返回，否则抛出异常
        model_override:指定用于 sideQuery 的模型名称，如果为 None，则使用当前 Agent 的默认模型
        """
        if self._anthropic_client:
            client = self._anthropic_client
            model = model_override or self.model
            async def _sq(system:str, user_message:str)->str:
                """system：系统提示词，描述角色、规则、任务要求。
                   user_message：本次需要处理的具体内容。"""
                # 首先先检查预算，如果超出预算则抛出异常
                self._require_model_budget()
                # 调用客户端的 messages.create 方法发送请求。
                resp = await client.messages.create(
                    model=model, max_tokens=max(1, int(max_tokens)), system=system,
                messages=[{"role": "user", "content": user_message}],
                )
                usage = getattr(resp, "usage", None)
                # 更新账单
                self.side_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                self.side_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
                # 检查是否正常结束
                if require_complete and getattr(resp, "stop_reason", None) != "end_turn":
                    raise ValueError("incomplete_evaluation_response")
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                if not text.strip():
                    block_types = [str(getattr(b, "type", "")) for b in getattr(resp, "content", [])]
                    logging.warning(
                        "side_query returned empty Anthropic-compatible response: model=%s stop_reason=%s content_block_types=%s",
                        getattr(resp, "model", model),
                        getattr(resp, "stop_reason", ""),
                        block_types,
                    )
                return text
            return _sq
        if self._openai_client:
            client = self._openai_client
            model = model_override or self.model
            async def _sq_openai(system:str, user_message:str)->str:
                self._require_model_budget()
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=max(1, int(max_tokens)),
                    # 跟Anthropic不同的是messages需要手动添加一下系统提示词
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],

                )
                usage = getattr(resp, "usage", None)
                self.side_input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.side_output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                # resp会按照先后概率返回预测的选择choice
                if not resp.choices:
                    logging.warning("side_query returned no OpenAI-compatible choices: model=%s", model)
                    return ""
                choice = resp.choices[0]
                if require_complete and getattr(choice, "finish_reason", None) != "stop":
                    raise ValueError("incomplete_evaluation_response")
                content = choice.message.content or ""
                if not content.strip():
                    logging.warning(
                        "side_query returned empty OpenAI-compatible response: model=%s finish_reason=%s message=%s",
                        model,
                        getattr(choice, "finish_reason", ""),
                        choice.message,
                    )
                return content
            return _sq_openai
        return None
    #异步任务取消（Abort）
    def abort(self) -> None:
        self._aborted = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    def set_confirm_fn(self, fn:Callable[[str], Awaitable[bool]]) -> None:
        self.confirm_fn = fn

    def set_plan_approval_fn(self, fn:Callable[[str], Awaitable[dict[str, Any]]]) -> None:
        self._plan_approval_fn = fn


    #计划模式开关（“状态切换与现场保护”机制）
    def toggle_plan_mode(self) -> str:
        """
        切换为"plan"模式，同时保留之前的模式和当前提示词。
        Toggle plan mode while preserving the prior mode and current prompt."""
        if self.permission_mode == "plan":
            self.permission_mode = self._pre_plan_mode or "default"
            self._pre_plan_mode = None
            self._plan_file_path = None
            self._system_prompt = self._base_system_prompt
            # 存放系统提示词的位置不同：OpenAI 路径把它放在第一个消息列表中，Anthropic 路径通过独立的 system 参数传递。
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] =self._system_prompt
            print_info(f"Exited plan mode -> {self.permission_mode} mode")
            return self.permission_mode
        else:
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path = self._generate_plan_file_path()
            # 更新为plan mode 系统提示词除了base prompt之外，额外增加一些plan mode的说明，比如_plan_file_path
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
            self._refresh_runtime_system_prompt()
            print_info(f"Entered plan mode. Plan file: {self._plan_file_path}")
            return "plan"

    def get_token_usage(self) -> TokenUsage:
        return {"input":self.total_input_tokens, "output":self.total_output_tokens}

    #主入口

    async def chat(self, user_message: str) -> None:
        """对话主入口，user_message:用户输入的消息"""
        # 生成对话标题
        if not self._session_title:
            self._session_title = title_from_text(user_message)
        # 本轮对话的"用户是否主动中止"标志。
        self._aborted = False
        self.last_finish_reason = None
        self._turn_output_buffer = []
        # 创建当前的对话任务
        self._current_task = asyncio.create_task(self._chat_turn(user_message))
        try:
            # 等待任务执行完成
            await self._current_task
        except asyncio.CancelledError:
            # 这里的意思是只有是用户手动停止所导致的异常才会被在这一层处理掉
            # 否则依然会raise到上层去
            user_abort = self._aborted
            self._aborted = True
            # User abort cancels only the inner task; caller cancellation (e.g.
            # an evaluation timeout or parent abort) must continue propagating.
            if not user_abort:
                raise
        finally:
            # 在对话结束后，修复被中断的工具调用，并清理当前任务和输出缓冲区
            self._repair_interrupted_tool_calls()
            self._current_task = None
            self._turn_output_buffer = None
            # 如果不是子Agent/任务
            if not self.is_sub_agent:
                # 关闭UI的旋转器，表示对话结束
                stop_spinner()
                # 对话结束后，自动保存当前会话的状态和数据
                self._auto_save()

    async def _chat_turn(self, user_message: str) -> None:
        """开始一轮对话
        整体的流程：
        1、先恢复后台的skill(演进/创建)任务，把符合要求的任务恢复到后台执行
        2、第一次chat的时候，如果启用了MCP服务，并且MCP服务还没有初始化，并且当前不是子Agent，并且不是plan模式下，就会去加载和连接MCP服务
        3、....
        """
        # 恢复后台技能任务
        self._resume_background_skill_jobs()
        #懒加载MCP服务在第一次chat的时候，即开启mpc/mcp未被初始化/不是子agent/不是plan模式下，才会去加载和连接MCP服务
        if self.enable_mcp and not self._mcp_initialized and not self.is_sub_agent and self.permission_mode != "plan":
            try:
                # 初始化MCP服务，尝试连接所有可以连接的mcp服务器
                await self._mcp_manager.load_and_connect(authorize_start=self.authorize_operation)
                mcp_defs = self._mcp_manager.get_tool_definitions()
                if mcp_defs:
                    # 工具中除了内置的一些基础工具外，还会把MCP服务中定义的工具也加入到Agent的工具列表中
                    self.tools = self.tools + mcp_defs
                self._mcp_initialized = True
            except asyncio.CancelledError:
                await self._mcp_manager.disconnect_all()
                raise
            except Exception as e:
                print_error(f"MCP init failed: {e}")

        # 开始一段对话前更新一下Agent信息
        original_user_message = _safe_utf8_text(user_message)
        # 因为是开始，所以没有上一次的概念，需要清空
        # 后续original_user_message会重新使用这两个字段
        self._last_retrieved_skill_reference = None
        self._last_retrieved_skill_hits = []

        # 这里排除子Agent是因为，子Agent只有在skill场景才会被调用，此时只需要用用到给他的prompt context进行处理即可，
        # 不需要再去做技能评测的记录和后台任务调度
        if not self.is_sub_agent:
            # 因为是新的一轮对话，所以需要清空上一次的技能评测对话记录，
            # 并且把当前输入作为feedback一起放入Agent的等待处理的对话材料列表
            self._capture_skill_feedback(original_user_message)
            # 把每一个附加了feedback的对话都启动一个evolution的后台任务，并清空_skill_feedback_outbox
            self._flush_skill_feedback()
            # user_message 已经加上了skill_hits对应的context内容
            # _last_retrieved_skill_reference本质上就是算法相似读计算得到的top[0]的dict对象，对象中增加了一个["all_hits"]=top的list对象
            # 同时_last_retrieved_skill_hits也跟随更新了
            user_message, self._last_retrieved_skill_reference = self._augment_user_message_with_skill_context(
                original_user_message
            )
        # 构建一个协程对象coro，决定是调用OpenAI的聊天接口还是Anthropic的聊天接口
        coro = self._chat_openai(user_message) if self.use_openai else self._chat_anthropic(user_message)
        await coro
        # 收集本轮对话的输出文本
        assistant_text = "".join(self._turn_output_buffer or []).strip()
        # 当一次对话结束后，如果不是子Agent，并且没有被中止，并且上一次的结束原因是stop或者end_turn，就会去调度一个后台技能任务来跟踪技能使用情况
        if not self.is_sub_agent and not self._aborted and self.last_finish_reason in {"stop", "end_turn"}:
            usage_hits = [dict(hit) for hit in self._last_retrieved_skill_hits]
            # 将本轮的技能使用情况记录下来
            self._schedule_background_skill_task(self._run_skill_usage_tracking(
                original_user_message, assistant_text,
                hits=usage_hits,
            ), job={"kind": "usage", "user": original_user_message, "assistant": assistant_text, "hits": usage_hits})
            # 一轮对话解释，产生的消息重新放回_pending_skill_extraction_window中
            self._set_pending_skill_extraction_window(
                original_user_message=original_user_message,
                assistant_text=assistant_text,
                retrieved_reference=self._last_retrieved_skill_reference,
            )
        if not self.is_sub_agent:
            print_divider()

    def _repair_interrupted_tool_calls(self) -> None:
        self._recover_tool_checkpoints()


   #执行一次对话，收集本轮模型输出文本，并返回本轮消耗的 token 数
    async def run_once(self, prompt: str) -> RunResult:
        """执行一次对话，收集本轮模型输出文本，并返回本轮消耗的 token 数"""
        self._output_buffer = []
        prev_in = self.total_input_tokens
        prev_out = self.total_output_tokens
        try:
            await self.chat(prompt)
            # chat() 方法执行完后，_turn_output_buffer 中会存储本轮对话的输出文本
            # 把 _turn_output_buffer 中的文本拼接成一个完整的字符串，并返回
            text = "".join(self._output_buffer)
        finally:
            self._output_buffer = None
        return {
            "text": text,
            "tokens":{
                "input":self.total_input_tokens-prev_in,
                "output":self.total_output_tokens-prev_out
            },
        }

    #输出工具：统一处理模型输出文本。根据当前是否处于“收集输出”的模式
    # 决定是把文本存进缓冲区，还是直接打印到终端。
    def _emit_text(self, text:str)->None:
        """输出工具：统一处理模型输出文本。根据当前是否处于“收集输出”的模式
        决定是把文本存进缓冲区，还是直接打印到终端。"""
        text = _safe_utf8_text(text)
        # 独立记录本轮输出
        if self._turn_output_buffer is not None:
            self._turn_output_buffer.append(text)
        # 决定输出去向：缓冲区或终端
        if self._output_buffer is not None:
            self._output_buffer.append(text)
        else:
            print_assistant_text(text)

    def _emit_thinking(self, text: str) -> None:
        """
        这个接口暂时没有实现，这个函数主要是预留给“思考内容展示”的独立通道
        Optional adapter hook; reasoning never enters answer/learning buffers."""
        pass

    def _build_fold_guidance_section(self) -> str:
        """构建折叠记忆后的提示词段落，包含当前上下文占用率、工具调用失败次数、重复工具调用次数、上一次折叠时间等信息"""
        if self._custom_system_prompt is not None or not self.enable_folding:
            return ""
        if self._fold_limit_reached():
            return "\n\n# Runtime Fold Guidance\n- Context folding limit reached. Continue without calling `compact_context`.\n"
        # 计算当前上下文的占用率，占用率 = 上一次输入的 token 数 / 有效上下文窗口大小
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0.0
        last_fold = "never" if not self._fold_last_time else f"{int((time.time() - self._fold_last_time) / 60)}m ago"
        return (
            "\n\n# Runtime Fold Guidance\n"
            f"- Current context utilization: {utilization:.0%}\n"
            f"- Recent tool error streak: {self._tool_error_streak}\n"
            f"- Same tool repeat count: {self._same_tool_repeat_count}\n"
            f"- Last fold: {last_fold}\n"
            "- If the context is getting long, the same tool is being retried without progress, or tool failures are accumulating, call `compact_context` before trying more tools.\n"
            "- If you folded very recently and the next step is clear, prefer continuing rather than folding again.\n"
        )

    def _refresh_runtime_system_prompt(self) -> None:
        """刷新运行时的系统提示词，可以发现只要系统提示词相关的状态发生变化，就会调用这个方法刷新系统提示词而系统提示词跟_base_system_prompt + 折叠记忆指导有关
        _base_system_prompt 跟memory、skills、deferred tool、subagent都有关"""
        if self._custom_system_prompt is not None:
            return
        self._base_system_prompt = build_system_prompt()
        if self.permission_mode == "plan":
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
        else:
            self._system_prompt = self._base_system_prompt
        # 添加记忆折叠后的指导
        self._system_prompt += self._build_fold_guidance_section()
        # 需要重新刷新折叠记忆之后对于折叠的提示词段落，确保系统提示词中包含最新的折叠记忆信息
        if self.use_openai and self._openai_messages:
            self._openai_messages[0]["content"] = self._system_prompt

    def _record_tool_outcome(self, tool_name: str, success: bool) -> None:
        """记录工具调用的结果，包括连续失败次数和重复调用次数"""
        if tool_name == self._last_tool_name:
            self._same_tool_repeat_count += 1
        else:
            self._same_tool_repeat_count = 1
        self._last_tool_name = tool_name
        if success:
            self._tool_error_streak = 0
        else:
            self._tool_error_streak += 1

    def _record_fold_event(self) -> None:
        """更新折叠事件后Agent的状态，包括折叠时间、折叠计数、工具错误计数和重复工具计数"""
        self._fold_last_time = time.time()
        self._fold_count += 1
        self._tool_error_streak = 0
        self._same_tool_repeat_count = 0
        self._last_tool_name = ""

    def _looks_like_tool_failure(self, tool_name: str, raw: str, result: str) -> bool:
        """检查工具调用的结果是否看起来像是失败了"""
        return tool_failed(tool_name, raw) or tool_failed(tool_name, result)

    def _augment_user_message_with_skill_context(self, user_message: str) -> tuple[str, dict[str, Any] | None]:
        try:
            from .skills import format_retrieved_skill_context

            # 返回已经格式化为上下文的被检索到的skill和最高相关度的skill引用
            context, top_ref = format_retrieved_skill_context(user_message, limit=3)
        except Exception:
            return user_message, None
        if top_ref and isinstance(top_ref.get("all_hits"), list):
            # 把所有命中的skill信息保存到_agent的_last_retrieved_skill_hits属性中
            self._last_retrieved_skill_hits = list(top_ref.get("all_hits") or [])
        if not context.strip():
            return user_message, top_ref
        return f"{user_message}\n\n{context}", top_ref

    def _strip_runtime_injections(self, text: str) -> str:
        """"把模型输出中可能包含的 <retrieved_skills> 标签剔除掉"""
        return re.sub(r"\n*<retrieved_skills>.*?</retrieved_skills>\s*", "", str(text or ""), flags=re.DOTALL).strip()

    def _message_text(self, msg: dict[str, Any]) -> str:
        """对message进行解析，提取出文本内容，剔除掉message中生成的<retrieved_skills>占位段"""
        content = msg.get("content")
        if isinstance(content, str):
            return self._strip_runtime_injections(content)
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    # 文本块
                    if block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                    # 带 content 的非工具块，排除tool_result / tool_use：：这两类是工具调用相关块，它们的 content 往往是工具输入/输出的结构化数据（JSON、结果对象），不是"对话文本"。
                    elif "content" in block and block.get("type") not in {"tool_result", "tool_use"}:
                        parts.append(str(block.get("content") or ""))
            return self._strip_runtime_injections("\n".join(parts))
        return ""

    def _recent_dialog_messages(self, *, max_messages: int = 8) -> list[dict[str, str]]:
        """返回最近的max_messages条对话消息列表，最多包含 max_messages 条消息，剔除掉非 user/assistant 消息"""
        raw_messages = self._openai_messages if self.use_openai else self._anthropic_messages
        out: list[dict[str, str]] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            text = self._message_text(msg)
            if text:
                out.append({"role": role, "content": text})
        return out[-max(2, int(max_messages)) :]

    async def authorize_operation(self, tool_name: str, inp: dict, *, interactive: bool = True) -> bool:
        """
        鉴权操作：把当前处于的相同权限应用到模型工具循环之外的操作上，即tool_name工具inp参数下是否能够执行操作。
        使用场景是user在cli中调用`/xxxx-xxxx`指令，需要先行判断这个指令+参数是否可以被允许
        Apply the same permission policy to operations outside the model tool loop."""
        permission = check_permission(tool_name, inp, self.permission_mode, self._plan_file_path)
        if permission["action"] == "allow":
            return True
        if permission["action"] == "confirm" and interactive and self.confirm_fn is not None:
            return await self._confirm_tool_call(tool_name, permission)
        return False

    async def _confirm_online_skill_write(self, tool_name: str, inp: dict) -> bool:
        return await self.authorize_operation(tool_name, inp)

    async def _confirm_background_online_skill_write(self, tool_name: str, inp: dict) -> bool:
        return await self.authorize_operation(tool_name, inp, interactive=False)

    def _online_evolution_enabled(self) -> bool:
        if not self.enable_online_evolution:
            return False
        raw = os.environ.get("BEAR_AUTO_SKILL_EVOLUTION", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    async def aclose(self) -> None:
        """
        关闭当前Agent，释放所有资源，包括后台技能任务、memory召回任务、MCP连接、大模型客户端等。
        Finish bounded background work and release every owned resource."""
        try:
            # 尝试在有限时间内关闭当前Agent下面的所有后台技能任务，如果超时则取消这些任务
            await self.drain_background_skill_tasks(cancel_on_timeout=True)
        finally:
            # 最终即使是未能即使完成的任务，也要尝试释放资源
            from .storage import write_json
            for path in self._background_jobs_dir().glob("*.json"):
                try:
                    job = json.loads(path.read_text())
                    # 如果记录的这个任务session是当前我的session且任务状态不是running，则释放这个任务的owner_pid
                    if job.get("owner_session") == self.session_id and job.get("status") != "running":
                        job["owner_pid"] = None
                        write_json(path, job)
                except (ValueError, OSError, AttributeError, TypeError):
                    logging.exception("Unable to release persisted Skill job %s", path)
            # 所有memory prefetch任务取消，并等待它们完成，随后清空
            for task in self._memory_tasks:
                task.cancel()
            await asyncio.gather(*self._memory_tasks, return_exceptions=True)
            self._memory_tasks.clear()
            try:
                await self._mcp_manager.disconnect_all()
            finally:
                # 最后关闭llm client，确保资源释放
                if self._openai_client:
                    await self._openai_client.close()
                if self._anthropic_client:
                    await self._anthropic_client.close()

    def _track_memory_prefetch(self, prefetch: MemoryPrefetch | None) -> None:
        """对prefetch句柄进行跟踪，确保在prefetch的task完成后从_memory_tasks中移除，并在prefetch失败时记录警告日志"""
        if prefetch:
            self._memory_tasks.add(prefetch.task)
            # 对句柄中的循环事件任务添加回调,保证在执行完成后从_memory_tasks中移除，并在prefetch失败时记录警告日志
            def done(task):
                self._memory_tasks.discard(task)
                if not task.cancelled() and task.exception():
                    logging.warning("Memory prefetch failed: %s", task.exception())
            prefetch.task.add_done_callback(done)

    def _pop_pending_skill_extraction_window(self, next_user_feedback: str) -> dict[str, Any] | None:
        """获取待处理的技能提取窗口，并将其从内部状态中移除，同时将用户的反馈信息添加到窗口中"""
        # 只有当前的对话结束，_pending_skill_extraction_window内容需要被清空，同时需要加上next_user_feedback信息，更新pending的内容信息
        pending = self._pending_skill_extraction_window
        self._pending_skill_extraction_window = None
        if not pending:
            return None
        messages = list(pending.get("messages") or [])
        feedback = _safe_utf8_text(next_user_feedback).strip()
        if feedback:
            # messages 更新用户的反馈信息
            messages.append({"role": "user", "content": feedback})
        # 只取最新的10条消息
        pending["messages"] = messages[-10:]
        # 更新用户的反馈信息到next_user_feedback字段上
        pending["next_user_feedback"] = feedback
        return pending

    def _set_pending_skill_extraction_window(
        self,
        *,
        original_user_message: str,
        assistant_text: str,
        retrieved_reference: dict[str, Any] | None,
    ) -> None:
        """重新设置_pending_skill_extraction_window"""
        if not original_user_message.strip() or not assistant_text.strip():
            return
        self._pending_skill_extraction_window = {
            "messages": self._recent_dialog_messages(max_messages=8),
            "latest_user": original_user_message,
            "latest_assistant": assistant_text,
            "retrieved_reference": self._compact_retrieved_reference(retrieved_reference),
            "session_id": self.session_id,
            "replay_context": {
                "schema_version": 2,
                "event_id": str(uuid.uuid4()),
                "session_id": getattr(self, "_conversation_id", self.session_id),
                "task_user": original_user_message,
                "context_messages": list(self._skill_eval_dialog),
                "original_response": assistant_text,
            },
        }
        # Keep original turns, without retrieved Skill text or tool messages.
        # 更新取最新的8条对话消息到_skill_eval_dialog中，确保_skill_eval_dialog中只保留最近的8条对话消息
        self._skill_eval_dialog = (self._skill_eval_dialog + [
            {"role": "user", "content": original_user_message},
            {"role": "assistant", "content": assistant_text},
        ])[-8:]

    def _compact_retrieved_reference(self, ref: dict[str, Any] | None) -> dict[str, Any] | None:
        if not ref:
            return None
        return {k: v for k, v in ref.items() if k != "all_hits"}

    async def _run_online_skill_evolution(self, window: dict[str, Any], *, interactive_confirm: bool = False) -> dict:
        """执行一次在线技能演变，window是待处理的技能提取窗口，interactive_confirm表示是否需要用户交互确认"""
        # 如果线上技能演变开关关闭，或者当前处于计划模式(计划模式只能编辑文件)，则直接返回错误信息
        if not self._online_evolution_enabled() or self.permission_mode == "plan":
            return {"ok": False, "error": "online skill extraction is disabled or blocked in plan mode"}
        messages = list(window.get("messages") or [])
        if not messages:
            return {"ok": False, "error": "no extraction messages"}

        # 构建一个 side_query 可调用对象，用于在技能演变过程中进行记忆召回。
        # 其中 max_tokens=2200 表示 side_query 的输出 token 上限，require_complete=True 表示要求 side_query 返回完整的响应。
        side_query = self._build_side_query(max_tokens=2200, require_complete=True)
        if side_query is None:
            return {"ok": False, "error": "no model client available"}

        try:
            from .online_skill_evolution import online_ingest
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        # 调用 online_ingest 函数执行在线技能演变，传入消息、回放上下文、side_query、检索到的参考信息、提示信息以及确认写入的回调函数。
        # online_ingest 从当前对话中提取技能，并交给技能管理流程处理”的入口函数
        result = await online_ingest(
            messages=messages,
            replay_context={**window.get("replay_context", {}),
                            "feedback": str(window.get("next_user_feedback") or "")},
            side_query=side_query,
            retrieved_reference=window.get("retrieved_reference") or None,
            hint=str(window.get("hint") or ""),
            confirm_write=self._confirm_online_skill_write if interactive_confirm else self._confirm_background_online_skill_write,
            target=os.environ.get("BEAR_AUTO_SKILL_TARGET", "project"),
            stage_only=True,
        )
        if result.get("ok"):
            if result.get("action") in {"add", "merge"}:
                self._refresh_runtime_system_prompt()
                print_info(f"Online skill {result.get('action')}: {result.get('skill')}")
            # 表示技能候选已经保存到暂存区，需要用户在后续使用 /skill-candidates 命令查看和确认
            elif result.get("action") in {"add_staged", "merge_staged"}:
                print_info(f"Skill candidate saved: {result.get('candidate_id')} ({result.get('skill')}). Review with /skill-candidates before /skill-apply.")
        elif result.get("action") not in {"add_denied", "merge_denied"}:
            print_error(f"Online skill evolution failed: {result.get('error') or result}")
        return result

    async def _run_skill_usage_tracking(self, original_user_message: str, assistant_text: str,
                                        *, hits: list[dict[str, Any]] | None = None) -> None:
        """_run_skill_usage_tracking这个函数是用来进行技能使用追踪的，
        其核心作用包括
            1、判断模型输出中的技能是否真的被使用；
            2、记录技能使用的判断结果；
            3、如果技能被归档，则刷新系统提示词，确保系统提示词中不再包含被归档的技能信息"""

        # 如果线上技能进化关闭/或者当前处于计划模式(计划模式只能编辑文件)，则直接返回错误信息
        if not self._online_evolution_enabled() or self.permission_mode == "plan":
            return {"ok": False, "error": "Skill usage tracking disabled"}
        # 如果没有提供 hits,则使用上一次检索到的技能命中记录
        # _last_retrieved_skill_hits都是每次chat_turn的时候根据original_user_message检索相似度高的skill，并不一定表示真的会被使用
        # 这也是后续需要judge_retrieved_skill_usage去判断是否真的被使用了
        hits = list(self._last_retrieved_skill_hits if hits is None else hits)
        # 如果都没有检索到的技能命中记录，或者模型输出为空，就没有后续继续判断的意义了
        # 因为后续需要判断模型输出中是否有技能使用，如果没有检索到的技能命中记录，或者模型输出为空，就无法判断技能使用情况
        # 则直接返回
        if not hits or not assistant_text.strip():
            return
        side_query = self._build_side_query(max_tokens=700)
        try:
            from .online_skill_evolution import judge_retrieved_skill_usage
            from .skills import record_usage_judgments

            # 用来判断每一个被检索到的技能命中记录是否真的被模型使用，
            # 并返回一个包含判断结果的列表
            judgments = await judge_retrieved_skill_usage(
                hits=hits,
                user_message=original_user_message,
                assistant_text=assistant_text,
                side_query=side_query,
            )
            # 检查是否有技能归档的权限
            permission = check_permission("skill_prune", {}, self.permission_mode, self._plan_file_path)
            # 根据上面这次对话以及被检索到的skill命中记录，记录技能使用的判断结果，更新技能使用的统计信息，并返回一个包含是否有技能被归档的结果
            result = record_usage_judgments(judgments, allow_prune=permission["action"] == "allow")
            if result.get("pruned"):
                # 因为归档，导致部分的技能被移除了，所以需要刷新系统提示词，确保系统提示词中不再包含被归档的技能信息
                self._refresh_runtime_system_prompt()
            return result
        except Exception:
            logging.exception("Skill usage tracking failed")
            return {"ok": False, "error": "Skill usage tracking failed"}

    async def extract_now(self, hint: str = "") -> dict[str, Any]:
        """立即从待处理的技能提取窗口中提取技能，hint是用户提供的提示信息"""
        # 获取待处理的技能提取窗口的信息
        # 这里使用_pending_skill_extraction_window而不是_skill_feedback_outbox，
        # extract_now是根据当前对话状态下记录的所有信息来进行提取，此时必然不可能有feedback信息，毕竟可能还没有完全完成对话
        # PS：真的想考虑feedback信息的话，这部分工作是在chat_turn中的_flush_skill_feedback()函数实现的
        pending = self._pending_skill_extraction_window
        if not pending:
            return {"ok": False, "error": "no pending online skill extraction window"}
        window = dict(pending)
        window["hint"] = hint
        result = await self._run_online_skill_evolution(window, interactive_confirm=True)
        if result.get("ok"):
            self._pending_skill_extraction_window = None
        # 因为_pending_skill_extraction_window参数是持久化在session文件中的，所以在技能提取完成后需要自动保存session文件
        self._auto_save()
        return result


    def clear_history(self) -> None:
        # 第一步：清空消息（保留系统提示词），重置上下文统计
        self._clear_history_keep_system()
        # 第二步：清空对话附带的临时状态
        self._skill_eval_dialog = []
        self._skill_feedback_outbox = []
        self._tool_checkpoints = {}
        self._pending_skill_extraction_window = None
        self._last_retrieved_skill_reference = None
        self._last_retrieved_skill_hits = []
        self._already_surfaced_memories.clear()
        self._session_memory_bytes = 0
        self._folded_session_memories = []
        print_info("Conversation cleared.")

    def show_cost(self):
        total = self._get_current_cost_usd()
        budget_info = f" / ${self.max_cost_usd} budget" if self.max_cost_usd else ""
        turn_info = f" | Turns: {self.current_turns}/{self.max_turns}" if self.max_turns else ""
        print_info(
            f"Tokens: {self.total_input_tokens + self.side_input_tokens} in / {self.total_output_tokens + self.side_output_tokens} out\n  Estimated cost: ${total:.4f}{budget_info}{turn_info}")

    #获取当前的花费，
    def _get_current_cost_usd(self) -> float:
        """计算当前会话的花费（以美元为单位）"""
        own = ((self.total_input_tokens + self.side_input_tokens) * self.input_price_per_million +
               (self.total_output_tokens + self.side_output_tokens) * self.output_price_per_million) / 1_000_000
        return own + sum(child._get_current_cost_usd() for child in self._active_children)

    def _require_model_budget(self) -> None:
        """查看预算，超出预算则报错"""
        budget = self._check_budget()
        if budget["exceeded"]:
            raise RuntimeError(f"Model request blocked: {budget['reason']}")

    #检查预算
    def _check_budget(self) -> dict:
        """检查开销和轮数是否超过预算，返回字典，包含是否超出预算和原因"""
        if self._budget_parent is not None:
            parent_budget = self._budget_parent._check_budget()
            if parent_budget["exceeded"]:
                return parent_budget
        if self.max_cost_usd is not None and self._get_current_cost_usd() >= self.max_cost_usd:
            return {"exceeded": True, "reason": f"Cost limit reached (${self._get_current_cost_usd():.4f} >= ${self.max_cost_usd})"}
        if self.max_turns is not None and self.current_turns >= self.max_turns:
            return {"exceeded": True, "reason": f"Turn limit reached ({self.current_turns} >= {self.max_turns})"}
        return {"exceeded": False}

    #压缩会话
    async def compact(self)->None:
        """压缩，前提是开启了enable_folding功能"""
        if not self.enable_folding:
            print_info("Context folding is disabled for this run.")
            return
        compacted = await self._compact_conversation(trigger="manual")
        if not compacted:
            print_info("Compaction skipped; conversation history was retained.")


    #恢复会话信息
    def restore_session(self, data:dict, *, restore_permission_mode: bool = True)->None:
        """恢复会话信息，主要是根据session中的runtimeState中的相关内容进行重置

        潜在可能data(session_file)结构如下：
        {
            "metadata": {
                "id": "session-20261104-001",
                "cwd": "C:\\Users\\luzixuan\\Desktop\\python\\BearCode",
                "apiProtocol": "openai"
            },
            # openAI协议不管是tool_calls还是tool_call_id都不会在content里面
            "openaiMessages": [
                {
                    "role": "system"/"assistant"/"user"/"tool",
                    "content": "我先读取文件。",
                    # "tool_call_id": "call_123",
                    # "tool_calls": [
                    #     {
                    #         "id": "call_123",
                    #         "type": "function",
                    #         "function": {
                    #             "id": "xxxx",
                    #             "name": "read_file",
                    #             "arguments": '{"path": "main.py"}'
                    #         }
                    #     },
                    #     {xxx,xxxx}
                    # ]
                },
                ...
            ],
            # Anthropic协议则是需要tool_use/tool_result放到content里面
            "anthropicMessages": [
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "我先读取文件。",
                    },
                    {
                        "type": "tool_use",
                        "id: "tool_123",
                        "name": "read_file",
                        "input": {"path": "main.py"}
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "xxxx",
                        "content": {xxx}
                    }
                ]
            ],
            "foldedSessionMemories": [],
            "runtimeState": {
                "conversationId": "conversation-20261104-001",
                "skillFeedbackOutbox": [],
                "toolCheckpoints": {
                    "call_id": {},
                    ...,
                },
                "skillEvalDialog": [
                {
                    "role": "user",
                    "content": "请分析这个项目，并给出修改计划。"
                },
                {
                    "role": "assistant",
                    "content": ["我会先检查项目结构，然后整理修改计划。"]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"content": "我会先检查项目结构，然后整理修改计划。", "xxx": "xxxx"},
                        {"xxx":xxxx}
                    ]
                }
            ],
            "pendingSkillExtractionWindow": null,
            "permissionMode": "plan",
            "planFilePath": "C:\\Users\\luzixuan\\.bear\\plans\\session-20261104-001.md"
            }
        }
        """

        # 先做前置校验
        if not isinstance(data, dict) or not isinstance(data.get("metadata"), dict):
            raise ValueError("Invalid session metadata")
        if not isinstance(data.get("runtimeState", {}), dict):
            raise ValueError("Invalid session runtime state")
        for key in ("openaiMessages", "anthropicMessages"):
            messages = data.get(key)
            if messages is not None and (not isinstance(messages, list) or any(not isinstance(msg, dict) for msg in messages)):
                raise ValueError(f"Invalid session message list: {key}")
        metadata = data.get("metadata") or {}
        if not metadata.get("cwd") or Path(metadata["cwd"]).resolve() != Path.cwd().resolve():
            raise ValueError("Session belongs to another project or has no project metadata")
        saved_protocol = metadata.get("apiProtocol") or ("openai" if data.get("openaiMessages") else "anthropic")
        if saved_protocol != ("openai" if self.use_openai else "anthropic"):
            raise ValueError("Session API protocol differs from the current configuration")
        state = data.get("runtimeState") or {}
        # 旧 Session 没有这两个字段时使用空状态；显式损坏的数据不能静默重置召回预算。
        surfaced_memories = state.get("alreadySurfacedMemories", [])
        memory_bytes = state.get("sessionMemoryBytes", 0)
        if not isinstance(surfaced_memories, list) or any(
                not isinstance(path, str) or not path.strip() or "\x00" in path for path in surfaced_memories):
            raise ValueError("Invalid session recalled memory paths")
        if type(memory_bytes) is not int or memory_bytes < 0:
            raise ValueError("Invalid session memory byte count")
        self._session_title = session_title(data)
        if data.get("anthropicMessages"):
            self._anthropic_messages = _sanitize_for_utf8(data["anthropicMessages"])
        if data.get("openaiMessages"):
            self._openai_messages = _sanitize_for_utf8(data["openaiMessages"])
        if isinstance(data.get("foldedSessionMemories"), list):
            self._folded_session_memories = _sanitize_for_utf8(data["foldedSessionMemories"])
        # 恢复召回管理状态，保留历史注入量；不按可能已修改或删除的 Memory 文件重算。
        self._already_surfaced_memories = set(surfaced_memories)
        self._session_memory_bytes = memory_bytes
        # 恢复技能反馈队列
        outbox = state.get("skillFeedbackOutbox", [])
        if not isinstance(outbox, list) or any(not isinstance(item, dict) for item in outbox):
            raise ValueError("Invalid session feedback outbox")
        self._skill_feedback_outbox = _sanitize_for_utf8(outbox)
        # 恢复工具检查点
        journal = state.get("toolCheckpoints", {})
        self._tool_checkpoints = {str(key): value for key, value in journal.items() if isinstance(value, dict)} if isinstance(journal, dict) else {}
        # 恢复tool的checkpoints，修复好最近一条有tool_call的message对应的tool_result回复，
        # 同时更新_recover_tool_checkpoints中对应tool_call的id的状态
        self._recover_tool_checkpoints()
        if not self.use_openai:
            # 上面顺便跟着checkpoints清理了最近一条tool_call可能的tool_result缺失的情况
            # 这次直接全量扫描，清除所有没有紧跟完整tool_result的tool_call消息和tool_result消息
            self._anthropic_messages = self._normalize_anthropic_messages(self._anthropic_messages)
        self._conversation_id = str(state.get("conversationId") or metadata.get("id") or self.session_id)
        dialog = state.get("skillEvalDialog")
        if isinstance(dialog, list):
            # 摘取当前session下面的运行状态runtimeState中的最后8条技能评估对话记录
            self._skill_eval_dialog = [dict(role=item["role"], content=_safe_utf8_text(item["content"]))
                for item in dialog if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
                and isinstance(item.get("content"), str)][-8:]
        else:
            self._skill_eval_dialog = self._recent_dialog_messages(max_messages=8)
        pending = state.get("pendingSkillExtractionWindow")
        self._pending_skill_extraction_window = _sanitize_for_utf8(pending) if isinstance(pending, dict) else None
        # restore_permission_mode参数用于控制是否在恢复会话时，permission_mode可以决定是否恢复成messages中的permissionMode状态
        if state.get("permissionMode") == "plan" and (self.permission_mode == "plan" or
                (restore_permission_mode and self.permission_mode == "default")):
            self.permission_mode = "plan"
            # 重新定义_plan_file_path
            candidate = state.get("planFilePath")
            plans_dir = (Path.home() / ".bear" / "plans").resolve()
            self._plan_file_path = (candidate if isinstance(candidate, str) and Path(candidate).is_absolute()
                                   and Path(candidate).resolve().parent == plans_dir else self._generate_plan_file_path())
            self._pre_plan_mode = "default"
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
        # Retain the current project's prompt/configuration on resume.
        # 重置OpeAI协议下的第一条messages的content信息，因为他是包含系统提示词的，恢复会话时需要重新设置为当前项目的系统提示词
        if self.use_openai and self._openai_messages and self._openai_messages[0].get("role") == "system":
            self._openai_messages[0]["content"] = self._system_prompt
        self._fold_count = len(self._folded_session_memories)
        print_info(f"Session restored ({self._get_message_count()} messages).")



#整理 Anthropic 的历史消息，修正部分角色错误，并丢弃不合法的工具调用消息。
    def _normalize_anthropic_messages(self, messages: list[dict]) -> list[dict]:
        """整理 Anthropic 格式的历史消息，修正部分角色，并返回剔除没有紧跟完整工具结果的工具调用消息。"""
        role_normalized = []
        for msg in messages:
            # 做一个浅拷贝，但是内部值信息共享
            copied = dict(msg)
            content = copied.get("content")
            # 如果一条消息同时满足：
            # - role 是 "user"；
            # - content 是列表；
            # - 列表中有字典块的 type 为 "tool_use"；
            # 就把角色改为 "assistant"
            if copied.get("role") == "user" and isinstance(content, list):
                if any(isinstance(block, dict) and block.get("type") == "tool_use" for block in content):
                    copied["role"] = "assistant"
            role_normalized.append(copied)

        normalized = []
        i = 0
        # 遍历 role_normalized 列表，检查每条消息是否有工具调用，并确保工具调用的结果紧跟在调用之后。
        while i < len(role_normalized):
            msg = role_normalized[i]
            # 找寻这个msg调用的tool_call的id集合
            tool_use_ids = self._anthropic_tool_use_ids(msg)
            # 如果是空的，就直接加入normalized
            if not tool_use_ids:
                normalized.append(msg)
                i += 1
                continue

            next_msg = role_normalized[i + 1] if i + 1 < len(role_normalized) else None
            result_ids = self._anthropic_tool_result_ids(next_msg) if next_msg else set()
            if tool_use_ids.issubset(result_ids):
                normalized.append(msg)
                normalized.append(next_msg)
                i += 2
                continue

            i += 1
        return normalized

    @staticmethod
    def _anthropic_tool_use_ids(msg: dict | None) -> set[str]:
        """获取 Anthropic 消息msg中工具的使用 ID
        """
        if not msg or msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            return set()
        return {
            block.get("id")
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
        }

    @staticmethod
    def _anthropic_tool_result_ids(msg: dict | None) -> set[str]:
        """获取 Anthropic 消息中工具的结果 ID
        并且Anthropic中tool调用的返回结果必须是从user侧这里出来的，注意openAI中tool的调用结果角色就是存粹的tool
        """
        if not msg or msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            return set()
        return {
            block.get("tool_use_id")
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id")
        }

    def _get_message_count(self) -> int:
        return len(self._openai_messages) if self.use_openai else len(self._anthropic_messages)

    def _auto_save(self, *, strict: bool = False) -> None:
        try:
            save_session(self.session_id, {
                "metadata": {
                    "id": self.session_id,
                    "title": self._session_title,
                    "model": self.model,
                    "apiProtocol": "openai" if self.use_openai else "anthropic",
                    "cwd": str(Path.cwd()),
                    "startTime": self.session_start_time,
                    "messageCount": self._get_message_count(),
                },
                "anthropicMessages": _sanitize_for_utf8(self._anthropic_messages) if not self.use_openai else None,
                "openaiMessages": _sanitize_for_utf8(self._openai_messages) if self.use_openai else None,
                "foldedSessionMemories": _sanitize_for_utf8(self._folded_session_memories),
                "runtimeState": _sanitize_for_utf8({
                    "conversationId": self._conversation_id,
                    "alreadySurfacedMemories": sorted(self._already_surfaced_memories),
                    "sessionMemoryBytes": self._session_memory_bytes,
                    "skillEvalDialog": self._skill_eval_dialog,
                    "pendingSkillExtractionWindow": self._pending_skill_extraction_window,
                    "skillFeedbackOutbox": self._skill_feedback_outbox,
                    "toolCheckpoints": self._tool_checkpoints,
                    "permissionMode": self.permission_mode,
                    "planFilePath": self._plan_file_path,
                }),
            })
        except Exception:
            logging.exception("Failed to save session %s", self.session_id)
            if strict:
                raise

    #自动压缩
    async def _check_and_compact(self)->None:
        """在没有退出、开启折叠、没有达到最大折叠次数、当前输入token数量大于有效窗口*自动压缩阈值的情况下，进行自动压缩"""
        # 判断在没有退出、开启折叠、没有达到最大折叠次数、当前输入token数量大于有效窗口*自动压缩阈值的情况下，进行自动压缩
        if not self._aborted and self.enable_folding and not self._fold_limit_reached() and self._estimated_input_tokens() > self.effective_window * self.auto_compact_threshold:
            print_info("Context window filling up, compacting conversation...")
            await self._compact_conversation(trigger="auto")

    async def _compact_conversation(self, *, trigger: str = "manual")->bool:
        """对话压缩"""
        if not self.enable_folding or self._fold_limit_reached():
            return False
        if self.use_openai:
            compacted = await self._compact_openai(trigger=trigger)
        else:
            compacted = await self._compact_anthropic(trigger=trigger)
        if compacted:
            print_info("Conversation compacted.")
        return compacted

    def _fold_limit_reached(self) -> bool:
        """检查是否达到最大折叠次数"""
        return self.max_fold_limit is not None and self._fold_count >= self.max_fold_limit

    async def _compact_anthropic(self, *, trigger: str)->bool:
        """压缩Anthropic协议下的对话"""

        if len (self._anthropic_messages)<4:
            return False

        transcript = build_anthropic_transcript(_sanitize_for_utf8(self._anthropic_messages))
        if not transcript.strip():
            return False
        memory = await self._generate_folded_session_memory(transcript)
        if memory is None:
            return False
        self._notify_fold_observer(trigger, transcript, memory)
        self._record_folded_session_memory(trigger, memory)
        self._record_fold_event()
        self._anthropic_messages = [{"role": "user", "content": format_folded_memory(memory)}]
        self.last_input_token_count = 0
        self._last_request_bytes = None
        self._refresh_runtime_system_prompt()
        return True

    async def _compact_openai(self, *, trigger: str)->bool:
        """压缩OpenAI协议下的对话"""

        # 长度小于4不需要压缩
        if len (self._openai_messages)<4:
            return False
        system_msg = self._openai_messages[0]
        transcript = build_openai_transcript(_sanitize_for_utf8(self._openai_messages))
        if not transcript.strip():
            return False
        # 生成折叠后的json格式记忆
        memory = await self._generate_folded_session_memory(transcript)
        if memory is None:
            return False
        # 通知折叠观察者
        self._notify_fold_observer(trigger, transcript, memory)
        # 更新折叠后的会话记忆列表，并持久化到硬盘中
        self._record_folded_session_memory(trigger, memory)
        # 更新折叠事件后的Agent状态，记录折叠事件
        self._record_fold_event()
        # 重置_openai_messages为系统提示词(可能是自定义的)+折叠后的记忆
        self._openai_messages=[
            system_msg,
            {"role": "user", "content": format_folded_memory(memory)},
        ]
        # 重置上一次输入token数量，以为上一次交流主要是为了压缩记忆而不是单纯的对话和上一次请求的字节数
        self.last_input_token_count=0
        self._last_request_bytes = None
        # 更新runtime下的系统提示词，确保系统提示词中包含了最新的折叠后的会话记忆
        # 这里本质上就是会把上面的self._openai_messages[0]进行覆盖修改，因为此时压缩后memory已经更新了
        self._refresh_runtime_system_prompt()
        return True

    async def _generate_folded_session_memory(self, transcript: str) -> dict[str, Any] | None:
        """生成一个折叠后的会话记忆"""
        try:
            # "three_parallel"这里这个表示会根据FOLD_COMPONENT_SYSTEMS中的三个系统分别去生成三份记忆，然后再合并成一个折叠后的记忆
            parallel = self.fold_memory_mode == "three_parallel"
            # 构建一个用于记忆召回（memory recall）的 sideQuery 可调用对象，兼容anthropic, openai。
            side_query = self._build_side_query(max_tokens=2500 if parallel else 6000, require_complete=True)
            if side_query is None:
                raise ValueError("No model client available for compaction")
            # 是否分别进行记忆压缩
            if parallel:
                prompt = "Extract the requested memory from this conversation.\n\nConversation transcript:\n" + transcript
                # 获取FOLD_COMPONENT_SYSTEMS中的所有的key名称
                names = tuple(FOLD_COMPONENT_SYSTEMS)
                # asyncio.gather() 并发执行这些调用，并等待结果。
                results = await asyncio.gather(
                    *(side_query(FOLD_COMPONENT_SYSTEMS[name], prompt) for name in names),
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
                memory = {name: parse_fold_component(name, raw) for name, raw in zip(names, results)}
                return parse_folded_memory(json.dumps(memory))
            # 进行总的记忆压缩
            raw = await side_query(FOLD_SESSION_MEMORY_SYSTEM, build_folding_user_prompt(transcript))
            return parse_folded_memory(raw)
        except Exception as exc:
            logging.warning("Compaction failed; retaining original conversation: %s", exc)
            return None

    def _notify_fold_observer(self, trigger: str, transcript: str, memory: dict[str, Any]) -> None:
        """通知折叠观察者，传递折叠记忆的触发类型、对话记录和折叠后的记忆"""
        if self.fold_observer is None:
            return
        try:
            self.fold_observer({
                "trigger": trigger,
                "fold_memory_mode": self.fold_memory_mode,
                "transcript": transcript,
                "memory": memory,
                # 把上次对话输入的token数量作为折叠前的输入token数量传递给观察者，方便观察者做统计或记录
                "pre_fold_input_tokens": self.last_input_token_count,
            })
        except Exception:
            logging.exception("fold observer failed")

    def _record_folded_session_memory(self, trigger: str, memory: dict[str, Any]) -> None:
        """每次记忆折叠后，更新上下文折叠后保留的会话记忆列表，同时持久化到硬盘中"""
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trigger": trigger,
            "fold_memory_mode": self.fold_memory_mode,
            "session_id": self.session_id,
            "pre_fold_input_tokens": self.last_input_token_count,
            "effective_window": self.effective_window,
            "folded_memory_characters": len(format_folded_memory(memory)),
            **memory,
        }
        # 把每一次的折叠后的会话记忆记录到self._folded_session_memories列表中，方便后续分析和回溯
        # 包含了每次的压缩信息和一些元数据，比如折叠触发类型、折叠前的输入token数量、有效窗口大小、折叠后的记忆字符数等
        self._folded_session_memories.append(record)
        try:
            # 持久化记忆到历史文件和latest文件中，方便后续分析和回溯
            save_folded_session_memory(self.session_id, _sanitize_for_utf8(record))
        except Exception:
            pass

    #大结果持久化
    #如果工具返回的结果太大（超过 30KB），不要硬塞进上下文里，而是把它存成一个临时文件。
    # 然后在对话里只留一个‘文件路径’和‘内容预览’。如果模型后面还需要看完整内容，它可以再次调用工具去读取这个文件

    def _persist_large_result(self, tool_name: str, result: str) -> str:
        from .result_storage import persist_tool_result
        return persist_tool_result(tool_name, result)["result"]

    #执行工具入口

    async def _execute_tool_call(self, name: str, inp: dict) -> str:
        """执行工具的入口函数"""

        # 检查工具是否可用，如果不可用则返回错误信息
        if name not in {tool["name"] for tool in self.tools} and not self._mcp_manager.is_mcp_tool(name):
            return f"Tool not available in this run: {name}"
        # 上下问压缩
        if name == "compact_context":
            return await self._execute_compact_context_tool(inp)
        # 进入plan mode/退出plan mode
        if name in ("enter_plan_mode", "exit_plan_mode"):
            return await self._execute_plan_mode_tool(name)
        # 启动子代理
        if name == "agent":
            return await self._execute_agent_tool(inp)
        # 执行skill工具，skill工具是通过技能管理器来执行的
        if name == "skill":
            return await self._execute_skill_tool(inp)
        # Route MCP tool calls to the MCP manager
        # 调用mcp管理器来执行mcp工具
        if self._mcp_manager.is_mcp_tool(name):
            # 利用_mcp_manager来调用工具，返回结果
            return await self._mcp_manager.call_tool(name, inp)
        # 如果不是上述的一些特殊的tool调用，都调用统一的tool call执行函数
        result = await execute_tool(name, inp, self._read_file_state, truncate_result=False,
                                    permission_mode=self.permission_mode)
        # 如果指令时技能创建/演化，则在成功之后进行system_prompt的刷新，确保系统提示词中包含最新的技能信息
        if name in {"skill_create", "skill_evolve"}:
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and parsed.get("ok"):
                    self._refresh_runtime_system_prompt()
            except Exception:
                pass
        return result

    async def _execute_compact_context_tool(self, inp: dict) -> str:
        """_execute_compact_context_tool是Agent根据prompt中调用compact_context工具时的执行函数，本质跟manual调用一样的"""
        if not self.enable_folding:
            return "Context folding is disabled for this run. Continue without compacting."
        if self._fold_limit_reached():
            return f"Context folding limit reached ({self.max_fold_limit}). Continue without compacting."
        reason = str(inp.get("reason") or "").strip()
        compacted = await self._compact_conversation(trigger="tool")
        if not compacted:
            self._record_tool_outcome("compact_context", False)
            return "Compaction skipped: insufficient history or memory generation failed validation. Original conversation history was retained."
        self._record_tool_outcome("compact_context", True)
        # 因为压缩,所以必然是清空了上下文的
        self._context_cleared = True
        suffix = f"\nReason: {reason}" if reason else ""
        return (
            "Context compacted into structured session memory. "
            "Continue from the folded memory now present in the conversation context."
            f"{suffix}"
        )


    async def _execute_skill_tool(self, inp: dict) -> str:
        """执行skill工具，主要包括了判断是否需要利用新的子Agent进行skill调用还是用当前的Agent进行执行"""
        from .skills import execute_skill
        # 保存技能调用记录，并返回解析后的 prompt 和执行配置。
        result = execute_skill(inp.get("skill_name", ""), inp.get("args", ""))

        if not result:
            return f"Unknown skill: {inp.get('skill_name', '')}"

        # 如果他是fork参数的话，就利用子Agent进行skill调用
        # 如果context内容选择的是inline，则表示直接在当前Agent中执行skill的prompt内容
        if result["context"] == "fork":
            allowed_tools = result.get("allowed_tools")
            # 筛选出符合allowed_tools的工具，如果allowed_tools为None，则表示允许所有工具
            tools = self.tools if allowed_tools is None else [
                tool for tool in self.tools if tool["name"] in allowed_tools
            ]
            # 生成一个子Agent，提供当前skill的prompt信息+获取的tool列表
            return await self._run_child_agent(
                agent_type="skill-fork",
                description=inp.get("skill_name", ""),
                system_prompt=result["prompt"],
                tools=tools,
                prompt=inp.get("args") or "Execute this skill task.",
                label="Skill fork",
                empty_result="(Skill produced no output)",
            )

        return f'[Skill "{inp.get("skill_name", "")}" activated]\n\n{result["prompt"]}'

    async def _execute_plan_mode_tool(self, name: str) -> str:
        """执行切换/修改plan模式工具，本质就是切换plan mode，然后更新Agent数据"""
        if name == "enter_plan_mode":
            if self.permission_mode == "plan":
                return "Already in plan mode."
            # 记录进入plan模式前的权限模式
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path =  self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info("Entered plan mode (read-only). Plan file: " + self._plan_file_path)
            return f"Entered plan mode. You are now in read-only mode.\n\nYour plan file: {self._plan_file_path}\nWrite your plan to this file. This is the only file you can edit.\n\nWhen your plan is complete, call exit_plan_mode."
        if name == "exit_plan_mode":
            # 修改前需要先进入plan mode，否则不允许退出plan mode
            if self.permission_mode != "plan":
                return "Not in plan mode."
            plan_content = "(No plan file found)"
            if self._plan_file_path and Path(self._plan_file_path).exists():
                try:
                    plan_content = Path(self._plan_file_path).read_text(encoding="utf-8")
                except Exception as exc:
                    plan_content = f"(Unable to read plan file: {exc})\n\nPlan file: {self._plan_file_path}"
            # 交互式审批流程（如果有审批函数）
            if self._plan_approval_fn:
                # 审批当前plan内容，决定到底该怎么做，按照choice进行返回特定结果
                result = await self._plan_approval_fn(plan_content)
                choice = result.get("choice") if isinstance(result, dict) else None
                if not isinstance(choice, str):
                    choice = None

                # 还需要进一步计划
                if choice =="keep-planning":
                    feedback = result.get("feedback") or "Please revise the plan."
                    return (
                        f"User rejected the plan and wants to keep planning.\n\n"
                        f"User feedback: {feedback}\n\n"
                        f"Please revise your plan based on this feedback. When done, call exit_plan_mode again."
                    )

                # 仅仅是计划，不需要任何修改
                if choice == "plan-only":
                    self._aborted = True
                    return (
                        f"Plan completed. One-shot plan mode does not execute changes.\n\n"
                        f"Plan file: {self._plan_file_path}\n\n"
                        f"## Plan:\n{plan_content}"
                    )

                # 如果审批函数返回的choice不在预期的选项中，则认为审批被取消或无效，保持在plan模式下，不执行任何操作
                if choice not in {"clear-and-execute", "execute", "manual-execute"}:
                    self._aborted = True
                    return "Plan approval cancelled or invalid. Remaining in plan mode; no implementation authorized."

                if choice == "clear-and-execute":
                    target_mode = "acceptEdits"
                elif choice == "execute":
                    target_mode = "acceptEdits"
                else:  # manual-execute
                    target_mode = self._pre_plan_mode or "default"

                #按照上面的评审结果，离开计划模式
                self.permission_mode = target_mode
                self._pre_plan_mode = None
                saved_plan_path = self._plan_file_path
                self._plan_file_path = None
                self._system_prompt = self._base_system_prompt
                self._refresh_runtime_system_prompt()
                if self.use_openai and self._openai_messages:
                    self._openai_messages[0]["content"] = self._system_prompt

                # 清空当前对话，并执行
                if choice == "clear-and-execute":
                    self._clear_history_keep_system()
                    self._context_cleared = True
                    print_info(f"Plan approved. Context cleared, executing in {target_mode} mode.")
                    return (
                        f"User approved the plan. Context was cleared. Permission mode: {target_mode}\n\n"
                        f"Plan file: {saved_plan_path}\n\n"
                        f"## Approved Plan:\n{plan_content}\n\n"
                        f"Proceed with implementation."
                    )
                print_info(f"Plan approved. Executing in {target_mode} mode.")
                return (
                    f"User approved the plan. Permission mode: {target_mode}\n\n"
                    f"## Approved Plan:\n{plan_content}\n\n"
                    f"Proceed with implementation."
                )
            self._aborted = True
            return "No plan approval handler available. Remaining in plan mode; no implementation authorized."

        return f"Unknown plan mode tool: {name}"

    def _clear_history_keep_system(self) -> None:
        """清空历史信息，但是保留系统prompt."""
        self._anthropic_messages = []
        self._openai_messages = []
        if self.use_openai:
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        self.last_input_token_count = 0
        self._last_request_bytes = None
        self._fold_last_time = 0.0
        self._fold_count = 0
        self._tool_error_streak = 0
        self._same_tool_repeat_count = 0
        self._last_tool_name = ""

    async def _execute_agent_tool(self, inp: dict) -> str:
        """执行Agent工具，本质就是启动一个子Agent，执行子Agent的任务，然后返回结果"""
        # 获取子Agent的配置，包括系统提示词和工具列表
        config = get_sub_agent_config(inp.get("type", "general"))
        return await self._run_child_agent(
            # agent类型(Agent内置的类型包括了plan/explore/general)
            agent_type=inp.get("type", "general"),
            # 问题秒胡
            description=inp.get("description", "sub-agent task"),
            # 系统提示词
            system_prompt=config["system_prompt"],
            # 可使用工具
            tools=config["tools"],
            # 提示词(这个类似于hits)
            prompt=inp.get("prompt", ""),
            label="Sub-agent",
            empty_result="(Sub-agent produced no output)",
        )

    def _create_child_agent(self, *, system_prompt: str, tools: list[ToolDef]) -> Agent:
        """
        为每个子节点一致地继承运行时限制和权限(这样子避免管理混乱)
        Inherit runtime limits and permissions consistently for every child."""
        # 黑名单，禁止子Agent使用的工具，避免子Agent进入无限循环调用父Agent的情况/避免子Agent进入计划模式/退出计划模式的情况
        forbidden = {"agent", "enter_plan_mode", "exit_plan_mode"}
        child_tools = [tool for tool in tools if tool["name"] not in forbidden]
        # 子Agent的本质就是继承了父Agent的大部分属性，除了系统提示词和工具列表
        child = Agent(
            model=self.model,
            api_base=self._api_base,
            anthropic_base_url=self._anthropic_base_url,
            api_key=self._api_key,
            api_protocol="openai" if self.use_openai else "anthropic",
            input_price_per_million=self.input_price_per_million,
            output_price_per_million=self.output_price_per_million,
            shutdown_timeout=self.shutdown_timeout,
            context_window=self.context_window,
            max_output_tokens=self.max_output_tokens,
            memory_recall_timeout=self.memory_recall_timeout,
            background_skill_concurrency=self.background_skill_concurrency,
            background_skill_queue_limit=self.background_skill_queue_limit,
            custom_system_prompt=system_prompt,
            custom_tools=child_tools,
            is_sub_agent=True,
            confirm_fn=self.confirm_fn,
            permission_mode=self.permission_mode,
            max_cost_usd=(max(0.0, self.max_cost_usd - self._get_current_cost_usd())
                          if self.max_cost_usd is not None else None),
            max_turns=(max(0, self.max_turns - self.current_turns)
                       if self.max_turns is not None else None),
            enable_folding=self.enable_folding,
            max_fold_limit=self.max_fold_limit,
            fold_memory_mode=self.fold_memory_mode,
            auto_compact_threshold=self.auto_compact_threshold,
            enable_memory_prefetch=self.enable_memory_prefetch,
            enable_mcp=self.enable_mcp,
            enable_online_evolution=self.enable_online_evolution,
        )
        inherited_mcp_tools = {tool["name"] for tool in child_tools if tool["name"].startswith("mcp__")}
        if inherited_mcp_tools:
            child._mcp_manager = BorrowedMcpManager(self._mcp_manager, inherited_mcp_tools)
            child._mcp_initialized = True
        return child

    async def _run_child_agent(self, *, agent_type: str, description: str,
                               system_prompt: str, tools: list[ToolDef], prompt: str,
                               label: str, empty_result: str) -> str:
        """根据配置信息运行一个子Agent，并返回结果"""
        # 第一步，检查预算是否超出，如果超出，则直接返回预算超出的信息
        budget = self._check_budget()
        if budget["exceeded"]:
            return f"{label} skipped: {budget['reason']}"
        # print 子Agent启动信息，包括agent类型和描述
        print_sub_agent_start(agent_type, description)
        child = None
        try:
            # 创建一个子Agent(重点需要prompt+tool信息)本质是相信Agent可以根据我们提供的提示词+工具做出正确的合适选择
            child = self._create_child_agent(system_prompt=system_prompt, tools=tools)
            # 并注册到父Agent的活跃子Agent列表中
            self._register_child(child)
            # 进行一次子Agent的运行，传入提示词，并等待结果
            result = await child.run_once(prompt)
            # 这个子Agent的结果进行返回，作为后续主Agent调用Agent这个tool的结果
            return result["text"] or empty_result
        except Exception as exc:
            return f"{label} error: {exc}"
        finally:
            try:
                if child is not None:
                    try:
                        # 关闭子Agent的所有资源连接(包括后台技能任务、召回记忆任务、mcp服务、llm client服务)
                        await child.aclose()
                    finally:
                        # 关闭之后统计子Agent的使用情况到父Agent中，包括token消耗和轮次
                        self._account_child_usage(child)
            finally:
                print_sub_agent_end(agent_type, description)

    def _register_child(self, child: "Agent") -> None:
        """注册一个子Agent，记录其父Agent，并加入活跃子Agent列表"""
        child._budget_parent = self
        self._active_children.append(child)

    def _account_child_usage(self, child: "Agent") -> None:
        """更新子Agent的相关使用情况到父Agent中，包括token消耗和轮次"""
        # Each child is newly created; settle even if it failed or was cancelled.
        # 父Agent的active_children列表中移除这个子Agent，并清空子Agent的父Agent引用
        self._active_children = [active for active in self._active_children if active is not child]
        child._budget_parent = None
        # 更新父Agent的token消耗和轮次统计信息
        self.total_input_tokens += child.total_input_tokens
        self.total_output_tokens += child.total_output_tokens
        self.side_input_tokens += child.side_input_tokens
        self.side_output_tokens += child.side_output_tokens
        self.current_turns += child.current_turns

#--------------Anthropic 后端---------------
    async def _chat_anthropic(self, user_message: str) -> None:
        # 整理 Anthropic 格式的历史消息，修正部分角色，并返回剔除没有紧跟完整工具结果的工具调用消息
        self._anthropic_messages = self._normalize_anthropic_messages(_sanitize_for_utf8(self._anthropic_messages))
        budget = self._check_budget()
        if budget["exceeded"]:
            self.last_finish_reason = "budget_exceeded"
            self._emit_text(f"Budget exceeded: {budget['reason']}")
            return
        user_message = _safe_utf8_text(await self._recall_before_response(user_message))
        # 先把本轮用户输入放入 Anthropic 消息历史，后续每轮模型调用都会带上这段上下文。
        self._anthropic_messages.append({"role": "user", "content": user_message})

        while True:
            # 外部请求中止时，结束整个 agent loop。
            if self._aborted:
                break

            # 每轮调用模型前尝试压缩上下文，避免消息历史过长。
            await self._prepare_model_context()

            if not self.is_sub_agent:
                start_spinner()


            # 等完整响应成功后才执行工具。流式请求可能重试，提前执行会让同一工具产生重复副作用。
            budget = self._check_budget()
            if budget["exceeded"]:
                self.last_finish_reason = "budget_exceeded"
                if not self.is_sub_agent:
                    stop_spinner()
                self._emit_text(f"Budget exceeded: {budget['reason']}")
                break
            # 这个response是Anthropic的响应对象，里面包含了模型的输出内容、token消耗、结束原因等信息
            response = await self._call_anthropic_stream()
            # 获取结束原因
            self.last_finish_reason = response.stop_reason
            if not self.is_sub_agent:
                stop_spinner()

            # 记录本次模型调用的耗时点和 token 消耗，用于成本展示与预算控制。
            self.last_api_call_time = time.time()
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.last_input_token_count = response.usage.input_tokens

            # Anthropic 的响应内容里可能混有 text block 和 tool_use block，这里只挑出工具调用。
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            # 把模型返回的所有 content block 写入消息历史，后续 tool_result 要与这些 tool_use 对应。
            self._anthropic_messages.append({
                "role": "assistant",
                "content": [self._block_to_dict(b) for b in response.content],
            })

            # 没有工具调用，说明模型已经给出最终回复，本轮对话结束。
            if not tool_uses:
                if not self.is_sub_agent:
                    print_cost(self.total_input_tokens + self.side_input_tokens, self.total_output_tokens + self.side_output_tokens,
                               self.input_price_per_million, self.output_price_per_million)
                break

            # 如果是非工具使用/非结束轮次的结束原因,则结束当前回应，并把工具调用的结果标记为跳过执行。
            if self.last_finish_reason not in {"tool_use", "end_turn"}:
                self._anthropic_messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tu.id,
                     "content": "Tool execution skipped: incomplete model response."} for tu in tool_uses]})
                break

            # Reserve one tool round only after verifying that a round is available.
            # 开始执行tool调用前，必须先检查校验
            budget = self._check_budget()
            if budget["exceeded"]:
                print_info(f"Budget exceeded: {budget['reason']}")
                self._anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": f"Tool execution skipped: {budget['reason']}",
                        }
                        for tu in tool_uses
                    ],
                })
                break
            self.current_turns += 1
            self._begin_tool_round((tu.id, tu.name) for tu in tool_uses)


            # 收集本轮所有工具结果，之后作为 tool_result 消息回传给模型。
            tool_results: list[dict] = []
            context_break = False

            for tu in tool_uses:
                # context_break 表示某个工具执行期间清理了上下文，需要停止继续处理本轮剩余工具。
                if context_break or self._aborted:
                    break

                # 将工具入参转为普通 dict，便于权限检查、打印和实际执行。
                inp = tu.input
                res = await self._run_tracked_tool(tu.id, tu.name, inp)
                print_tool_result(tu.name, res)
                self._record_tool_outcome(tu.name, not self._tracked_tool_failed(tu.id, tu.name, res))

                if self._context_cleared:
                    # 工具执行过程中如果清理了上下文，就把结果作为新的用户消息写入，
                    # 并停止继续处理本轮剩余工具，避免旧上下文和新上下文混在一起。
                    self._context_cleared = False
                    # 这里直接把工具结果作为用户消息写入，是因为上下文已经更新过了,前面的tool call已经没有匹配对应的了.
                    # 这里用user代为返回tool的值既可以传递工具信息还可以保证Anthropic的协议要求
                    self._anthropic_messages.append({"role": "user", "content": res})
                    context_break = True
                    # 这里的break,会导致旧的部分tool call在checkpoint一直处于pending状态
                    # 只能等下一次chat_turn()时,再把这些pending的tool call给清理掉,所以这里直接break掉,不再继续执行后续的tool call
                    break

                # Anthropic 要求 tool_result 使用 tool_use_id 对应到前面的 tool_use block。
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": res})

            if not context_break and tool_results:
                # Anthropic 要求 assistant/tool_use 后面紧跟一条 user/tool_result 消息，
                # 且这条消息必须包含本轮所有 tool_use 的对应结果。
                self._anthropic_messages.append({"role": "user", "content": tool_results})

            self._context_cleared = False

            # 工具结果可能很长，每轮工具执行后检查是否需要压缩上下文。
            self._refresh_runtime_system_prompt()

    @staticmethod
    def _block_to_dict(block) -> dict:
        """将Anthropic content block 转为 dict，便于存储和传递"""
        if block.type == "text":
            return {"type": "text", "text": _safe_utf8_text(block.text)}
        if block.type == "tool_use":
            raw_input = dict(block.input) if hasattr(block.input, 'items') else block.input
            return {"type": "tool_use", "id": _safe_utf8_text(block.id), "name": _safe_utf8_text(block.name), "input": _sanitize_for_utf8(raw_input)}
        # Fallback
        return {"type": _safe_utf8_text(block.type)}

    async def _call_anthropic_stream(self, on_tool_block_complete=None):
        """类似于openAI协议的逻辑,具体的可以看openAI下面的分析"""

        async def _do():
            self._require_model_budget()
            max_output = self.max_output_tokens or _get_max_output_tokens(self.model)

            create_params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_output if self._thinking_mode != "disabled" else (self.max_output_tokens or 16384),
                # 加入系统提示词
                "system": _safe_utf8_text(self._system_prompt),
                "tools": _sanitize_for_utf8(get_active_tool_definitions(self.tools)),
                "messages": _sanitize_for_utf8(self._anthropic_messages),
            }
            #如果开启了思考模式，就给 Anthropic 请求加上 thinking 参数。
            if self._thinking_mode  in ("adaptive", "enabled"):
                create_params["thinking"]={"type": "enabled", "budget_tokens": max_output - 1}

            first_text = True

            tool_blocks_by_index: dict[int, dict] = {}

            # 跟openAI不同的是,anthropic协议的流式请求不是塞在params里,
            # 而是直接调用stream()方法,返回一个异步迭代器,每次迭代返回一个事件(event),事件类型包括了content_block_start/content_block_delta/content_block_stop等
            async with self._anthropic_client.messages.stream(**create_params)as stream:
                async for event in stream:
                    if not hasattr(event, 'type'):
                        continue
                    # 当事件是工具调用开始：
                    if event.type == "content_block_start":
                        # 获取内容块
                        cb = getattr(event, 'content_block', None)
                        #如果 block 类型是 tool_use，就记录这个工具调用：
                        if cb and getattr(cb, 'type', None) == "tool_use":
                            #因为工具参数 JSON 是流式分片返回的，所以先准备一个空字符串 input_json。
                            tool_blocks_by_index[event.index]= {
                                "id": cb.id, "name": cb.name, "input_json": "",
                            }
                    #当事件是内容增量，分三种情况。
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        # 第一种，普通文本：模型输出正文时，
                        # 调用 _emit_text()。如果是普通交互，就打印；
                        # 如果是 run_once()，就写入 _output_buffer。
                        if hasattr(delta, "text"):
                            if first_text:
                                stop_spinner()
                                self._emit_text("\n")
                                first_text = False
                            self._emit_text(delta.text)
                        # Reasoning has its own channel and is not answer text.
                        # 思考有自己的通道,不在_emit_text中进行展示打印
                        elif hasattr(delta, 'thinking'):
                            self._emit_thinking(_safe_utf8_text(delta.thinking))
                        #第三种，工具参数 JSON 片段：工具调用的参数不是一次性返回，
                        # 而是一段一段返回，所以这里不断拼接到 input_json。
                        elif hasattr(delta, 'partial_json'):
                            # 这里用 event.index 来找到对应的工具调用记录，并把增量 JSON 拼接到 input_json。
                            tb = tool_blocks_by_index.get(event.index)
                            if tb:
                                tb["input_json"] += _safe_utf8_text(delta.partial_json)
                    #当一个 content block 结束：
                    #如果结束的是之前记录的工具调用，就把拼好的 JSON 解析出来：
                    elif event.type == "content_block_stop":
                        tb = tool_blocks_by_index.pop(event.index, None)
                        if tb and on_tool_block_complete:
                            import json as _json
                            try:
                                parsed = _json.loads(tb["input_json"] or "{}")
                            except Exception:
                                parsed = {}
                            #然后调用回调：
                            #这个回调的作用通常是：工具调用一完整，
                            # 就可以提前开始执行工具，不必等整条 assistant 消息全部结束。
                            on_tool_block_complete({
                                "type": "tool_use", "id": _safe_utf8_text(tb["id"]),
                                "name": _safe_utf8_text(tb["name"]), "input": _sanitize_for_utf8(parsed),
                            })
                # `final_message` 不是最后一个流式片段，而是 SDK 将本次流式事件汇总后得到的完整模型响应。
                # final_message 包含什么？
                # content	内容块列表，例如正文、工具调用
                # stop_reason	本次生成为什么停止
                # usage	输入、输出 token 用量
                # id、`role`、`model`	消息标识、角色、模型信息
                final_message = await stream.get_final_message()

            #过滤思考的message（因为 thinking 内容一般不应该进入历史消息，否则后续上下文会变大，也可能不符合 API 消息格式要求。）
            final_message.content = [b for b in final_message.content if b.type != "thinking"]
            return final_message
        #调用 _do()，如果遇到可重试错误，就由 _with_retry() 负责重试。
        return await self._with_stream_retry(_do, retry=_with_retry)

    #openAI后端

    async def _chat_openai(self, user_message:str) -> None:
        budget = self._check_budget()
        # 如果账单已经超过了限制，就直接返回，不再继续执行。
        if budget["exceeded"]:
            self.last_finish_reason = "budget_exceeded"
            self._emit_text(f"Budget exceeded: {budget['reason']}")
            return
        # 利用user_message调用_recall_before_response(),寻找当前cwd下面的最相近的5篇memory.md文件(最多)
        # 并将它们格式化后拼接到user_message后面(标记是<system-reminder>)，作为模型的上下文输入。
        user_message = _safe_utf8_text(await self._recall_before_response(user_message))
        # 在真正的循环前,先更新messages，把用户输入放入消息历史中.
        self._openai_messages.append({"role": "user", "content": user_message})

        # 这个长训话本质上是：
        # self._call_openai_stream() -> _run_tracked_tool() -> _run_checked_tool() ->_execute_tool_call() -> -> tool_func()
        # -> Update OpenAI.messages with tool result -> self._call_openai_stream() -> ...
        # 知道出现异常，或者被显式_aborted
        while True:
            # 首先判断当前是否被外部请求中止，如果是，就结束整个 agent loop。
            if self._aborted:
                break

            # 每次跟llm对话前，对上下文信息进行相应处理。包括messages中的tool result压缩、折叠上下文、清理上下文等。
            await self._prepare_model_context()

            if not self.is_sub_agent:
                # 只有主Agent才需要显示执行动画
                start_spinner()

            budget = self._check_budget()
            # 超过budget直接停止
            if budget["exceeded"]:
                self.last_finish_reason = "budget_exceeded"
                if not self.is_sub_agent:
                    stop_spinner()
                self._emit_text(f"Budget exceeded: {budget['reason']}")
                break

            # 开始进行openAI的流式请求，获取模型的响应。这个方法会处理流式输出，返回token usage 和 tool call的情况用于后续分析
            response = await self._call_openai_stream()

            # 如果不是子Agent再次下发停止动画
            if not self.is_sub_agent:
                stop_spinner()

            # 记录上次流式对话后的时间
            self.last_api_call_time = time.time()

            # 开始更新成本
            if response.get("usage"):
                self.total_input_tokens += response["usage"]["prompt_tokens"]
                self.total_output_tokens += response["usage"]["completion_tokens"]
                self.last_input_token_count = response["usage"]["prompt_tokens"]

            choice = response.get("choices", [{}])[0] if response.get("choices") else {}
            # 更新结束原因
            self.last_finish_reason = choice.get("finish_reason")
            message = choice.get("message", {})

            # 更新消息历史，记录本轮模型的输出。
            self._openai_messages.append(message)

            # 记录工具需要调用的情况
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                if not self.is_sub_agent:
                    print_cost(self.total_input_tokens + self.side_input_tokens, self.total_output_tokens + self.side_output_tokens,
                               self.input_price_per_million, self.output_price_per_million)
                break

            # 如果借宿原因不是工具调用或者stop，说明模型没有给出完整的响应，直接跳过本轮工具执行。
            if self.last_finish_reason not in {"tool_calls", "stop"}:
                self._append_openai_skipped_tool_results(tool_calls, "Tool execution skipped: incomplete model response.")
                break

            # 检查预算，预算超过也需要强行停止那些还没调用的 tool
            budget = self._check_budget()
            if budget["exceeded"]:
                print_info(f"Budget exceeded: {budget['reason']}")
                self._append_openai_skipped_tool_results(tool_calls, f"Tool execution skipped: {budget['reason']}")
                break
            self.current_turns += 1

            # 预处理工具请求：提取函数名并解析参数
            oai_checked: list[dict] = []
            for tc in tool_calls:
                if tc.get("type") != "function":
                    continue
                fn_name = tc["function"]["name"]
                try:
                    inp = json.loads(tc["function"]["arguments"])
                except Exception:
                    inp = None
                oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp})

            # 为新产生的tool call记录一个新的工具调用轮次，方便后续统计和分析。
            self._begin_tool_round((ct["tc"]["id"], ct["fn"]) for ct in oai_checked)

            # 划分工具调用批次
            # 这段代码的规则是：
            # - 当前工具允许并发，而且上一批也是并发批次：追加到上一批。
            # - 否则：创建一个新批次。
            oai_batches: list[dict] = []
            for ct in oai_checked:
                # CONCURRENCY_SAFE_TOOLS 是允许并发执行的工具集合。
                safe = ct["fn"] in CONCURRENCY_SAFE_TOOLS
                if safe and oai_batches and oai_batches[-1]["concurrent"]:
                    oai_batches[-1]["items"].append(ct)
                else:
                    oai_batches.append({"concurrent": safe, "items": [ct]})

            oai_context_break = False
            try:
                for batch in oai_batches:
                    if oai_context_break or self._aborted:
                        break

                    # 如果当前批次是可以并发执行的，就使用 asyncio.gather() 来同时执行所有工具调用。
                    if batch["concurrent"]:
                        async def _run_oai_safe(ct_item: dict) -> tuple[dict, str]:
                            # 这里的ct_item["tc"]["id"]表示的是这个tool 调用在llm下的id是什么
                            # ct_item["fn"] 表示的是工具的名称
                            # ct_item["inp"] 表示的是工具的输入参数
                            res = await self._run_tracked_tool(ct_item["tc"]["id"], ct_item["fn"], ct_item["inp"])
                            # 展示工具执行结果
                            print_tool_result(ct_item["fn"], res)
                            return ct_item, res

                        # 批量等待所有的并发下的item完成_run_oai_safe方法
                        results = await asyncio.gather(*[_run_oai_safe(ct) for ct in batch["items"]])
                        for ct_item, res in results:
                            # 记录工具执行结果
                            self._record_tool_outcome(
                                ct_item["fn"],
                                not self._tracked_tool_failed(ct_item["tc"]["id"], ct_item["fn"], res),
                            )
                            # 更新openAI.messages，记录工具调用的结果
                            self._openai_messages.append(
                                {"role": "tool", "tool_call_id": ct_item["tc"]["id"], "content": res})
                    else:
                        # 如果不是并发的，本质逻辑跟上面是一致的
                        for ct in batch["items"]:
                            if self._aborted:
                                break
                            res = await self._run_tracked_tool(ct["tc"]["id"], ct["fn"], ct["inp"])
                            print_tool_result(ct["fn"], res)
                            self._record_tool_outcome(
                                ct["fn"],
                                not self._tracked_tool_failed(ct["tc"]["id"], ct["fn"], res),
                            )

                            # 这个条件只有在这里有，是因为上面是并发安全的，所以不可能会更新外部环境
                            # 这个表示如果上下文清理过
                            if self._context_cleared:
                                # 清理过就代表当前的上下文已经被清理了，后续的工具调用都没有任何意义了，但是还需要在最后把没有完成的调用补全原因
                                self._context_cleared = False
                                # 定义成user单纯是因为context clear(原有的messages都已经被清理了)，
                                # 所以这里直接把工具结果作为用户消息写入，是因为上下文已经更新过了,前面的tool call已经没有匹配对应的了.
                                # 用user则没有这个问题
                                self._openai_messages.append({"role": "user", "content": res})
                                oai_context_break = True
                                break

                            self._openai_messages.append(
                                {"role": "tool", "tool_call_id": ct["tc"]["id"], "content": res})
            except asyncio.CancelledError:
                self._recover_tool_checkpoints()
                raise

            if self._aborted:
                # 如果终止，恢复那些还没有执行完成的tool的checkpoint
                self._recover_tool_checkpoints()
            elif oai_context_break:
                # 仅为当前历史中仍存在、且缺少结果的工具调用补充跳过说明。
                # 若上下文重置已移除旧调用，则不追加结果
                # PS: 根据上下问的解释,其实在被清理完上下文之后,其实本质上没必要去补充这些tool的调用,因为对应的信息已经在上下文中被清理掉了
                self._append_openai_skipped_tool_results(
                    tool_calls,
                    "Tool execution skipped because the conversation context was reset.",
                )

            self._context_cleared = False
            self._refresh_runtime_system_prompt()

    def _append_openai_skipped_tool_results(self, tool_calls: list[dict], reason: str) -> None:
        """
        _append_openai_skipped_tool_results用于在chat出现异常的时候们帮忙补齐那些已经记录在messages中要调用的工具调用，但是还没有complete的。
        它会检查当前消息历史中已经存在的工具调用ID，并将那些还没有执行的工具调用的结果补充到消息历史中，标记为被跳过，并附上原因。
        Complete calls that still belong to the current conversation."""

        # 找到messages中仍然还存活的工具调用ID（即assistant消息中的tool_calls）
        live_ids = {
            call.get("id")
            for msg in self._openai_messages if msg.get("role") == "assistant"
            for call in msg.get("tool_calls") or []
        }
        # 找到messages中已经记录的工具调用ID（即tool消息中的tool_call_id）
        existing_ids = {
            msg.get("tool_call_id")
            for msg in self._openai_messages
            if msg.get("role") == "tool"
        }
        for tc in tool_calls:
            if tc.get("type") != "function":
                continue
            tool_call_id = tc.get("id")
            if tool_call_id and tool_call_id in live_ids and tool_call_id not in existing_ids:
                self._openai_messages.append(
                    {"role": "tool", "tool_call_id": tool_call_id, "content": reason}
                )
                existing_ids.add(tool_call_id)

    async def _call_openai_stream(self) -> dict:
        """向模型发起一次流式请求，一边接收、一边显示回答，同时把分散的文本、工具调用参数和 token 用量收集起来，最后整理成一个完整字典，交给 _chat_openai() 继续处理。

        PS:这个函数会收集模型提出的工具调用，但不会执行工具。 工具执行仍然发生在前面的 _chat_openai() 中。
        """

        async def _do():
            """这是进行的一次尝试"""

            # 在调用OpenAI接口前，检查当前预算是否允许继续执行模型调用。
            # 如果超出预算则，直接报错
            self._require_model_budget()
            create_params = dict(
                model=self.model,
                # 这里的self.tools其实就是项目内置的方法还有mcp中返回的方法
                tools=_sanitize_for_utf8(_to_openai_tools(get_active_tool_definitions(self.tools))),
                # 这里包含了历史的对话记录+runtime-prompt+可能需要使用的tool+被召回的memory.md文件内容+原始的用户输入
                messages=_sanitize_for_utf8(self._openai_messages),
                # 定义了本次模型调用为流式
                stream=True,
                # 需要包含token使用情况
                stream_options={"include_usage": True},
            )
            if self.max_output_tokens is not None:
                create_params["max_tokens"] = self.max_output_tokens

            # 调用OpenAI的chat.completions.create接口，返回一个异步迭代器stream
            stream = await self._openai_client.chat.completions.create(**create_params)

            # content	拼接完整的普通回答文本
            # first_text	标记是否还没收到第一个非空文本片段
            # tool_calls	按工具调用序号收集工具名称、ID 和参数
            # finish_reason	保存模型停止生成的原因
            # usage	保存本次请求的输入、输出 token 用量
            content = ""
            first_text = True
            tool_calls: dict[int, dict] = {}
            finish_reason = ""
            usage = None

            try:
                async for chunk in stream:
                    # 接收 chunk，先提取用量，再判断是否有回答内容
                    # 因为开启 include_usage=True 后，接口会额外发送一个包含整次请求用量的片段，而该片段的 choices 为空
                    if chunk.usage:
                        usage = {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                        }

                    # 如果没有 choices项就等下一个分片，choices可以理解为大模型预测的回复信息
                    if not chunk.choices:
                        continue
                    # 表示它只处理第一个候选回答
                    delta = chunk.choices[0].delta

                    # 尝试获取候选回答中的reasoning选项
                    # 这里的 reasoning 是模型服务额外返回的推理说明文本
                    reasoning = (getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)) if delta else None
                    if isinstance(reasoning, str) and reasoning:
                        self._emit_thinking(_safe_utf8_text(reasoning))

                    # 接受文本增量，delta.content是模型输出的文本片段
                    if delta and delta.content:
                        # 如果是首文本
                        if first_text:
                            # 停止动画展示
                            stop_spinner()
                            # 开始向通道输入信息
                            self._emit_text("\n")
                            first_text = False
                        self._emit_text(delta.content)
                        # 逐步统计模型输出的文本内容，最后返回给调用者
                        content += _safe_utf8_text(delta.content)

                    # 判断是否需要工具调用
                    # 工具调用：按 index 分开保存，持续拼接参数
                    if delta and delta.tool_calls:
                        for tc in delta.tool_calls:
                            # 以 tc.index 为键，分别收集各个工具调用。保证即使分片，也不会当parameters被拆开时丢失。
                            existing = tool_calls.get(tc.index)
                            if existing:
                                if tc.function and tc.function.arguments:
                                    existing["arguments"] += _safe_utf8_text(tc.function.arguments)
                            else:
                                # 表示这是一个新的工具调用，初始化一个字典，保存工具调用的id、name和参数
                                tool_calls[tc.index] = {
                                    "id": _safe_utf8_text(tc.id or ""),
                                    "name": _safe_utf8_text((tc.function.name if tc.function else "") or ""),
                                    "arguments": _safe_utf8_text((tc.function.arguments if tc.function else "") or ""),
                                }

                    # 收集结束原因
                    # 但这里并没有直接关闭，而是等他chunk没有“自然结束”
                    if chunk.choices[0].finish_reason:
                        # 常见值包括：
                        # "stop"	正常停止生成
                        # "tool_calls"	模型发出了工具调用
                        # "length"	达到生成长度限制
                        # "content_filter"	内容因过滤而被省略
                        finish_reason = chunk.choices[0].finish_reason

            finally:
                await stream.close()

            # 将收集到的工具调用转换为完整消息格式
            assembled = None
            if tool_calls:
                assembled = [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for _, tc in sorted(tool_calls.items())
                ]

            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": assembled,
                    },
                    "finish_reason": finish_reason or "incomplete",
                }],
                "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
            }

        # 重试默认至多3次
        return await self._with_stream_retry(_do, retry=_with_retry)

    async def _confirm_tool_call(self, tool_name: str, permission: dict) -> bool:
        """
        _confirm_tool_call函数主要是当遇到需要confirm的工具调用时调用。
        主要逻辑：根据tool_name是否在always_confirm中，还有permission["message"]中的路径是否在self._confirmed_paths中，来决定是否需要人工确认。
        人工确认”Yes“后其对应的调用路径会被加入self._confirmed_paths中，之后相同路径的调用就不需要再次确认了。

        permission格式：{"action": "allow"|"deny"|"confirm", "message": "..."}"""
        message = permission.get("message", "")
        # Repeated shell and file changes need fresh approval, even with identical arguments.
        always_confirm = tool_name in {"run_shell", "write_file", "edit_file", "skill_create", "skill_evolve",
                                       "skill_prune", "skill_restore", "mcp_start"} or tool_name.startswith("mcp__")
        if not always_confirm and message in self._confirmed_paths:
            return True
        if not await self._confirm_dangerous(message):
            return False
        if not always_confirm:
            self._confirmed_paths.add(message)
        return True

    async def _confirm_dangerous(self, command: str) -> bool:
        """调用确认函数，或者阻塞等待用户输入确认。"""
        print_confirmation(command)
        if self.confirm_fn:
            return await self.confirm_fn(command)
        # Fallback: blocking input
        try:
            answer = input("  Allow? (y/n): ")
            return answer.lower().startswith("y")
        except EOFError:
            return False
