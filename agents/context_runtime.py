"""Shared context preflight and bounded memory recall for both protocols."""
from __future__ import annotations

import asyncio
import json
import logging
import time

from .memory import (MAX_SESSION_MEMORY_BYTES, start_memory_prefetch,
                     format_memories_for_injection)
from .tool_results import compact_tool_results


class ContextRuntime:
    def _run_compression_pipeline(self) -> None:
        """执行压缩流水线,本质就是压缩messages中一些老旧的/过时的/已经没有用被覆盖的工具调用的结果"""
        messages = self._openai_messages if self.use_openai else self._anthropic_messages
        # 上下文占用比例 = 最近一次请求的输入 token 数 ÷ 有效上下文窗口大小
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        # 计算距离上次 API 调用的时间
        idle_seconds = time.time() - self.last_api_call_time if self.last_api_call_time else 0
        compact_tool_results(messages, use_openai=self.use_openai,
                             utilization=utilization, idle_seconds=idle_seconds)

    async def _recall_before_response(self, user_message: str) -> str:
        """在模型回答之前，根据当前用户消息召回相关记忆，筛选后追加到用户消息末尾，让模型带着这些记忆回答。

        没有可用记忆 → 返回原始 user_message
        有可用记忆   → 返回 user_message + 格式化后的记忆"""

        # 如果是子agent，或者没有开启记忆预取功能，就直接返回原始用户消息
        if self.is_sub_agent or not self.enable_memory_prefetch:
            return user_message
        query = self._build_side_query()
        if query is None:
            return user_message
        # 根据user_message/query/_already_surfaced_memories/_session_memory_bytes,创建了一个进行记忆召回的循环事件任务
        # 然后返回了一个可以控制这个循环事件中的该任务的句柄
        prefetch = start_memory_prefetch(user_message, query,
                                        self._already_surfaced_memories, self._session_memory_bytes)
        # 句柄没获取到,表示存在条件不符合情况,那么直接返回原始用户消息
        if prefetch is None:
            return user_message
        # 将prefetch添加到Agent的_memory_tasks集合中,并在prefetch的task完成后从_memory_tasks中移除，并在prefetch失败时记录警告日志
        self._track_memory_prefetch(prefetch)
        try:
            # 这里显示的等待prefetch.task完成,并设置了超时时间为self.memory_recall_timeout
            # 任务主要返回最多5个被选中的RelevantMemory对象,每个对象包括了path/content/mtime_ms/header这四个属性
            memories = await asyncio.wait_for(prefetch.task, timeout=self.memory_recall_timeout)
        except asyncio.TimeoutError:
            logging.info("Memory recall timed out; continuing without recalled memory")
            return user_message
        except Exception:
            logging.exception("Memory recall failed; continuing without recalled memory")
            return user_message
        selected = []
        for memory in memories:
            # 二次校验，确保memory.path不在已经被召回的集合中，并且当前session的记忆总字节数加上这个memory的大小不超过最大限制
            if (memory.path not in self._already_surfaced_memories
                    and self._session_memory_bytes + memory.size <= MAX_SESSION_MEMORY_BYTES):
                selected.append(memory)
                # 更新已经引用的memory集合和当前session的记忆总字节数
                self._already_surfaced_memories.add(memory.path)
                self._session_memory_bytes += memory.size
        if not selected:
            return user_message
        return user_message + "\n\n" + format_memories_for_injection(selected)

    def _context_request_bytes(self) -> int:
        """计算当前上下文请求的字节数，包含工具/系统文本和非ASCII文本；
        这是一种保守的启发式方法，并非精确的分词器，也无法保证适用于每个模型的上下文限制。

        PS：这个计算非常的模糊！！！！！比如openAI写协议下prompt就算了2遍（messages中有system-prompt）"""
        messages = self._openai_messages if self.use_openai else self._anthropic_messages
        # Include tools/system and non-ASCII text; this is a conservative heuristic,
        # not an exact tokenizer or a guarantee for every model's context limit.
        return len(json.dumps([self._system_prompt, self.tools, messages],
                              ensure_ascii=False, default=str).encode("utf-8", errors="replace"))

    def _estimated_input_tokens(self) -> int:
        """估算当前上下文（输入）Token 数量"""

        # 计算当前上下文请求的字节数，主要包括了prompt、tools和messages的内容，这个是自己估计的
        size = self._context_request_bytes()
        # 计算上一次chat_turn的时候的输入token数
        anchor = self._last_request_bytes
        # 计算增长量，如果anchor不为None，则增长量 = max(0, size - anchor)，否则增长量为0
        # None的情况比如说上一次是messages的fold压缩
        growth = max(0, size - anchor) if anchor is not None else 0
        # (size + 2) // 3：全量粗估。+2 是整数除法向上取整的惯用写法，即 ceil(size / 3)。这里用 3 字节 ≈ 1 token 的经验比例（对英文/代码大致成立，中文偏低）。
        # self.last_input_token_count + (growth + 2) // 3：增量估算。在上次已知的精确 Token 数基础上，只把新增的字节按 1/3 换算后加上去。(growth + 2) // 3 同样是向上取整。

        # last_input_token_count
        return max((size + 2) // 3, self.last_input_token_count + (growth + 2) // 3)

    async def _prepare_model_context(self) -> None:
        """准备模型所需要的上下文
        主要步骤包括：
        1、执行压缩流水线,本质就是压缩messages中一些老旧的/过时的/已经没有用被覆盖的工具调用的结果
        2、检查是否需要进行自动压缩
        3、更新发送前request前的字节数
        """
        self._run_compression_pipeline()
        await self._check_and_compact()
        self._last_request_bytes = self._context_request_bytes()
