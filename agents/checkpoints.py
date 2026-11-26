"""Durable tool boundaries; recovery never replays an uncertain side effect."""
from __future__ import annotations

from contextvars import ContextVar

from .result_storage import persist_tool_result

# ContextVar 是上下文变量。这里用来存放当前工具调用的 ID。
# 它和普通全局变量的关键区别是：在异步并发任务中，不同任务可以拥有各自的值，避免互相覆盖。
# 如果是同线程的，只要不是在一个await中的就可以区分
CURRENT_CALL = ContextVar("checkpoint_call", default=None)

# Mixin（混入类）单纯为了集成一些方法，起作用就是被某个类所继承，这样子self就可以调用这些方法了
class ToolCheckpoints:
    def _save_tool_checkpoint(self) -> None:
        """保存当前的工具调用状态到会话中"""
        if not self.is_sub_agent:
            # A failed write must prevent starting an unjournaled operation.
            self._auto_save(strict=True)

    def _begin_tool_round(self, calls) -> None:
        """为所有的calls添加一个checkpoint，初始状态为not_started的记录，并保存到self._tool_checkpoints中"""
        self._tool_checkpoints = {
            call_id: {"name": name, "status": "not_started"}
            for call_id, name in calls
        }
        self._save_tool_checkpoint()

    async def _run_tracked_tool(self, call_id: str, name: str, inp) -> str:
        """执行工具调用，并在self._tool_checkpoints中记录状态和结果"""
        record = self._tool_checkpoints[call_id]
        record["status"] = "running"
        self._save_tool_checkpoint()
        # CURRENT_CALL：让执行链知道“当前是哪一次调用”，后续web._run_checked_tool会进行调用检查
        token = CURRENT_CALL.set(call_id)
        try:
            result = await self._run_checked_tool(name, inp)
        except BaseException:
            record["status"] = "unknown"
            # The persisted running record already means unknown after a crash.
            raise
        finally:
            CURRENT_CALL.reset(token)
        # 判断当前的调用结果是否算失败的
        failed = self._looks_like_tool_failure(name, result, "")
        # 对结果进行一个持久化操作
        stored = persist_tool_result(name, result)
        record.update(status="completed", failed=failed, **stored)
        self._save_tool_checkpoint()
        return stored["result"]

    def _tracked_tool_failed(self, call_id: str, name: str, result: str) -> bool:
        record = self._tool_checkpoints.get(call_id, {})
        # 根据checkpoint记录的failed字段来判断工具调用是否失败，如果没有failed字段，就用_looks_like_tool_failure来判断
        if isinstance(record.get("failed"), bool):
            return record["failed"]
        return self._looks_like_tool_failure(name, result, result)

    def _recover_tool_checkpoints(self) -> None:
        """检查最近一条 assistant 消息中的工具调用，为缺失的结果补写消息：
        已完成的调用恢复保存的结果；未开始或状态未知的调用补写状态说明。
        本函数不会重新执行工具。其中会更新 self._tool_checkpoints 还有 _openai_messages/_anthropic_messages 中的消息内容。"""
        messages = self._openai_messages if self.use_openai else self._anthropic_messages
        # 从最新到最老，倒着遍历
        # 找到最新的assistant消息中所有的工具调用，检查其是否已经有对应的结果，如果没有就补写结果，补写完成后直接break
        # 如果最新的assistant消息中没有工具调用，就不需要补写结果了直接break
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            # 只有"role": "assistant"(表示这条消息的主角是大模型本身)的消息中，才可能包含了工具调用的记录（OpenAI 的 tool_calls 或 Anthropic 的 tool_use），
            if message.get("role") != "assistant":
                continue
            # self._anthropic_tool_use_ids(message)这里的self本质上调用这个方法的对象，其实是Agent，
            # 所以即使这个方法写在了ToolCheckpoints类里面，self也可以调用Agent的属性和方法
            ids = ([call["id"] for call in message.get("tool_calls") or []]
                   if self.use_openai else list(self._anthropic_tool_use_ids(message)))
            if not ids:
                break
            # 收集"已经存在的工具结果 id 集合"
            if self.use_openai:
                # OpenAI：结果可以散落在后面的多条 tool 消息里
                present = {item.get("tool_call_id") for item in messages[index + 1:] if item.get("role") == "tool"}
            else:
                if index + 1 >= len(messages):
                    messages.append({"role": "user", "content": []})
                following = messages[index + 1]
                # Anthropic：工具结果放在紧接着的 user 消息中，而且该消息里的工具结果块要排在普通文本前面。
                present = self._anthropic_tool_result_ids(following)
                # 如果assistant后面跟的消息里面没有content字段，或者content字段不是list类型，
                # 此时前者的present必定是一个空的set，那么需要再后面插入一条空的消息
                if not isinstance(following.get("content"), list):
                    messages.insert(index + 1, {"role": "user", "content": []})
                following = messages[index + 1]
            # 循环所有后续的tool call id，只有不在context，才认为时可能遗漏的结果，需要补写(反之认为只要写到了context中一定对应的更新了checkpoint)
            for call_id in ids:
                # 在messages中所有调用的方法中，如果已经存在了这个工具调用的结果，就不需要再补充占位结果了
                if call_id in present:
                    continue
                # 如果record没有找到的话而导致结果为{}，后续对record的操作不会影响到_tool_checkpoints，反之则会影响
                record = self._tool_checkpoints.get(call_id, {})
                # 如果已完成，就进行落库，然后更新_tool_checkpoints中对应项的result和result_file字段
                if record.get("status") == "completed" and isinstance(record.get("result"), str):
                    # Older sessions stored raw output. Migrate only when it is
                    # recovered, without injecting the entire artifact again.
                    # 先用persist_tool_result把工具调用的结果持久化，然后更新_tool_checkpoints中对应项的result和result_file字段

                    # 完成的tool call结果如果太大就尝试落库，然后裁剪result后最后更新到messages.runtimeStatus.checkpoints中对应的result字段处
                    record.update(persist_tool_result(record.get("name", "tool"), record["result"], reuse_existing=True))
                    result = record["result"]
                elif record.get("status") == "not_started":
                    result = "Tool execution skipped: interrupted before execution; this tool was not started."
                else:
                    result = "Tool execution status unknown after interruption. Inspect actual state before retrying; do not automatically repeat side effects."
                    record["status"] = "unknown"
                if self.use_openai:
                    # OpenAI协议可以直接在messages中插入一条新的tool消息，表示工具调用的结果
                    messages.insert(index + 1, {"role": "tool", "tool_call_id": call_id, "content": result})
                else:
                    # Anthropic协议不能像OpenAI一样直接用insert，是因为必须要吧所有tool_result块放在紧接着的user消息中。
                    following["content"].append({"type": "tool_result", "tool_use_id": call_id, "content": result})
            break
