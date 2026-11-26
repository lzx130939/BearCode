"""Session inspection is local, protocol-aware, and leaves runtime state intact."""
from __future__ import annotations

from copy import deepcopy
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from rich.console import Console

from agents import memory, session
from agents.main import run_repl
from agents.session_detail import build_session_detail, print_session_detail


def sample_agent(**overrides):
    values = dict(
        use_openai=True, session_id="active-session", _conversation_id="stable-conversation",
        _session_title="恢复文件任务", model="offline-model", permission_mode="dontAsk",
        _openai_messages=[], _anthropic_messages=[], _tool_checkpoints={},
        _folded_session_memories=[], _already_surfaced_memories=set(), _session_memory_bytes=0,
        set_confirm_fn=Mock(), set_plan_approval_fn=Mock(), chat=AsyncMock(), aclose=AsyncMock(),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def openai_call(call_id, name="read_file"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": '{"file_path":"demo.txt"}'}}


class SessionDetailTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for patcher in (patch("pathlib.Path.home", return_value=self.root),
                        patch.object(session, "SESSION_DIR", self.root / "sessions")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_empty_inspection_creates_no_files_and_does_not_call_model(self):
        agent = sample_agent(_openai_messages=[{"role": "system", "content": "test"}])
        before = deepcopy(agent._openai_messages)
        report = build_session_detail(agent)
        self.assertEqual(report["message_counts"], {"system": 1})
        self.assertEqual(report["tools"], [])
        self.assertEqual(report["memories"], [])
        self.assertEqual(report["session_id"], "active-session")
        self.assertEqual(report["conversation_id"], "stable-conversation")
        self.assertFalse(report["session_file_exists"])
        self.assertEqual(list(self.root.rglob("*")), [])
        self.assertEqual(agent._openai_messages, before)
        agent.chat.assert_not_called()

    def test_openai_calls_are_deduplicated_and_responses_are_paired(self):
        messages = [
            {"role": "user", "content": "read the files"},
            {"role": "assistant", "content": None, "tool_calls": [openai_call("read"), openai_call("skip", "write_file")]},
            {"role": "tool", "tool_call_id": "read", "content": "file body"},
            {"role": "tool", "tool_call_id": "skip", "content": "Tool execution skipped: not started."},
            {"role": "assistant", "tool_calls": [openai_call("read"), openai_call("")]},
            {"role": "assistant", "tool_calls": [openai_call("")]},
            {"role": "tool", "tool_call_id": "orphan", "content": "older result"},
        ]
        checkpoints = {"skip": {"name": "write_file", "status": "not_started"}}
        agent = sample_agent(_openai_messages=messages, _tool_checkpoints=checkpoints)
        before = deepcopy((messages, checkpoints))
        report = build_session_detail(agent)
        self.assertEqual(report["call_occurrences"], 5)
        self.assertEqual(len(report["tools"]), 4)
        self.assertEqual(report["tool_summary"]["read_file"], {"calls": 3, "with_response": 1, "without_response": 2})
        self.assertEqual(report["tool_summary"]["write_file"]["with_response"], 1)
        self.assertEqual(report["unmatched_results"], 1)
        self.assertEqual(report["tools"][0]["message_indices"], [2, 5])
        self.assertEqual(report["tools"][0]["results"][0]["message_index"], 3)
        self.assertEqual(report["tools"][1]["checkpoint"]["status"], "not_started")
        self.assertEqual((messages, checkpoints), before)

    def test_anthropic_content_blocks_and_active_protocol_are_counted(self):
        messages = [
            {"role": "user", "content": "read the file"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "reading"},
                {"type": "tool_use", "id": "a", "name": "read_file", "input": {"file_path": "a.txt"}},
                {"type": "tool_use", "id": "b", "name": "run_shell", "input": {"command": "echo test"}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": [{"type": "text", "text": "body"}]}]},
        ]
        agent = sample_agent(use_openai=False, _anthropic_messages=messages,
                             _openai_messages=[{"role": "assistant", "tool_calls": [openai_call("ignore")]}])
        report = build_session_detail(agent)
        self.assertEqual(report["protocol"], "anthropic")
        self.assertEqual(report["message_counts"], {"user": 2, "assistant": 1})
        self.assertEqual(len(report["tools"]), 2)
        self.assertEqual(report["tool_summary"]["run_shell"]["without_response"], 1)
        self.assertEqual(report["tools"][0]["arguments"], {"file_path": "a.txt"})

    def test_project_memory_contents_and_session_recall_flags_are_preserved(self):
        first = memory.save_memory("偏好", "格式约束", "user", "正文 [red] 必须原样显示")
        second = memory.save_memory("项目", "背景资料", "project", "项目内容")
        directory = memory.get_memory_dir()
        before = {path.name: path.read_bytes() for path in directory.glob("*.md")}
        agent = sample_agent(_already_surfaced_memories={str(directory / first)}, _session_memory_bytes=12)
        report = build_session_detail(agent)
        entries = {entry["filename"]: entry for entry in report["memories"]}
        self.assertEqual(set(entries), {first, second})
        self.assertTrue(entries[first]["recalled_in_session"])
        self.assertFalse(entries[second]["recalled_in_session"])
        self.assertIn("[red]", entries[first]["content"])
        self.assertEqual(report["recalled_path_count"], 1)
        self.assertEqual({path.name: path.read_bytes() for path in directory.glob("*.md")}, before)

    def test_current_context_is_counted_without_extrapolating_folded_history(self):
        agent = sample_agent(
            _openai_messages=[{"role": "user", "content": "<session-folded-memory>previous task</session-folded-memory>"}],
            _folded_session_memories=[{"episode_memory": {"task_description": "a previous long task"}}],
            _tool_checkpoints={"old": {"name": "read_file", "status": "completed", "result": "saved"}},
        )
        report = build_session_detail(agent)
        self.assertEqual(len(report["messages"]), 1)
        self.assertEqual(report["tools"], [])
        self.assertIn("old", report["checkpoints"])
        self.assertEqual(len(report["folds"]), 1)

    def test_renderer_includes_checkpoints_and_memory_by_default_and_expands_messages_with_detail(self):
        filename = memory.save_memory("Detail memory", "Memory description", "project", "[green] " + "m" * 900 + " MEMORY_END")
        call = openai_call("detail-call")
        call["function"]["arguments"] = '{"file_path":"' + "a" * 900 + ' TOOL_END"}'
        agent = sample_agent(_openai_messages=[
            {"role": "user", "content": "[red] " + "x" * 900 + " MESSAGE_END"},
            {"role": "assistant", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "detail-call", "content": "r" * 900 + " RESULT_END"},
        ], _tool_checkpoints={"detail-call": {
            "name": "read_file", "status": "completed", "failed": False,
            "result_file": "checkpoint-result.txt",
        }})
        report = build_session_detail(agent)
        output = StringIO()
        with patch("agents.session_detail.console", Console(file=output, width=120, color_system=None)):
            print_session_detail(report)
        for detail in ("Message #", "Tool #", "[red]", "MESSAGE_END", "TOOL_END", "RESULT_END"):
            self.assertNotIn(detail, output.getvalue())
        for detail in ("Current tool checkpoints", "detail-call", "completed", "checkpoint-result.txt",
                       "Project memory entries", "Memory description", "[green]", "MEMORY_END", filename):
            self.assertIn(detail, output.getvalue())
        self.assertEqual(output.getvalue().count("MEMORY_END"), 1)
        self.assertIn("Tool usage summary", output.getvalue())
        self.assertIn("read_file", output.getvalue())
        self.assertIn("/session-detail -d", output.getvalue())
        self.assertIn("recalled paths in this session", output.getvalue())
        self.assertIn("Memory bytes injected in this session", output.getvalue())
        self.assertNotIn("in this process", output.getvalue())
        self.assertNotIn("role", output.getvalue().lower())
        self.assertNotIn("角色", output.getvalue())
        output = StringIO()
        with patch("agents.session_detail.console", Console(file=output, width=120, color_system=None)):
            print_session_detail(report, full=True)
        for detail in ("Message #", "Tool #", "[red]", "MESSAGE_END", "TOOL_END", "RESULT_END", "MEMORY_END", filename):
            self.assertIn(detail, output.getvalue())
        self.assertNotIn("characters omitted", output.getvalue())
        self.assertIn("Project memory entries", output.getvalue())
        self.assertIn("Current tool checkpoints", output.getvalue())
        self.assertEqual(output.getvalue().count("MEMORY_END"), 1)
        self.assertIn("Session Detail", output.getvalue())


class SessionDetailReplTests(unittest.IsolatedAsyncioTestCase):
    async def test_repl_summary_detail_and_legacy_full_option_are_handled_locally(self):
        agent = sample_agent()
        with patch("agents.main._read_repl_input", new=AsyncMock(side_effect=[
            "/session-detail", "/session-detail -d", "/session-detail --full", "/session-detail --invalid", "quit",
        ])), patch("agents.main.signal.signal"), patch("agents.main.print_welcome"), \
             patch("agents.main.print_user_prompt"), patch("agents.main.print_goodbye"), \
             patch("agents.main.print_session_detail") as render, patch("agents.main.print_error") as error, \
             patch("agents.session_detail.memory.list_memories", return_value=[]), \
             patch("agents.session_detail.memory.get_memory_dir", return_value=Path.cwd()):
            await run_repl(agent)
        self.assertEqual(render.call_count, 3)
        self.assertFalse(render.call_args_list[0].kwargs["full"])
        self.assertTrue(render.call_args_list[1].kwargs["full"])
        self.assertTrue(render.call_args_list[2].kwargs["full"])
        error.assert_called_once()
        self.assertEqual(error.call_args.args[0], "Usage: /session-detail [-d]")
        agent.chat.assert_not_awaited()
        agent.aclose.assert_awaited_once()

    async def test_report_read_error_keeps_the_repl_open(self):
        agent = sample_agent()
        with patch("agents.main._read_repl_input", new=AsyncMock(side_effect=["/session-detail", "quit"])), \
             patch("agents.main.signal.signal"), patch("agents.main.print_welcome"), \
             patch("agents.main.print_user_prompt"), patch("agents.main.print_goodbye"), \
             patch("agents.main.build_session_detail", side_effect=OSError("unreadable memory")), \
             patch("agents.main.print_error") as error:
            await run_repl(agent)
        error.assert_called_once()
        self.assertIn("unreadable memory", error.call_args.args[0])
        agent.chat.assert_not_awaited()
        agent.aclose.assert_awaited_once()
