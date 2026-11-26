"""Persistence stays UTF-8 even when the OS and terminal use a legacy encoding."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents import memory, session, tools, ui
from agents.provenance_reader import load_recent_sources


class LegacyFileEncodingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # pathlib uses io.text_encoding() for calls without an explicit encoding.
        # Simulate the CI locale even when the interpreter runs in UTF-8 mode.
        for patcher in (
            patch("pathlib.Path.home", return_value=self.root),
            patch.object(session, "SESSION_DIR", self.root / "sessions"),
            patch("io.text_encoding", side_effect=lambda encoding, stacklevel=2: encoding or "cp1252"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_session_load_and_listing_preserve_chinese_messages_and_runtime_state(self):
        state = {"alreadySurfacedMemories": [str(self.root / "记忆.md")], "sessionMemoryBytes": 721}
        messages = [{"role": "user", "content": "请恢复中文会话，保留约束 📖"}]
        session.save_session("unicode-session", {
            "metadata": {"id": "unicode-session", "cwd": str(Path.cwd()), "title": "恢复中文会话"},
            "openaiMessages": messages, "runtimeState": state,
        })
        restored = session.load_session("unicode-session")
        self.assertIsNotNone(restored)
        self.assertEqual(restored["openaiMessages"], messages)
        self.assertEqual(restored["runtimeState"], state)
        self.assertEqual(session.list_sessions()[0]["title"], "恢复中文会话")

    def test_memory_index_and_contents_preserve_chinese(self):
        filename = memory.save_memory("代码偏好", "中文描述", "user", "保留正文 📖")
        self.assertIn("代码偏好", memory.load_memory_index())
        self.assertIn("保留正文 📖", memory.list_memories()[0].content)
        path = memory.get_memory_dir() / filename
        tools._auto_update_memory_index(str(path))
        self.assertIn("中文描述", memory.load_memory_index())

    def test_editing_utf8_file_keeps_unrelated_chinese_text(self):
        path = self.root / "example.txt"
        path.write_text('说明：保留中文 📖\nvalue="before"\n', encoding="utf-8")
        self.assertIn("说明：保留中文 📖", tools._read_file({"file_path": str(path)}))
        result = tools._edit_file({"file_path": str(path), "old_string": 'value="before"',
                                   "new_string": 'value="after"'})
        self.assertTrue(result.startswith("Successfully"), result)
        self.assertEqual(path.read_text(encoding="utf-8"), '说明：保留中文 📖\nvalue="after"\n')

    def test_permission_settings_preserve_nonascii_paths(self):
        path = self.root / "settings.json"
        settings = {"permissions": {"deny": ["write_file(中文目录/文件.txt)"]}}
        path.write_text(json.dumps(settings, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(tools._load_settings(path), settings)

    def test_provenance_text_and_binary_paths_read_the_same_utf8_records(self):
        path = self.root / "source.jsonl"
        record = {"skill": "中文技能", "description": "保留历史 📖"}
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        self.assertEqual(load_recent_sources(path, 2, persist=False), [record])
        self.assertEqual(load_recent_sources(path, 2), [record])


class TerminalEncodingTests(unittest.TestCase):
    def render(self, encoding):
        buffer = io.BytesIO()
        with io.TextIOWrapper(buffer, encoding=encoding, errors="strict", write_through=True) as stream:
            with redirect_stdout(stream):
                ui.print_tool_call("read_file", {"file_path": "中文目录/文件.txt"})
                ui.print_info("中文会话已恢复 📖")
                ui.print_assistant_text("回答：中文 📖\n")
                ui._safe_stdout_write("⠋ Thinking...\n")
            stream.flush()
            return buffer.getvalue().decode(encoding)

    def test_cp1252_output_escapes_unsupported_characters_without_interrupting_execution(self):
        output = self.render("cp1252")
        self.assertIn("read_file", output)
        self.assertIn("\\u4e2d", output)
        self.assertIn("\\U0001f4d6", output)
        self.assertIn("\\u280b", output)

    def test_utf8_output_preserves_original_text_and_honors_redirection(self):
        output = self.render("utf-8")
        self.assertIn("中文会话已恢复 📖", output)
        self.assertIn("回答：中文 📖", output)
        self.assertIn("⠋ Thinking...", output)
        self.assertNotIn("\\u4e2d", output)
