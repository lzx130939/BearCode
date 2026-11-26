"""Memory recall deduplication and budgets survive real Session save/load."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from agents import memory, session
from agents.agent import Agent
from agents.session_detail import build_session_detail


class MemoryResumeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for patcher in (
            patch("pathlib.Path.home", return_value=self.root),
            patch.object(session, "SESSION_DIR", self.root / "sessions"),
            patch("agents.agent.print_info"),
            patch("agents.agent.openai.AsyncOpenAI", side_effect=lambda **_: AsyncMock()),
            patch("agents.agent.anthropic.AsyncAnthropic", side_effect=lambda **_: AsyncMock()),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def agent(self, protocol="openai"):
        agent = Agent(api_key="offline", api_protocol=protocol, custom_system_prompt="test",
                      enable_mcp=False, enable_online_evolution=False, enable_folding=False)
        self.addAsyncCleanup(agent.aclose)
        return agent

    def restore(self, source):
        source._auto_save(strict=True)
        data = session.load_session(source.session_id)
        restored = self.agent("openai" if source.use_openai else "anthropic")
        restored.restore_session(data)
        return restored, data

    async def test_disk_round_trip_preserves_recall_and_prevents_duplicate_injection(self):
        filename = memory.save_memory("Preference", "Reply format", "user", "RECALL_MARKER 中文")
        path = str(memory.get_memory_dir() / filename)
        for protocol in ("openai", "anthropic"):
            with self.subTest(protocol=protocol):
                source = self.agent(protocol)
                side_query = AsyncMock(return_value=json.dumps({"selected_memories": [filename]}))
                with patch.object(source, "_build_side_query", return_value=side_query):
                    message = await source._recall_before_response("Please use my saved preferences")
                self.assertIn("RECALL_MARKER", message)
                messages = source._openai_messages if source.use_openai else source._anthropic_messages
                messages.append({"role": "user", "content": message})
                expected_bytes = source._session_memory_bytes
                self.assertGreater(expected_bytes, 0)

                restored, data = self.restore(source)
                self.assertEqual(data["runtimeState"]["alreadySurfacedMemories"], [path])
                self.assertEqual(data["runtimeState"]["sessionMemoryBytes"], expected_bytes)
                self.assertEqual(restored._already_surfaced_memories, {path})
                self.assertEqual(restored._session_memory_bytes, expected_bytes)
                retained = restored._openai_messages if restored.use_openai else restored._anthropic_messages
                self.assertIn("RECALL_MARKER", retained[-1]["content"])
                report = build_session_detail(restored)
                self.assertEqual(report["recalled_path_count"], 1)
                self.assertEqual(report["recalled_bytes"], expected_bytes)
                self.assertTrue(report["memories"][0]["recalled_in_session"])

                side_query.reset_mock()
                with patch.object(restored, "_build_side_query", return_value=side_query):
                    message = await restored._recall_before_response("Please use my saved preferences again")
                self.assertEqual(message, "Please use my saved preferences again")
                side_query.assert_not_awaited()
                self.assertEqual(restored._session_memory_bytes, expected_bytes)

    async def test_restored_budget_blocks_further_injection_at_and_near_limit(self):
        filename = memory.save_memory("Budget", "Recall budget", "project", "BUDGET_MARKER")
        historical_path = str(self.root / "removed-memory.md")
        for protocol in ("openai", "anthropic"):
            for remaining in (0, 1):
                with self.subTest(protocol=protocol, remaining=remaining):
                    source = self.agent(protocol)
                    source._already_surfaced_memories = {historical_path}
                    source._session_memory_bytes = memory.MAX_SESSION_MEMORY_BYTES - remaining
                    restored, _ = self.restore(source)
                    side_query = AsyncMock(return_value=json.dumps({"selected_memories": [filename]}))
                    with patch.object(restored, "_build_side_query", return_value=side_query):
                        message = await restored._recall_before_response("Please recall the project budget")
                    self.assertEqual(message, "Please recall the project budget")
                    self.assertEqual(restored._already_surfaced_memories, {historical_path})
                    self.assertEqual(restored._session_memory_bytes, memory.MAX_SESSION_MEMORY_BYTES - remaining)
                    if remaining:
                        side_query.assert_awaited_once()
                    else:
                        side_query.assert_not_awaited()

    async def test_legacy_session_resets_tracking_without_changing_saved_messages(self):
        for protocol in ("openai", "anthropic"):
            with self.subTest(protocol=protocol):
                source = self.agent(protocol)
                messages = source._openai_messages if source.use_openai else source._anthropic_messages
                messages.append({"role": "user", "content": "<system-reminder>old memory</system-reminder>"})
                source._auto_save(strict=True)
                data = session.load_session(source.session_id)
                del data["runtimeState"]["alreadySurfacedMemories"]
                del data["runtimeState"]["sessionMemoryBytes"]
                before = deepcopy(data)
                target = self.agent(protocol)
                target._already_surfaced_memories = {"another-session.md"}
                target._session_memory_bytes = 100
                target.restore_session(data)
                self.assertEqual(target._already_surfaced_memories, set())
                self.assertEqual(target._session_memory_bytes, 0)
                retained = target._openai_messages if target.use_openai else target._anthropic_messages
                self.assertIn("old memory", retained[-1]["content"])
                self.assertEqual(data, before)

    async def test_invalid_tracking_is_rejected_before_replacing_session_state(self):
        source = self.agent()
        source._auto_save(strict=True)
        data = session.load_session(source.session_id)
        target = self.agent()
        target._already_surfaced_memories = {"current-memory.md"}
        target._session_memory_bytes = 100
        before = deepcopy(target._openai_messages)
        invalid_values = (
            ("alreadySurfacedMemories", "memory.md"),
            ("alreadySurfacedMemories", [None]),
            ("alreadySurfacedMemories", [" "]),
            ("alreadySurfacedMemories", ["bad\x00path"]),
            ("alreadySurfacedMemories", None),
            ("sessionMemoryBytes", -1),
            ("sessionMemoryBytes", 1.5),
            ("sessionMemoryBytes", "721"),
            ("sessionMemoryBytes", True),
            ("sessionMemoryBytes", None),
        )
        for key, value in invalid_values:
            with self.subTest(key=key, value=value):
                broken = deepcopy(data)
                broken["runtimeState"][key] = value
                with self.assertRaisesRegex(ValueError, "Invalid session"):
                    target.restore_session(broken)
                self.assertEqual(target._already_surfaced_memories, {"current-memory.md"})
                self.assertEqual(target._session_memory_bytes, 100)
                self.assertEqual(target._openai_messages, before)

    async def test_clear_history_saves_empty_tracking_for_the_next_resume(self):
        source = self.agent()
        source._already_surfaced_memories = {"previous-memory.md"}
        source._session_memory_bytes = 721
        source.clear_history()
        restored, data = self.restore(source)
        self.assertEqual(data["runtimeState"]["alreadySurfacedMemories"], [])
        self.assertEqual(data["runtimeState"]["sessionMemoryBytes"], 0)
        self.assertEqual(restored._already_surfaced_memories, set())
        self.assertEqual(restored._session_memory_bytes, 0)
