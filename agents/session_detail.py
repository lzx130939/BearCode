"""Read-only reports of the active session, tool records, and project memory."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import memory, session
from .ui import _safe_text, console


def _message_calls(message: dict, use_openai: bool) -> list[dict]:
    if message.get("role") != "assistant":
        return []
    calls = []
    if use_openai:
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            if not isinstance(function, dict):
                function = {}
            calls.append({"id": str(call.get("id") or ""),
                          "name": str(function.get("name") or "unknown"),
                          "arguments": deepcopy(function.get("arguments", ""))})
    elif isinstance(message.get("content"), list):
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append({"id": str(block.get("id") or ""),
                              "name": str(block.get("name") or "unknown"),
                              "arguments": deepcopy(block.get("input", {}))})
    return calls


def _message_results(message: dict, use_openai: bool) -> list[dict]:
    if use_openai and message.get("role") == "tool":
        return [{"id": str(message.get("tool_call_id") or ""),
                 "content": deepcopy(message.get("content"))}]
    results = []
    if not use_openai and message.get("role") == "user" and isinstance(message.get("content"), list):
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results.append({"id": str(block.get("tool_use_id") or ""),
                                "content": deepcopy(block.get("content"))})
    return results


def build_session_detail(agent) -> dict:
    """Snapshot current retained context without saving or calling a model."""
    use_openai = agent.use_openai
    messages = deepcopy(agent._openai_messages if use_openai else agent._anthropic_messages)
    checkpoints = deepcopy(agent._tool_checkpoints)
    folds = deepcopy(agent._folded_session_memories)
    calls = {}
    results = defaultdict(list)
    call_occurrences = 0
    for index, message in enumerate(messages, 1):
        for position, call in enumerate(_message_calls(message, use_openai), 1):
            call_occurrences += 1
            # Older records can lack IDs. Keep each such occurrence distinct.
            key = ("id", call["id"]) if call["id"] else ("position", index, position)
            if key not in calls:
                calls[key] = {**call, "message_indices": [], "results": []}
            calls[key]["message_indices"].append(index)
        for result in _message_results(message, use_openai):
            results[result["id"]].append({**result, "message_index": index})

    tool_summary = {}
    matched_result_ids = set()
    for call in calls.values():
        if call["id"]:
            call["results"] = results.get(call["id"], [])
            matched_result_ids.add(call["id"])
        call["checkpoint"] = checkpoints.get(call["id"], {})
        totals = tool_summary.setdefault(call["name"], {"calls": 0, "with_response": 0, "without_response": 0})
        totals["calls"] += 1
        totals["with_response"] += bool(call["results"])
        totals["without_response"] += not bool(call["results"])

    memory_directory = memory.get_memory_dir(create=False)
    surfaced_paths = {Path(path).resolve() for path in agent._already_surfaced_memories}
    memories = []
    for entry in memory.list_memories(create_dir=False):
        path = memory_directory / entry.filename
        memories.append({"name": entry.name, "type": entry.type,
                         "description": entry.description, "filename": entry.filename,
                         "path": str(path), "content": entry.content,
                         "recalled_in_session": path.resolve() in surfaced_paths})

    session_file = session.SESSION_DIR / f"{agent.session_id}.json"
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "session_id": agent.session_id,
        "conversation_id": agent._conversation_id,
        "title": agent._session_title,
        "project": str(Path.cwd()),
        "protocol": "openai" if use_openai else "anthropic",
        "model": agent.model,
        "permission_mode": agent.permission_mode,
        "session_file": str(session_file),
        "session_file_exists": session_file.is_file(),
        "messages": messages,
        "message_counts": dict(Counter(str(message.get("role") or "unknown") for message in messages)),
        "tools": list(calls.values()),
        "tool_summary": tool_summary,
        "call_occurrences": call_occurrences,
        "unmatched_results": sum(len(items) for key, items in results.items() if key not in matched_result_ids),
        "checkpoints": checkpoints,
        "folds": folds,
        "memory_directory": str(memory_directory),
        "memories": memories,
        "recalled_path_count": len(surfaced_paths),
        "recalled_bytes": agent._session_memory_bytes,
    }


def _display(value) -> Text:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return Text(_safe_text(text))


def _print_table(title: str, columns: list[str], rows) -> None:
    table = Table(title=Text(title), box=box.SIMPLE_HEAVY, header_style="bold cyan")
    for column in columns:
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(*(Text(_safe_text(value)) for value in row))
    console.print(table)


def print_session_detail(report: dict, *, full: bool = False) -> None:
    """Render literal text so message contents cannot become Rich markup."""
    console.print()
    _print_table("Session Detail · Current Session Snapshot", ["Field", "Value"], [
        ("Generated at", report["generated_at"]),
        ("Session ID", report["session_id"]),
        ("Conversation ID", report["conversation_id"]),
        ("Title", report["title"] or "Untitled"),
        ("Project directory", report["project"]),
        ("Protocol / model", f'{report["protocol"]} / {report["model"]}'),
        ("Permission mode", report["permission_mode"]),
        ("Session file", report["session_file"]),
        ("Saved file status", "Present" if report["session_file_exists"] else "Not saved yet"),
        ("Messages / unique tool calls", f'{len(report["messages"])} / {len(report["tools"])}'),
        ("Call records / unmatched responses", f'{report["call_occurrences"]} / {report["unmatched_results"]}'),
        ("Checkpoints / session summaries", f'{len(report["checkpoints"])} / {len(report["folds"])}'),
        ("Project memories / recalled paths in this session", f'{len(report["memories"])} / {report["recalled_path_count"]}'),
        ("Memory bytes injected in this session", report["recalled_bytes"]),
    ])
    rows = [(name, totals["calls"], totals["with_response"], totals["without_response"])
            for name, totals in sorted(report["tool_summary"].items())]
    _print_table("Tool usage summary", ["Tool", "Unique calls", "With response", "Without response"], rows)
    if full:
        for index, message in enumerate(report["messages"], 1):
            console.print(Panel(_display(message), title=Text(f"Message #{index}"), border_style="cyan"))
        for number, call in enumerate(report["tools"], 1):
            checkpoint = call["checkpoint"]
            heading = f'Tool #{number} · {call["name"]} · ID {call["id"] or "missing"}'
            console.print(Panel(_display(call["arguments"]), title=Text(heading), border_style="yellow"))
            console.print(Text(f'Messages {call["message_indices"]} · Checkpoint {checkpoint.get("status", "no current record")}'))
            for result in call["results"]:
                console.print(Panel(_display(result["content"]),
                                    title=Text(f'Result · Message #{result["message_index"]}'), border_style="dim"))
            if not call["results"] and isinstance(checkpoint.get("result"), str):
                console.print(Panel(_display(checkpoint["result"]), title="Result saved in checkpoint", border_style="dim"))
    _print_table("Current tool checkpoints", ["Call ID", "Tool", "State", "Failed", "Result file"], [
        (call_id, record.get("name", ""), record.get("status", ""),
         record.get("failed", "Not recorded"), record.get("result_file", ""))
        for call_id, record in report["checkpoints"].items()
    ])

    console.print(Text(f'Project memory directory {report["memory_directory"]}'))
    _print_table("Project memory entries", ["File", "Type", "Name", "Recalled in this session"], [
        (entry["filename"], entry["type"], entry["name"], "Yes" if entry["recalled_in_session"] else "Not recorded")
        for entry in report["memories"]
    ])
    for entry in report["memories"]:
        console.print(Text(f'{entry["path"]}\n{entry["description"]}'))
        console.print(Panel(_display(entry["content"]), title=Text(entry["name"]), border_style="green"))
    if not full:
        console.print(Text(
            "Message and tool counts cover the current retained messages. Tool responses do not guarantee successful execution.\n"
            "Use /session-detail -d for full messages and individual tool call details.",
            style="dim",
        ))
        return
    for index, record in enumerate(report["folds"], 1):
        console.print(Panel(_display(record), title=f"Session summary #{index}", border_style="blue"))
    console.print(Text(
        "Message and tool counts cover messages retained by the current Agent. Calls are deduplicated by ID, and ID-less records are counted individually.\n"
        "Responses may include denial, skipped execution, or unknown status. Verify actual side effects using checkpoints and the environment.\n"
        "Checkpoints describe the latest tool batch. Memory recall counts and bytes cover the saved session state and subsequent injections.\n"
        "Older sessions without recall tracking fields start these counters at zero; retained messages may already contain injected memory.\n"
        "This report uses a runtime snapshot. Saved-file contents can be compared separately with the runtime state.",
        style="dim",
    ))
