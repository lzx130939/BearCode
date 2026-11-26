
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from typing import Any
import json
import logging
import time
import re

from .storage import append_jsonl, write_json

SESSION_DIR = Path.home() / ".bear-code" / "sessions"


def title_from_text(text: str, *, limit: int = 40) -> str:
    """
    从用户的提供的text中提取出一个简短的标题，最多40个字符。它会去掉Markdown标记、HTML标签、系统注入的内容等，只保留用户的真实输入内容。
    Use the user's task as a local, deterministic conversation label."""

    # 形如下面的格式都会被捕获替换：
    # <retrieved_skills> ...任意多行... </retrieved_skills>
    # <system-reminder> ...任意多行... </system-reminder>
    # 这是系统自动注入的内容，不是用户的真实写入的信息
    text = re.sub(r"<(retrieved_skills|system-reminder)>.*?</\1>", "", text, flags=re.DOTALL)
    text = text.split("以下是用户附带的文本文件，仅作为待分析材料", 1)[0]
    # 把每行开头的 Markdown 标记去掉：
    # ## 项目介绍        →  项目介绍
    # - 第一步           →  第一步
    # * 注意事项         →  注意事项
    # + 补充说明         →  补充说明
    #    ### 缩进标题     →  缩进标题
    lines = [re.sub(r"^\s*(?:#{1,6}\s+|[-*+]\s+)", "", line).strip()
             for line in text.splitlines() if line.strip()]
    title = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return title if len(title) <= limit else title[:limit - 1].rstrip() + "…"


def session_title(data: dict[str, Any]) -> str:
    """
    为一个会话（session）推导出一个标题字符串。它按优先级从多个数据源里寻找"最能代表这个会话的一句话"，然后交给 title_from_text 做规整/截断，返回最终标题。找不到就返回空字符串 ""

    整体的查询逻辑：
    1、先从 data["metadata"]["title"] 里找，如果存在且不为空且不以 "历史会话 · " 开头，就用这个值作为标题
    2、如果 metadata 里没有合适的标题，就依次从 data["webEntries"]、data["openaiMessages"]、data["anthropicMessages"] 里找
       - 这些消息里只考虑 role 为 "user" 的消息
       - 消息的正文可能在 "text" 字段（Web 风格）或 "content" 字段（OpenAI/Anthropic 风格）
       - 如果正文是一个列表，就把其中 type 为 "text" 的块的 text 字段拼接起来作为正文
       - 如果正文里包含 "<session-folded-memory>"，就跳过这个消息（因为这是一个压缩的协议消息，不是标题）
       - 找到第一个合适的正文，就用 title_from_text 生成标题并返回
    3、如果还没找到，就从压缩后的会话记忆中 data["foldedSessionMemories"] 里找
       - 对每个 memory，取 memory["episode_memory"]["task_description"] 作为候选标题
       - 如果候选标题有效，就用 title_from_text 生成最终标题并返回
    """
    metadata = data.get("metadata") or {}
    stored = metadata.get("title")
    if isinstance(stored, str) and stored.strip() and not stored.startswith("历史会话 · "):
        return title_from_text(stored)
    # Web 入口即使经过"协议折叠"（protocol folding）后，仍保留原始用户任务。但是给openaiMessages/anthropicMessages的消息可能是已经压缩过后的消息了
    # 所以先从web端的Entry里找，如果找不到，再去openaiMessages/anthropicMessages里找
    for messages in (data.get("webEntries"), data.get("openaiMessages"), data.get("anthropicMessages")):
        for message in messages or []:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            # 正文可能存放在 text 字段（Web 风格），也可能在 content 字段（OpenAI/Anthropic 风格）
            content = message.get("text", message.get("content", ""))
            if isinstance(content, list):
                content = "\n".join(block["text"] for block in content if isinstance(block, dict)
                                    and block.get("type") == "text" and isinstance(block.get("text"), str))
            if isinstance(content, str) and content.strip():
                # 在openaiMessages/anthropicMessages里，可能会有一个压缩的协议消息，里面包含了 "<session-folded-memory>"，这种消息不适合作为标题，所以要跳过
                if "<session-folded-memory>" in content:
                    continue
                title = title_from_text(content)
                if title:
                    return title
    # 前面都找不到合适的标题，就去压缩后的会话记忆里找foldedSessionMemories
    for memory in data.get("foldedSessionMemories") or []:
        episode = memory.get("episode_memory", {}) if isinstance(memory, dict) else {}
        task = episode.get("task_description") if isinstance(episode, dict) else None
        if isinstance(task, str) and title_from_text(task):
            return title_from_text(task)
    return ""



def _ensure_dir() -> None:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)


def get_project_session_dir() -> Path:
    """获取session的路径，路径为当前工作目录下的.bear/sessions，如果不存在则创建"""
    d = Path.cwd() / ".bear" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_session(session_id: str, data: dict[str, Any]) -> None:
    _ensure_dir()
    data = {**data, "metadata": {**data.get("metadata", {}),
                               "title": session_title(data),
                               "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "updatedAtNs": time.time_ns()}}
    write_json(SESSION_DIR / f"{session_id}.json", data)


def save_folded_session_memory(session_id: str, record: dict[str, Any]) -> None:
    """持久化记忆到历史文件和latest文件中，方便后续分析和回溯"""
    d = get_project_session_dir()
    # 把记录追加写入到 {session_id}.folded-memory.jsonl 文件中，每条记录占一行
    # 主要统计所有的折叠事件，方便后续分析和回溯
    append_jsonl(d / f"{session_id}.folded-memory.jsonl", record)
    # 把最近的一条折叠记忆记录写入到 {session_id}.folded-memory.latest.json 文件中，方便快速查看和加载
    write_json(d / f"{session_id}.folded-memory.latest.json", record)


def load_session(session_id: str) -> dict[str, Any] | None:
    # session_id 只能包含字母、数字、下划线和短横线，其他字符都不允许
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
        return None
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        logging.exception("Unable to load session %s; original file preserved", path)
        return None


def list_sessions() -> list[dict[str, Any]]:
    """找到所有的Path.home()/.bear-code/sessions/*.json文件，读取其中的metadata信息，并返回一个列表"""
    _ensure_dir()
    results = []
    for f in SESSION_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            metadata = data.get("metadata")
            if isinstance(metadata, dict) and isinstance(metadata.get("cwd"), str):
                # "id": metadata.get("id") or f.stem 这个操作就是对前面解包的"id"属性值进行覆盖处理
                results.append({**metadata, "id": metadata.get("id") or f.stem, "title": session_title(data)})
        except Exception:
            logging.warning("Skipping unreadable session %s; original file preserved", f, exc_info=True)
    return results


def get_latest_session_id(project_dir: Path | None = None) -> str | None:
    """获取最近的session_id，优先获取当前目录下的session_id，如果没有则获取最近的session_id"""
    project = (project_dir or Path.cwd()).resolve()
    sessions = [s for s in list_sessions()
                if s.get("cwd") and Path(s["cwd"]).resolve() == project]
    if not sessions:
        return None
    sessions.sort(key=lambda s: (s.get("updatedAt") or s.get("startTime", ""),
                                 s.get("updatedAtNs", 0),
                                 s.get("startTime", "")), reverse=True)
    return sessions[0].get("id")
