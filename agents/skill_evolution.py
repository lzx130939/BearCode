from __future__ import annotations

import json
import os
import re
import shutil
import time
import hashlib
from pathlib import Path
from typing import Any

from .frontmatter import format_frontmatter, parse_frontmatter
from .storage import append_jsonl, atomic_write_text, state_lock, write_json


# USAGE_LOG表示的是skill演化过程的更新记录文件，包括了创建、调用、进化、反馈、归档等行为的日志记录
USAGE_LOG = "usage.jsonl"
# 在线演化日志
ONLINE_PROVENANCE_LOG = "online_provenance.jsonl"
# 按 Skill 整理的来源索引
ONLINE_PROVENANCE_INDEX = "online_skill_provenance.json"
# SKILL_USAGE_STATS存储的是以skill为key的所有信息统计
SKILL_USAGE_STATS = "skill_usage_stats.json"
HISTORY_DIR = "history"


def get_evolution_dir() -> Path:
    return Path.cwd() / ".bear" / "skill-evolution"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    append_jsonl(path, row)


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    # Never turn corrupted counters or provenance into a new, empty history.
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    write_json(path, value)


def _safe_skill_slug(name: str) -> str:
    raw = str(name or "").strip()
    # 将连续出现的、不属于“英文字母、数字、下划线、点、连字符、指定范围汉字”的字符，统一替换成一个 -
    slug = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "-", raw)
    # 将连续出现的多个 - 替换成一个 -
    slug = re.sub(r"-{2,}", "-", slug).strip("-.")
    if slug:
        return slug[:120].rstrip("-.") or slug[:120]
    if raw:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        return f"skill-{digest}"
    return "unknown"


def _preview(value: object, limit: int = 500) -> str:
    """返回一个收到limit限制的str(value)"""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _compact_messages(messages: list[dict[str, Any]], *, max_messages: int = 12, max_chars: int = 4000) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in list(messages or [])[-max_messages:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        out.append({"role": role, "content": _preview(content, max_chars)})
    return out


def record_skill_invocation(
    *,
    skill_name: str,
    source: str,
    context: str,
    args: object = "",
) -> None:
    row = {
        "event": "invoke",
        "time": _utc_now(),
        "skill": skill_name,
        "source": source,
        "context": context,
        "args_preview": _preview(args),
    }
    _append_jsonl(get_evolution_dir() / USAGE_LOG, row)


def record_skill_feedback(
    *,
    skill_name: str,
    rating: str,
    note: str = "",
) -> None:
    row = {
        "event": "feedback",
        "time": _utc_now(),
        "skill": skill_name,
        "rating": str(rating or "").strip(),
        "note": _preview(note, 1200),
    }
    _append_jsonl(get_evolution_dir() / USAGE_LOG, row)


def record_online_skill_provenance(
    *,
    action: str,
    skill_name: str = "",
    result: dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
    retrieved_reference: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    error: str = "",
    replay_context: dict[str, Any] | None = None,
) -> None:
    """根据参数组成row，然后把row追加到online_provenance.jsonl文件中，并更新online_skill_provenance.json索引"""
    row = {
        "event": "online_ingest",
        "replay_context": dict(replay_context or {}),
        "time": _utc_now(),
        "action": str(action or "none").strip() or "none",
        "skill": str(skill_name or "").strip(),
        "ok": bool((result or {}).get("ok")) if result is not None else not bool(error),
        "result": result or {},
        "messages": _compact_messages(list(messages or [])),
        "retrieved_reference": retrieved_reference or {},
        "decision": decision or {},
        "error": _preview(error, 1200),
    }
    _append_jsonl(get_evolution_dir() / ONLINE_PROVENANCE_LOG, row)
    _update_online_provenance_index(row)


def _update_online_provenance_index(row: dict[str, Any]) -> None:
    """更新skill对应的索引信息"""
    skill = str(row.get("skill") or "").strip()
    if not skill:
        return
    root = get_evolution_dir()
    path = root / ONLINE_PROVENANCE_INDEX
    with state_lock(path):
        index = _read_json(path, {})
        item = index.setdefault(
            skill,
            {
                "skill": skill,
                "source_count": 0,
                "history_count": 0,
                "sources": [],
                "version_timeline": [],
                "usage": {},
            },
        )
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        item["current_version"] = result.get("version") or item.get("current_version", "")
        item["last_action"] = row.get("action")
        item["last_time"] = row.get("time")
        item["last_ok"] = row.get("ok")
        item["last_error"] = row.get("error", "")
        item["source_count"] = int(item.get("source_count", 0)) + 1
        source = {
            "replay_context": dict(row.get("replay_context") or {}),
            "time": row.get("time"),
            "action": row.get("action"),
            "ok": row.get("ok"),
            "messages": row.get("messages", []),
            "retrieved_reference": row.get("retrieved_reference", {}),
            "decision": row.get("decision", {}),
            "error": row.get("error", ""),
        }
        sources = list(item.get("sources") or [])
        sources.append(source)
        # 每个skill的sources只保留最近的20条记录，超过20条的会被丢弃
        # 也就是说只记录这这个skill的最近20次调用的详细情况
        item["sources"] = sources[-20:]
        item["history_count"] = len(sources)
        if result.get("version"):
            # 更新version_timeline时间线记录，保留最近50条记录
            timeline = list(item.get("version_timeline") or [])
            timeline.append({"time": row.get("time"), "version": result.get("version"), "action": row.get("action")})
            item["version_timeline"] = timeline[-50:]
        index[skill] = item
        _write_json(path, index)


def _parse_int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value or "").strip())
    except Exception:
        return default


def _bump_patch(version: str | None) -> str:
    """更新版本版号"""
    raw = str(version or "0.1.0").strip()
    parts = raw.split(".")
    if len(parts) < 3 or not all(p.isdigit() for p in parts[:3]):
        return "0.1.1"
    parts[2] = str(int(parts[2]) + 1)
    return ".".join(parts[:3])


def _find_skill_file_by_name(base_dir: Path, skill_name: str) -> Path | None:
    """在base_dir目录下查找指定技能名称的*/SKILL.md文件，返回Path对象，如果没有找到则返回None"""
    if not base_dir.is_dir():
        return None
    wanted = str(skill_name or "").strip()
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        skill_file = entry / "SKILL.md"
        if not skill_file.is_file():
            continue
        try:
            parsed = parse_frontmatter(skill_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = parsed.meta.get("name") or entry.name
        if name == wanted:
            return skill_file
    return None


def resolve_skill_file(skill_name: str, *, target: str = "active", active_dir: str = "") -> Path | None:
    """根据技能名称和查找范围，定位对应的 SKILL.md 文件
    target: "active" | "project" | "user"
    如果 target=active，则在这个目录下查找 SKILL.md 文件(前提active_dir不为空)
    如果active_dir是空的，那么就直接办理project/user中的所有skill
    返回值：Path对象，指向对应的 SKILL.md 文件；如果没有找到，则返回 None
    """
    target = (target or "active").strip().lower()
    if target == "active" and active_dir:
        skill_file = Path(active_dir) / "SKILL.md"
        if skill_file.is_file():
            return skill_file

    if target in ("project", "active"):
        found = _find_skill_file_by_name(Path.cwd() / ".bear" / "skills", skill_name)
        if found:
            return found

    if target in ("user", "active"):
        found = _find_skill_file_by_name(Path.home() / ".bear" / "skills", skill_name)
        if found:
            return found

    return None


def _skills_root(target: str) -> Path:
    target = (target or "project").strip().lower()
    if target == "user":
        return Path.home() / ".bear" / "skills"
    return Path.cwd() / ".bear" / "skills"


def _normalize_context(context: str | None) -> str:
    return "fork" if str(context or "").strip().lower() == "fork" else "inline"


def _normalize_allowed_tools(value: object) -> str:
    """把allowed_tools的值转换成一个逗号分隔的字符串"""
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return ",".join(part.strip() for part in str(value).split(",") if part.strip())


def _skill_body(instructions: str, evidence: str = "") -> str:
    """构建skill的body部分
    整体结构就是：
    # Skill Instructions\\n\\n + instructions + \\n\\n## Creation Evidence\\n\\n + evidence
    """
    body = str(instructions or "").strip()
    if not body:
        body = (
            "# Goal\n\n"
            "Apply this reusable skill when the user's request matches the trigger.\n\n"
            "# Workflow\n\n"
            "1. Identify the user's concrete task and constraints.\n"
            "2. Follow the reusable rules captured in this skill.\n"
            "3. Produce the requested output directly.\n"
        )
    if not body.lstrip().startswith("#"):
        body = "# Skill Instructions\n\n" + body
    if evidence:
        body = body.rstrip() + "\n\n## Creation Evidence\n\n" + str(evidence).strip() + "\n"
    return body.rstrip() + "\n"


def create_skill_file(
    *,
    name: str,
    description: str,
    instructions: str,
    when_to_use: str = "",
    target: str = "project",
    context: str = "inline",
    user_invocable: bool = False,
    allowed_tools: object = None,
    evidence: str = "",
    actor: str = "agent",
    tags: list[str] | None = None,
    _prepared_body: str | None = None,
) -> dict[str, Any]:
    resolved_name = str(name or "").strip()
    if not resolved_name:
        return {"ok": False, "error": "skill name is required"}
    if not str(description or "").strip():
        return {"ok": False, "error": "description is required"}

    # 查看是否有重复的skill已经存在了
    existing = resolve_skill_file(resolved_name, target="active")
    if existing:
        return {"ok": False, "error": f"Skill already exists: {resolved_name}", "file": str(existing)}

    # 如果没有重复的skill存在，就创建一个新的skill文件
    root = _skills_root(target)
    skill_dir = root / _safe_skill_slug(resolved_name)
    skill_file = skill_dir / "SKILL.md"
    if skill_file.exists():
        return {"ok": False, "error": f"Skill file already exists: {skill_file}"}

    meta = {
        "name": resolved_name,
        "description": re.sub(r"\s+", " ", str(description).strip()),
        "version": "0.1.0",
        "created-at": _utc_now(),
        "user-invocable": "true" if user_invocable else "false",
        "context": _normalize_context(context),
    }
    if when_to_use:
        meta["when-to-use"] = re.sub(r"\s+", " ", str(when_to_use).strip())
    if tags:
        meta["tags"] = ",".join(re.sub(r"\s+", "-", str(tag).strip()) for tag in tags if str(tag).strip())
    tools = _normalize_allowed_tools(allowed_tools)
    if tools:
        meta["allowed-tools"] = tools

    skill_dir.mkdir(parents=True, exist_ok=False)
    # Candidate publication supplies the exact body used during evaluation.
    body = _skill_body(instructions, evidence) if _prepared_body is None else _prepared_body
    atomic_write_text(skill_file, format_frontmatter(meta, body))

    event = {
        "event": "create",
        "time": _utc_now(),
        "actor": actor,
        "skill": resolved_name,
        "file": str(skill_file),
        "target": "user" if (target or "").strip().lower() == "user" else "project",
        "description": _preview(description, 1200),
        "when_to_use": _preview(when_to_use, 1200),
        "evidence": _preview(evidence, 1200),
    }
    # 记录行为到 usage.jsonl中
    _append_jsonl(get_evolution_dir() / USAGE_LOG, event)
    return {"ok": True, **event}


def _append_evolution_note(body: str, lesson: str, rationale: str = "", *, evolved_on: str | None = None) -> str:
    """追加演进记录
    记录包含主体信息body，经验 lesson、原因 rationale 和演进日期

    返回的样式：body + "\\n\\n## Evolution Notes\\n\\n" + "- 2024-06-01: {lesson}. Reason: {rationale}"
    """
    lesson = re.sub(r"\s+", " ", str(lesson or "").strip())
    rationale = re.sub(r"\s+", " ", str(rationale or "").strip())
    if not lesson:
        raise ValueError("lesson is required")

    bullet = f"- {evolved_on if evolved_on is not None else _today()}: {lesson}"
    if rationale:
        bullet += f" Reason: {rationale}"

    body = str(body or "").rstrip()
    marker = "## Evolution Notes"
    if marker in body:
        if lesson in body:
            return body + "\n"
        return body + "\n" + bullet + "\n"
    return body + "\n\n" + marker + "\n\n" + bullet + "\n"


def evolve_skill_file(
    *,
    skill_name: str,
    lesson: str,
    rationale: str = "",
    target: str = "active",
    active_dir: str = "",
    actor: str = "agent",
    instructions: str = "",
    description: str = "",
    when_to_use: str = "",
    tags: list[str] | None = None,
    preserve_existing: bool = False,
    _prepared_body: str | None = None,
) -> dict[str, Any]:
    """
    更新一个已有的技能文件

    params:
    skill_name	要更新的技能名称
    lesson	本次新增的经验
    rationale	为什么要加入这条经验
    target、`active_dir`	控制到哪里寻找技能文件
    actor	操作者标记，用于记录日志，默认 "agent"
    instructions	准备替换进去的完整技能正文
    description、`when_to_use`、`tags`	要更新的技能元信息
    _prepared_body	已经整理好的最终正文，优先级最高
    preserve_existing	当前函数中没有实际使用，后面会解释
    """
    skill_file = resolve_skill_file(skill_name, target=target, active_dir=active_dir)
    if not skill_file:
        return {"ok": False, "error": f"Skill not found: {skill_name}"}

    with state_lock(skill_file):
        raw = skill_file.read_text(encoding="utf-8")
        parsed = parse_frontmatter(raw)
        meta = dict(parsed.meta)
        resolved_name = meta.get("name") or skill_file.parent.name

        # 保存基于旧版的快照，并把它写到history_path中
        snapshot = {
            "time": _utc_now(),
            "event": "snapshot",
            "actor": actor,
            "skill": resolved_name,
            "file": str(skill_file),
            "version": meta.get("version", "0.1.0"),
            "lesson": _preview(lesson, 1200),
            "rationale": _preview(rationale, 1200),
            "content": raw,
        }
        # history/<技能名>.jsonl 修改前的完整旧文件，用于追溯和回滚
        history_path = get_evolution_dir() / HISTORY_DIR / f"{_safe_skill_slug(resolved_name)}.jsonl"
        _append_jsonl(history_path, snapshot)

        # 技能进化，更新元信息
        meta["name"] = resolved_name
        meta["version"] = _bump_patch(meta.get("version"))
        meta["last-evolved"] = _utc_now()
        meta["evolution-count"] = str(_parse_int(meta.get("evolution-count"), 0) + 1)
        if description.strip():
            meta["description"] = re.sub(r"\s+", " ", description.strip())
        if when_to_use.strip():
            meta["when-to-use"] = re.sub(r"\s+", " ", when_to_use.strip())
        # 如果有新的标签
        if tags:
            existing_tags = [part.strip() for part in str(meta.get("tags") or "").split(",") if part.strip()]
            merged_tags = existing_tags[:]
            for tag in tags:
                normalized = re.sub(r"\s+", "-", str(tag).strip())
                if normalized and normalized not in merged_tags:
                    merged_tags.append(normalized)
            if merged_tags:
                meta["tags"] = ",".join(merged_tags[:12])

        # 更新技能正文(主要跟_prepared_body，instructions，lesson，rationale有关)
        if _prepared_body is not None:
            new_body = _prepared_body
        elif instructions.strip():
            new_body = _skill_body(instructions.strip())
            # The maintainer supplies a complete merged body. Preserve previous
            # instructions in the snapshot, not as conflicting active guidance.
            note = _append_evolution_note("", lesson, rationale).strip()
            if note:
                new_body = new_body.rstrip() + "\n\n" + note + "\n"
        else:
            new_body = _append_evolution_note(parsed.body, lesson, rationale)
        # 重新更新对应的skill.md文件
        atomic_write_text(skill_file, format_frontmatter(meta, new_body))

        event = {
            "event": "evolve",
            "time": _utc_now(),
            "actor": actor,
            "skill": resolved_name,
            "file": str(skill_file),
            "version": meta["version"],
            "target": target,
            "lesson": _preview(lesson, 1200),
            "rationale": _preview(rationale, 1200),
            "history": str(history_path),
        }
        # usage.jsonl 记录技能的演进事件
        _append_jsonl(get_evolution_dir() / USAGE_LOG, event)
        return {"ok": True, **event}


def record_skill_usage_judgments(judgments: list[dict[str, Any]], *, allow_prune: bool = False) -> dict[str, Any]:
    """记录技能的使用判断结果，并统计使用情况到 skill_usage_stats.json 中
    返回judgements的数量和pruned(归档)的技能列表

    judgments：内部的结构
    [{
        "name": name,
        "source": hit.get("source", ""),
        "skill_dir": hit.get("skill_dir", ""),
        "retrieved": True,
        "relevant": bool(raw.get("relevant")),
        "used": bool(raw.get("used")),
        "score": float(hit.get("score", 0.0)),
        "reason": str(raw.get("reason") or ""),
    }]
    """
    # 先对judgements进行检查，保证used/relevant字段都需要为bool类型，并且如果used为True，则relevant也必须为False
    if any(type(item.get("relevant")) is not bool or type(item.get("used")) is not bool
           or (item["used"] and not item["relevant"]) for item in judgments):
        raise ValueError("Skill usage judgments require consistent boolean values")
    stats_path = get_evolution_dir() / SKILL_USAGE_STATS
    # 对于每个judgment，更新skill_usage_stats.json中的对应记录
    with state_lock(stats_path):
        stats = _read_json(stats_path, {})
        pruned: list[str] = []
        for judgment in judgments:
            skill = str(judgment.get("name") or judgment.get("skill") or "").strip()
            if not skill:
                continue
            # 获取文件种原本对应的skill的记录信息
            item = stats.setdefault(
                skill,
                {
                    "retrieved": 0,
                    "relevant": 0,
                    "used": 0,
                    "last_retrieved": "",
                    "last_used": "",
                    "source": judgment.get("source", ""),
                    "skill_dir": judgment.get("skill_dir", ""),
                },
            )
            # 根据judgement的内容，更新对应的skill的使用统计信息
            item["retrieved"] = int(item.get("retrieved", 0)) + 1
            item["last_retrieved"] = _utc_now()
            item["source"] = judgment.get("source", item.get("source", ""))
            item["skill_dir"] = judgment.get("skill_dir", item.get("skill_dir", ""))
            if judgment.get("relevant"):
                item["relevant"] = int(item.get("relevant", 0)) + 1
            if judgment.get("used"):
                item["used"] = int(item.get("used", 0)) + 1
                item["last_used"] = _utc_now()
            item["last_reason"] = _preview(judgment.get("reason", ""), 500)
            item["last_score"] = judgment.get("score", 0)
            # 判断是否进行归档
            if allow_prune and _maybe_prune_stale_skill(skill, item):
                pruned.append(skill)
        # 除了更新skill_usage_stats.json之外，
        # 还需要更新online_skill_provenance.json索引中所有的的usage统计，确保技能的使用记录和状态在不同的地方保持一致。
        _write_json(stats_path, stats)
        _sync_usage_into_provenance(stats)
        return {"ok": True, "judgments": len(judgments), "pruned": pruned}


def restore_pruned_skill(skill_name: str) -> dict[str, Any]:
    """
    这个函数负责把已归档的技能目录移回原来的位置，同时重置使用统计并记录恢复日志。
    Restore an archive after the caller has authorized this file operation."""
    stats_path = get_evolution_dir() / SKILL_USAGE_STATS
    with state_lock(stats_path):
        stats = _read_json(stats_path, {})
        item = stats.get(skill_name, {})
        # pruned	技能是否已被裁剪、归档
        # pruned_to	归档后的技能目录路径
        # skill_dir	技能原来的目录路径
        # 如果没有归档的技能，或者归档路径不存在，或者原始技能目录不存在，则返回错误信息
        if not item.get("pruned") or not item.get("pruned_to") or not item.get("skill_dir"):
            return {"ok": False, "error": "No archived skill found"}
        archive = Path(item["pruned_to"]).resolve()
        original = Path(item["skill_dir"]).resolve()
        archive_root = (get_evolution_dir() / "pruned").resolve()
        source_root = _skills_root(item.get("source", "project")).resolve()
        # 如果归档目录/原始目录不在预期的归档/技能目录下，则返回错误信息
        if archive.parent != archive_root or original.parent != source_root:
            return {"ok": False, "error": "Archive paths are outside the expected skill directories"}
        # 原始路径已存在，避免覆盖现有内容。
        # 归档目录里没有 SKILL.md 文件，说明缺少预期的技能入口文件。
        if original.exists() or not (archive / "SKILL.md").is_file():
            return {"ok": False, "error": "Original path is occupied or archive is missing"}
        original.parent.mkdir(parents=True, exist_ok=True)
        # 将整个归档目录移回原始位置，包括其中的 SKILL.md 和其他资源文件。成功后，旧的归档位置通常就不再保留该目录。
        shutil.move(str(archive), str(original))
        # 把原本已经归档的技能的使用统计重置为初始状态，表示该技能已恢复为可用状态。
        item.update(pruned=False, pruned_to="", retrieved=0, relevant=0, used=0,
                    last_retrieved="", last_used="")
        # 重新更新SKILL_USAGE_STATS
        _write_json(stats_path, stats)
        # restore的操作也需要记录到usage.jsonl中，方便追踪技能的恢复历史
        _append_jsonl(get_evolution_dir() / USAGE_LOG,
                      {"event": "restore", "time": _utc_now(), "skill": skill_name,
                       "from": str(archive), "to": str(original)})
        # 同步更新online_skill_provenance.json索引中所有的的usage统计，确保技能的使用记录和状态在不同的地方保持一致。
        _sync_usage_into_provenance(stats)
        return {"ok": True, "skill": skill_name, "file": str(original / "SKILL.md")}


def _maybe_prune_stale_skill(skill_name: str, stats: dict[str, Any]) -> bool:
    """_maybe_prune_stale_skill的作用是把“经常被检索到、但很少或从未被使用”的技能目录移到归档区，并记录这次清理。"""
    # 获取环境变量
    # min_retrieved	技能被归档的最小检索次数，默认40
    # max_used	技能被归档的最大使用次数，默认0
    # 这两个环境变量的意义是给技能足够的被选择机会，避免把“还没怎么被发现”的技能误判为无用技能。
    min_retrieved = _parse_int(os.environ.get("BEAR_SKILL_USAGE_PRUNE_MIN_RETRIEVED"), 40)
    max_used = _parse_int(os.environ.get("BEAR_SKILL_USAGE_PRUNE_MAX_USED"), 0)
    source = str(stats.get("source") or "").strip().lower()
    # 如果这个skill不是user级别同时BEAR_SKILL_PRUNE_PROJECT(表示项目级别的技能修剪)这个条件变量不是"开启"的情况下直接不归档
    if source != "user" and os.environ.get("BEAR_SKILL_PRUNE_PROJECT", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    if int(stats.get("retrieved", 0)) < min_retrieved:
        return False
    if int(stats.get("used", 0)) > max_used:
        return False
    skill_dir = Path(str(stats.get("skill_dir") or ""))
    if not skill_dir.is_dir():
        return False
    # 校验kill的文件Path最后的文件名
    if skill_dir.name.startswith("."):
        return False
    archive_root = get_evolution_dir() / "pruned"
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / f"{_safe_skill_slug(skill_name)}-{int(time.time())}"
    try:
        # 把skill_dir移动到归档目录destination中，shutil.move会把整个目录及其内容都移动过去
        shutil.move(str(skill_dir), str(destination))
    except Exception:
        return False
    # 对归档的记录进行一个存储持久化
    event = {
        "event": "prune",
        "time": _utc_now(),
        "skill": skill_name,
        "from": str(skill_dir),
        "to": str(destination),
        "retrieved": stats.get("retrieved", 0),
        "used": stats.get("used", 0),
    }
    _append_jsonl(get_evolution_dir() / USAGE_LOG, event)
    stats["pruned"] = True
    stats["pruned_to"] = str(destination)
    return True


def _sync_usage_into_provenance(stats: dict[str, Any]) -> None:
    """把技能使用统计同步到online_skill_provenance.json索引中"""
    path = get_evolution_dir() / ONLINE_PROVENANCE_INDEX
    with state_lock(path):
        index = _read_json(path, {})
        if not isinstance(index, dict):
            return
        changed = False
        for skill, usage in stats.items():
            if skill in index:
                index[skill]["usage"] = usage
                changed = True
        if changed:
            _write_json(path, index)


def load_skill_stats() -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    # 从 usage.jsonl 中读取技能的使用记录，统计每个技能的创建、调用、反馈和演化次数
    usage_path = get_evolution_dir() / USAGE_LOG
    if usage_path.is_file():
        for line in usage_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            skill = str(row.get("skill") or "").strip()
            if not skill:
                continue
            item = stats.setdefault(skill, {"created": 0, "invocations": 0, "feedback": 0, "evolutions": 0})
            event = row.get("event")
            if event == "create":
                item["created"] = int(item.get("created", 0)) + 1
                item["created_at"] = row.get("time")
                item["file"] = row.get("file")
            elif event == "invoke":
                item["invocations"] = int(item.get("invocations", 0)) + 1
                item["last_invoked"] = row.get("time")
            elif event == "feedback":
                item["feedback"] = int(item.get("feedback", 0)) + 1
                item["last_feedback"] = row.get("time")
            elif event == "evolve":
                item["evolutions"] = int(item.get("evolutions", 0)) + 1
                item["last_evolved"] = row.get("time")
                item["version"] = row.get("version")
                item["file"] = row.get("file")

    # 从history目录中读取每个技能的快照数量(本质是统计有多少历史信息)
    history_root = get_evolution_dir() / HISTORY_DIR
    if history_root.is_dir():
        for path in history_root.glob("*.jsonl"):
            count = len([line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()])
            skill = path.stem
            item = stats.setdefault(skill, {"created": 0, "invocations": 0, "feedback": 0, "evolutions": 0})
            item["snapshots"] = count
    usage_stats = _read_json(get_evolution_dir() / SKILL_USAGE_STATS, {})
    if isinstance(usage_stats, dict):
        for skill, usage in usage_stats.items():
            if not isinstance(usage, dict):
                continue
            # 检索次数 retrieved、相关次数 relevant、使用次数 used、最近检索/使用时间、是否被裁剪 pruned
            item = stats.setdefault(str(skill), {"created": 0, "invocations": 0, "feedback": 0, "evolutions": 0})
            item["retrieved"] = usage.get("retrieved", 0)
            item["relevant"] = usage.get("relevant", 0)
            item["used"] = usage.get("used", 0)
            item["last_retrieved"] = usage.get("last_retrieved", "")
            item["last_used"] = usage.get("last_used", "")
            item["pruned"] = usage.get("pruned", False)
    return stats


def format_skill_stats() -> str:
    stats = load_skill_stats()
    if not stats:
        return "No skill evolution events recorded yet."

    lines = ["Skill evolution stats:"]
    for name in sorted(stats):
        item = stats[name]
        parts = [
            f"created={item.get('created', 0)}",
            f"invoked={item.get('invocations', 0)}",
            f"feedback={item.get('feedback', 0)}",
            f"evolved={item.get('evolutions', 0)}",
            f"snapshots={item.get('snapshots', 0)}",
            f"retrieved={item.get('retrieved', 0)}",
            f"used={item.get('used', 0)}",
        ]
        if item.get("created_at"):
            parts.append(f"created_at={item['created_at']}")
        if item.get("version"):
            parts.append(f"version={item['version']}")
        if item.get("last_invoked"):
            parts.append(f"last_invoked={item['last_invoked']}")
        if item.get("last_used"):
            parts.append(f"last_used={item['last_used']}")
        if item.get("pruned"):
            parts.append("pruned=true")
        lines.append(f"  {name}: " + ", ".join(parts))
    return "\n".join(lines)
