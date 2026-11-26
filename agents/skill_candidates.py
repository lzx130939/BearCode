"""Isolated online Skill proposals and explicit publish/rollback operations."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import uuid

from .skill_evolution import get_evolution_dir, resolve_skill_file, _safe_skill_slug
from .storage import atomic_write_text, state_lock, write_json


def _hash_file(path: Path | None) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None


def _path(candidate_id: str) -> Path:
    """溯源每一个candidate_id对应的文件路径"""
    # 这个正则表达是的意思是：必须完全匹配一个长度为24的字符串，且只能包含小写字母a-f和数字0-9
    if not re.fullmatch(r"[a-f0-9]{24}", candidate_id):
        raise ValueError("invalid candidate id")
    return get_evolution_dir() / "candidates" / f"{candidate_id}.json"


def stage_candidate(action: str, write_input: dict) -> dict:
    """持久化candidate的proposal对象，返回一个字典，包含ok、action、skill、candidate_id和file等信息"""
    from .skills import get_skill_by_name
    name = write_input.get("skill_name") or write_input.get("name")
    skill = get_skill_by_name(name)
    base = Path(skill.skill_dir) / "SKILL.md" if skill else resolve_skill_file(name)
    candidate_id = uuid.uuid4().hex[:24]
    proposal = {"id": candidate_id, "status": "pending", "action": action, "skill": name,
                "write_input": write_input, "base_hash": _hash_file(base)}
    _validate_candidate(proposal, candidate_id)
    _freeze_snapshot(proposal)
    write_json(_path(candidate_id), proposal)
    return {"ok": True, "action": action + "_staged", "skill": name,
            "candidate_id": candidate_id, "file": str(_path(candidate_id))}


def list_candidates() -> list[dict]:
    root = get_evolution_dir() / "candidates"
    return [read_candidate(path.stem) for path in sorted(root.glob("*.json"))]


def read_candidate(candidate_id: str) -> dict:
    """读取一个候选者的proposal对象，并校验其合法性，返回candidate_id对应的proposal对象"""
    path = _path(candidate_id)
    with state_lock(path):
        candidate = json.loads(path.read_text(encoding="utf-8"))
        _validate_candidate(candidate, candidate_id)
        # 每一个候选者都需要绑定好他们进行比较的快照(因为只有比快照更好才会进行进化操作)，所以必须要有snapshot和snapshot_date字段
        if "snapshot" not in candidate and "snapshot_date" not in candidate:
            # Old proposals have no reliable creation/evaluation date. Freeze on
            # first use; their old champion cannot authorize this new snapshot.
            # 更新snapshot_date为今天，更新snapshot为当前对象生成的快照结果
            _freeze_snapshot(candidate)
            write_json(path, candidate)
        candidate_snapshot(candidate)
        return candidate


def _validate_candidate(candidate: dict, candidate_id: str) -> None:
    """纯粹校验这个candidate参数需求是否合适，不合适抛出错误"""
    if (not isinstance(candidate, dict) or candidate.get("id") != candidate_id
            or candidate.get("action") not in {"add", "merge"}
            or candidate.get("status") not in {"pending", "applied"}
            or not isinstance(candidate.get("write_input"), dict)
            or not isinstance(candidate.get("skill"), str) or "base_hash" not in candidate
            or not (candidate["base_hash"] is None or isinstance(candidate["base_hash"], str))):
        raise ValueError("invalid Skill candidate")
    inp = candidate["write_input"]
    if (not isinstance(inp.get("instructions"), str) or not inp["instructions"].strip()
            or (inp.get("skill_name") or inp.get("name")) != candidate["skill"]):
        raise ValueError("invalid candidate instructions or identity")
    from .tools import tool_definitions
    from .validation import validate_json
    from copy import deepcopy
    # 可以看到候选者只有两种情况，一种是更新"merge"，另一种是新建"create"
    tool = "skill_evolve" if candidate["action"] == "merge" else "skill_create"
    schema = deepcopy(next(item["input_schema"] for item in tool_definitions if item["name"] == tool))
    schema["properties"].update({"instructions": {"type": "string"}, "description": {"type": "string"},
        "when_to_use": {"type": "string"}, "tags": {"type": ["array", "null"], "items": {"type": "string"}},
        "actor": {"type": "string"}, "preserve_existing": {"type": "boolean"}})
    # merge需要保留"preserve_existing"决定是否保留原有的技能内容，而create需要保留"actor"决定是谁创建的技能
    schema["properties"].pop("actor" if tool == "skill_evolve" else "preserve_existing", None)
    schema["additionalProperties"] = False
    error = validate_json(inp, schema)
    if error:
        raise ValueError(f"invalid candidate parameters: {error}")


def _freeze_snapshot(proposal: dict) -> None:
    """冻结快照"""
    from .skill_evolution import _today
    proposal["snapshot_date"] = _today()
    proposal["snapshot"] = _render_snapshot(proposal)


def _render_snapshot(proposal: dict) -> dict:
    """生成当前proposal的快照
    快照模板如下(这是merge的情况，add情况去掉note结果)

    {
      "name": "python_error_analysis",
      "description": "分析 Python 异常并提供修复建议",
      "instructions": "# Skill Instructions\n\n分析异常时，先读取完整堆栈，再定位相关变量与调用链。\n\n## Creation Evidence\n\n 创建的证据\n\n## Evolution Notes\n\n - 2026-11-04: 遇到类型错误时，优先检查函数边界处的参数类型。 Reason: 错误可能来自上游调用方，仅检查报错行容易遗漏根因。",
      "candidate_id": "0123456789abcdef01234567",
      "proposal_hash": "<程序计算的64位SHA-256值>"
    }
    """
    from .skill_evolution import _skill_body, _append_evolution_note
    inp = proposal["write_input"]
    body = _skill_body(inp.get("instructions", ""), inp.get("evidence", "") if proposal["action"] == "add" else "")
    # 如果当前的对象是merge，那么就需要追加记录
    if proposal["action"] == "merge":
        note = _append_evolution_note("", inp.get("lesson", ""), inp.get("rationale", ""),
                                      evolved_on=proposal["snapshot_date"]).strip()
        if note:
            body = body.rstrip() + "\n\n" + note + "\n"
    # 通过几个键值对生成一个唯一的哈希值，作为该候选者的proposal_hash
    identity = {key: proposal[key] for key in ("action", "skill", "write_input", "base_hash", "snapshot_date")}
    proposal_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {"name": proposal["skill"], "description": inp.get("description", ""),
            "instructions": body.strip(), "candidate_id": proposal["id"], "proposal_hash": proposal_hash}


def candidate_snapshot(proposal: dict) -> dict:
    """
    返回经过验证的、已分离的暂存进化输入副本，本质上校验当前的提案的snapshot_date和snapshot是否符合要求，如果不符合要求就抛出异常
    Return a validated, detached copy of the staged evaluation input."""
    try:
        date.fromisoformat(proposal["snapshot_date"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("invalid candidate snapshot date; reload or regenerate the candidate") from None
    snapshot = proposal.get("snapshot")
    if not isinstance(snapshot, dict) or snapshot != _render_snapshot(proposal):
        raise ValueError("candidate parameters or snapshot changed; regenerate the candidate")
    return dict(snapshot)


def apply_candidate(candidate_id: str, *, manual_review: bool = False, expected_hash: str | None = None) -> dict:
    """
    应用当前的candidate，要么更新skill，要么创建新的skill
    Caller must authorize the concrete create/evolve operation first."""
    from .skills import get_skill_by_name, reset_skill_cache
    from .skill_evolution import create_skill_file, evolve_skill_file
    path = _path(candidate_id)
    with state_lock(path):
        candidate = read_candidate(candidate_id)
        if expected_hash is not None and hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest() != expected_hash:
            return {"ok": False, "error": "candidate changed during approval"}
        if candidate["status"] != "pending":
            return {"ok": False, "error": "candidate is no longer pending"}
        snapshot = candidate_snapshot(candidate)
        # 获取当前的champion信息，确保当前的champion和候选者的快照是匹配的
        registry_path = get_evolution_dir() / "online-eval" / "champions.json"
        with state_lock(registry_path) if not manual_review else nullcontext():
            if not manual_review:
                from .online_skill_eval import _lineage_id_for_skill
                registry = get_evolution_dir() / "online-eval" / "champions.json"
                # 从registry中读取champion信息获取registry[champions][skill_lineage_id]，如果没有就返回空字典
                champion = (json.loads(registry.read_text()).get("champions", {}).get(_lineage_id_for_skill(candidate["skill"]), {})
                            if registry.exists() else {})
                # 只有当前candidate的快照跟champion的快照一致，才可以进行apply操作，否则就报错
                # champion中的snapshot 用来明确记录：究竟哪个技能版本赢了。
                if champion.get("snapshot") != snapshot:
                    return {"ok": False, "error": "candidate has no matching passed evaluation; evaluate or explicitly use --manual-review"}
            # 返回这个candidate["skill"]这个名字下的SkillDefinition对象
            skill = get_skill_by_name(candidate["skill"])
            # 返回skill文件的Path对象
            base = Path(skill.skill_dir) / "SKILL.md" if skill else resolve_skill_file(candidate["skill"])
            # 使用skill的地址/或者skill进化的公开地址
            lock_path = base or (get_evolution_dir() / "candidate-publish")
            with state_lock(lock_path):
                # candidate["base_hash"]就是拿对应的json文件进行哈希计算的结果。
                # 如果对应base文件的哈希值和candidate["base_hash"]不一致，说明当前的技能已经被修改过了
                if _hash_file(base) != candidate["base_hash"]:
                    return {"ok": False, "error": "active Skill changed since proposal; regenerate/review the candidate"}
                # _prepared_body=snapshot["instructions"] + "\n"这里强制使用_prepared_body是为了避免重复新增note
                # 因为candidate = read_candidate(candidate_id)已经保证了这个candidate下面有了snapshot字段了
                write_input = dict(candidate["write_input"], _prepared_body=snapshot["instructions"] + "\n")
                if candidate["action"] == "merge":
                    # 通过candidate新的信息更新原本的skill
                    result = evolve_skill_file(**write_input, active_dir=skill.skill_dir if skill else "")
                else:
                    # 创建新的skill文件，利用candidate中的信息
                    result = create_skill_file(**write_input)
                if result.get("ok"):
                    reset_skill_cache()
                    # 更新candidate状态为applied
                    candidate["status"] = "applied"
                    write_json(path, candidate)
                return result


def rollback_skill(name: str) -> dict:
    """Restore the most recent pre-change snapshot after caller authorization."""
    from .skills import get_skill_by_name, reset_skill_cache
    skill = get_skill_by_name(name)
    target = resolve_skill_file(name, active_dir=skill.skill_dir if skill else "")
    if target is None:
        return {"ok": False, "error": "Skill not found"}
    history = get_evolution_dir() / "history" / f"{_safe_skill_slug(name)}.jsonl"
    if not history.is_file():
        return {"ok": False, "error": "No pre-change snapshot found"}
    # 同时锁上skill文件和history文件，防止其他进程修改
    with state_lock(target), state_lock(history):
        rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
        # 找到符合条件的最后一条记录，要求event=snapshot，file=target，content是字符串类型
        snapshot = next((row for row in reversed(rows) if row.get("event") == "snapshot"
                         and row.get("file") == str(target) and isinstance(row.get("content"), str)), None)
        if snapshot is None:
            return {"ok": False, "error": "No matching snapshot found"}
        # Save the current version too, making an explicit rollback reversible.
        # 在回滚前依旧要保存当前的版本，方便回滚后再回滚回来
        from .storage import append_jsonl
        append_jsonl(history, {"event": "snapshot", "actor": "rollback", "skill": name,
                              "file": str(target), "content": target.read_text()})
        # 将snapshot中的内容写入到target文件中
        atomic_write_text(target, snapshot["content"])
        reset_skill_cache()
        return {"ok": True, "skill": name, "file": str(target)}
