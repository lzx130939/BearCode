from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable


SideQuery = Callable[[str, str], Awaitable[str]]
ConfirmWrite = Callable[[str, dict[str, Any]], Awaitable[bool]]


@dataclass
class OnlineSkillCandidate:
    name: str
    description: str
    when_to_use: str = ""
    instructions: str = ""
    evidence: str = ""
    tags: list[str] = field(default_factory=list)


def _parse_json_object(text: str) -> dict[str, Any]:
    """解析json对象，找到最外侧的大括号{}，返回一个dict对象，如果没有找到大括号或者解析失败，抛出异常"""
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty model response")
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except ValueError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(raw[start : end + 1])
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
    raise ValueError("model response must contain a JSON object")


def _normalize_identity(text: str) -> str:
    """把text归一化"""
    raw = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", " ", str(text or "").lower())
    return re.sub(r"\s+", " ", raw).strip()


def _candidate_search_text(candidate: OnlineSkillCandidate) -> str:
    """把OnlineSkillCandidate对象的name、description、when_to_use、instructions和tags拼接成一个字符串"""
    return "\n".join(
        [
            candidate.name,
            candidate.description,
            candidate.when_to_use,
            candidate.instructions,
            " ".join(candidate.tags),
        ]
    )


def _coerce_candidate(obj: dict[str, Any]) -> OnlineSkillCandidate | None:
    """对skill candidate对象进行类型检查和字段提取，如果缺少必要字段，返回None, 返回一个OnlineSkillCandidate对象"""
    name = str(obj.get("name") or "").strip()
    description = str(obj.get("description") or "").strip()
    instructions = str(obj.get("instructions") or obj.get("prompt") or "").strip()
    if not name or not description or not instructions:
        return None
    tags_raw = obj.get("tags") or []
    if isinstance(tags_raw, str):
        tags = [part.strip() for part in re.split(r"[,，]", tags_raw) if part.strip()]
    elif isinstance(tags_raw, list):
        tags = [str(part).strip() for part in tags_raw if str(part).strip()]
    else:
        tags = []
    return OnlineSkillCandidate(
        name=name,
        description=description,
        when_to_use=str(obj.get("when_to_use") or obj.get("when-to-use") or "").strip(),
        instructions=instructions,
        evidence=str(obj.get("evidence") or "").strip(),
        tags=tags[:8],
    )


async def extract_online_skill_candidate(
    *,
    messages: list[dict[str, Any]],
    side_query: SideQuery,
    retrieved_reference: dict[str, Any] | None = None,
    hint: str = "",
) -> OnlineSkillCandidate | None:
    """从messages中(辅以retrieved_reference和hint)提取技能候选，返回一个OnlineSkillCandidate对象，如果没有提取到任何技能候选，返回None"""
    system = (
        "You are Bear Code's online Skill Extractor.\n"
        "Extract at most ONE reusable skill candidate from a live conversation window.\n"
        "Output ONLY strict JSON: {\"skills\": []} or {\"skills\": [{...}]}.\n\n"
        "Candidate fields: name, description, when_to_use, instructions, evidence, tags.\n\n"
        "Rules:\n"
        "- USER turns are the primary evidence. Assistant turns are context only.\n"
        "- A next user feedback turn may confirm, reject, or refine the prior assistant behavior.\n"
        "- Do not extract assistant-only guesses, weak confirmations, one-off task payload, secrets, project facts, URLs, account IDs, exact dates, or temporary parameters.\n"
        "- Extract only durable workflow, output policy, implementation preference, correction, or repeated constraint likely useful for future similar tasks.\n"
        "- Remove entity names and runtime-specific payload; use placeholders where needed.\n"
        "- retrieved_reference is identity context only; never treat it as new user evidence.\n"
        "- If evidence is weak, generic, or low-value, return {\"skills\": []}.\n"
    )
    payload = {
        "messages": messages,
        "hint": hint,
        "retrieved_reference": retrieved_reference or None,
    }
    parsed = _parse_json_object(await side_query(system, json.dumps(payload, ensure_ascii=False)))
    skills = parsed.get("skills")
    # 检验是否只有一个技能候选，如果有多个技能候选，抛出异常
    if not isinstance(skills, list) or len(skills) > 1:
        raise ValueError("extractor must return a skills array with at most one candidate")
    if not skills:
        return None
    first = skills[0]
    candidate = _coerce_candidate(first) if isinstance(first, dict) else None
    if candidate is None:
        raise ValueError("invalid extracted Skill candidate")
    return candidate


def _exact_identity_match(candidate: OnlineSkillCandidate, skills: list[Any]) -> str:
    """判断OnlineSkillCandidate和skills是否完全匹配"""
    candidate_ids = {
        _normalize_identity(candidate.name),
        _normalize_identity(candidate.description),
        _normalize_identity(candidate.when_to_use),
    }
    # 清除空字符串
    candidate_ids.discard("")
    for skill in skills:
        skill_ids = {
            _normalize_identity(getattr(skill, "name", "")),
            _normalize_identity(getattr(skill, "description", "")),
            _normalize_identity(getattr(skill, "when_to_use", "") or ""),
        }
        skill_ids.discard("")
        # 如果完全一样，返回skill的name
        if candidate_ids & skill_ids:
            return getattr(skill, "name", "")
    return ""


async def maintain_online_skill_candidate(
    *,
    candidate: OnlineSkillCandidate,
    side_query: SideQuery,
    retrieved_reference: dict[str, Any] | None = None,
    confirm_write: ConfirmWrite | None = None,
    target: str = "project",
    stage_only: bool = False,
) -> dict[str, Any]:
    """维护在线技能候选，判断是新增、合并还是丢弃，并执行相应的操作
    其判断核心就是给llm提供：
    - candidate: 当前提取到的技能候选
    - exact_identity_target: 如果candidate和现有技能完全匹配，返回匹配
    - retrieved_reference: 检索到的参考信息
    - similar_skills: 检索到的相似技能
    - existing_skills: 当前已有的技能列表
    然后llm会返回一个决策，包含一个特定的schema

    核心流程：
    输入候选技能
        ↓
    查找已有技能、精确匹配项和相似技能
        ↓
    让大模型返回处理决策
        ↓
    用代码调整、检查决策
        ├─ 丢弃 → 直接返回
        ├─ 新增 → 准备新增参数
        └─ 合并 → 检查目标和完整内容，准备更新参数
                       ↓
                    写入确认
                       ├─ 未通过 → 返回拒绝结果
                       └─ 通过
                            ├─ stage_only=True → 暂存
                            └─ stage_only=False → 执行新增或合并
    """
    from .skills import create_skill, discover_skills, evolve_skill, retrieve_relevant_skills

    skills = discover_skills()
    # 判断是否有完全匹配的技能，如果有，返回技能的name，否则返回空字符串
    exact_target = _exact_identity_match(candidate, skills)
    # 检索相似技能，返回一个列表，包含相似技能的name、description、when_to_use、source、context和instructions
    similar_hits = retrieve_relevant_skills(_candidate_search_text(candidate), limit=8, min_score=0.03)
    top_reference_name = str((retrieved_reference or {}).get("name") or "").strip()

    system = (
        "You are Bear Code's online Skill Set Manager.\n"
        "Decide whether a candidate should add a new skill, merge into an existing skill, or be discarded.\n"
        "Output ONLY strict JSON.\n\n"
        "Schema:\n"
        "{\"action\":\"add|merge|discard\",\"target_skill\":\"existing name for merge\","
        "\"reason\":\"short reason\",\"merged_description\":\"optional\","
        "\"merged_when_to_use\":\"optional\",\"merged_instructions\":\"optional full merged SKILL.md body\"}\n\n"
        "Rules:\n"
        "- Prefer merge over add when the same capability already exists.\n"
        "- Discard if the candidate duplicates an existing shared/project skill and adds no user-specific durable improvement.\n"
        "- If merging, synthesize a complete merged instruction body, preserving useful existing guidance and adding only durable new guidance.\n"
        "- Do not preserve one-off payload, secrets, transient project facts, URLs, exact dates, or assistant-only claims.\n"
    )
    payload = {
        "candidate": asdict(candidate),
        "exact_identity_target": exact_target,
        "retrieved_reference": retrieved_reference or None,
        "similar_skills": similar_hits,
        # 只拿取了前80个技能，避免传输过多数据
        "existing_skills": [
            {
                "name": getattr(skill, "name", ""),
                "description": getattr(skill, "description", ""),
                "when_to_use": getattr(skill, "when_to_use", "") or "",
                "source": getattr(skill, "source", ""),
                "context": getattr(skill, "context", ""),
                "instructions": (getattr(skill, "prompt_template", "") or "")[:6000],
            }
            for skill in skills[:80]
        ],
    }

    # llm返回的json结构体信息
    decision = _parse_json_object(await side_query(system, json.dumps(payload, ensure_ascii=False)))
    action = str(decision.get("action") or "").strip().lower()
    # target_skill表示的是要合并的目标技能的名称
    target_skill = str(decision.get("target_skill") or "").strip()

    # 如果有exact target(精确的目标)
    # 就代表llm认为应该是合并到这个技能上，而不是新增技能
    if exact_target:
        action = "merge"
        target_skill = exact_target
    #  如果是merge确没有给指定的target_skill，就用top_reference_name作为目标技能
    elif action == "merge" and not target_skill:
        target_skill = top_reference_name

    if action not in {"add", "merge", "discard"}:
        raise ValueError("invalid Skill maintenance action")

    # 丢弃直接返回消息即可
    if action == "discard":
        return {"ok": True, "action": "discard", "skill": "", "decision": decision}

    # 合并
    if action == "merge":
        target_skill = target_skill or top_reference_name
        if not target_skill:
            return {"ok": False, "action": "merge", "error": "missing target_skill", "decision": decision}
        # merged_instructions参数残缺
        if not isinstance(decision.get("merged_instructions"), str) or not decision["merged_instructions"].strip():
            return {"ok": False, "action": "merge", "error": "merge requires complete merged_instructions", "decision": decision}
        # 确定后续操作是技能演化
        tool_name = "skill_evolve"
        write_input = dict(
            skill_name=target_skill,
            lesson=candidate.evidence or candidate.description,
            rationale=str(decision.get("reason") or "Online maintainer merge"),
            target="active",
            instructions=str(decision.get("merged_instructions") or candidate.instructions),
            description=str(decision.get("merged_description") or ""),
            when_to_use=str(decision.get("merged_when_to_use") or candidate.when_to_use),
            tags=candidate.tags,
            preserve_existing=True,
        )
    else:
        tool_name = "skill_create"
        write_input = dict(
            name=candidate.name, description=candidate.description, instructions=candidate.instructions,
            when_to_use=candidate.when_to_use, target=target, context="inline", user_invocable=False,
            evidence=candidate.evidence, actor="online", tags=candidate.tags,
        )

    # 不管是新增还是合并，都需要鉴权当前tool_name + params(本质就是tool_name + write_input)，
    # 如果confirm_write为None或者返回False，就返回拒绝结果
    if confirm_write is None or not await confirm_write(tool_name, write_input):
        return {
            "ok": False,
            "action": f"{action}_denied",
            # target_skill or candidate.name分别对象evolve和create的技能名称
            "skill": target_skill or candidate.name,
            "error": "permission denied",
            "decision": decision,
        }

    # 如果只是暂存的话，那么就只是本质上把skill candidate持久化，后续是否真的新增或者合并技能，取决于后续的操作
    if stage_only:
        from .skill_candidates import stage_candidate
        return stage_candidate(action, write_input)

    if action == "merge":
        result = evolve_skill(**write_input)
        return {"action": "merge", "candidate": asdict(candidate), "decision": decision, **result}

    result = create_skill(**write_input)
    return {"action": "add", "candidate": asdict(candidate), "decision": decision, **result}


async def online_ingest(
    *,
    messages: list[dict[str, Any]],
    side_query: SideQuery,
    retrieved_reference: dict[str, Any] | None = None,
    hint: str = "",
    confirm_write: ConfirmWrite | None = None,
    target: str = "project",
    replay_context: dict[str, Any] | None = None,
    stage_only: bool = True,
) -> dict[str, Any]:
    """online_ingest 从当前对话中提取技能，并交给技能管理流程处理”的入口函数

    参数：
    - messages: 当前对话的消息列表
    - side_query: 用于记忆召回的 side_query 可调用对象
    - retrieved_reference: 检索到的参考信息
    - hint: 用户提供的提示信息
    - confirm_write: 确认写入的回调函数(就是一个鉴权的判断回调函数，返回True表示允许写入，False表示拒绝写入)
    - target: 目标技能存储位置，默认为"project"，也可以是"user"表示存储在用户技能中
    - replay_context: 回放上下文
    - stage_only: 是否仅进行阶段化处理"""
    from .skills import record_online_provenance

    try:
        # 根据messages、side_query、retrieved_reference和hint提取技能候选(本质先是进行无中生有创建skill candidate，然后再进行技能演变的判断)
        # 如果没有提取到任何技能候选，返回{"skills": []}，
        candidate = await extract_online_skill_candidate(
            messages=messages,
            side_query=side_query,
            retrieved_reference=retrieved_reference,
            hint=hint,
        )
    except Exception as exc:
        result = {"ok": False, "action": "failed", "error": str(exc)}
        # 记录在线技能演变的溯源信息，包括动作、回放上下文、结果、消息、检索到的参考信息和错误信息
        record_online_provenance(
            action="failed",
            replay_context=replay_context,
            result=result,
            messages=messages,
            retrieved_reference=retrieved_reference,
            error=str(exc),
        )
        return result

    # 此时是因为当前的对话中没有提取到任何技能候选，返回的是{"skills": []}，所以candidate是None
    if candidate is None:
        result = {"ok": True, "action": "none"}
        record_online_provenance(
            action="none",
            replay_context=replay_context,
            result=result,
            messages=messages,
            retrieved_reference=retrieved_reference,
        )
        return result

    # 如果candidate非空那么就需要判断到底是否需要新增、合并还是丢弃，并执行相应的操作
    try:
        result = await maintain_online_skill_candidate(
            candidate=candidate,
            side_query=side_query,
            retrieved_reference=retrieved_reference,
            confirm_write=confirm_write,
            target=target,
            stage_only=stage_only,
        )
    except Exception as exc:
        result = {"ok": False, "action": "failed", "skill": candidate.name, "error": str(exc)}

    record_online_provenance(
        action=str(result.get("action") or "none"),
        replay_context=replay_context,
        skill_name=str(result.get("skill") or candidate.name),
        result=result,
        messages=messages,
        retrieved_reference=retrieved_reference,
        decision=result.get("decision") if isinstance(result.get("decision"), dict) else None,
        error="" if result.get("ok") else str(result.get("error") or ""),
    )
    return result


async def judge_retrieved_skill_usage(
    *,
    hits: list[dict[str, Any]],
    user_message: str,
    assistant_text: str,
    side_query: SideQuery | None = None,
) -> list[dict[str, Any]]:
    """判断检索到的技能是否真正的被使用，返回一个列表，包含每个技能的使用情况
    PS：这里的"真正被使用"具体指的是只有当回复遵循该技能特有的工作流或策略时，才视为该技能已被使用，而不仅仅是因为该技能被检索到了

    params:
        hits: 检索到的技能列表
        user_message: 用户发送的问题消息
        assistant_text: Agent后续返回的回复消息
        side_query: 用于判断的side query

    返回结构：
    [
        {
            {
                "name": name,
                "source": hit.get("source", ""),
                "skill_dir": hit.get("skill_dir", ""),
                "retrieved": True,
                "relevant": bool(raw.get("relevant")),
                "used": bool(raw.get("used")),
                "score": float(hit.get("score", 0.0)),
                "reason": str(raw.get("reason") or ""),
            }
        }
    ]
    """
    if not hits:
        return []
    if side_query is None:
        # No judge means no evidence. Mentioning a Skill name does not prove usage.
        return []

    system = (
        "Judge whether retrieved skills were relevant to the user request and actually used in the assistant reply.\n"
        "Output ONLY strict JSON: {\"judgments\":[{\"name\":\"...\",\"relevant\":true|false,\"used\":true|false,\"reason\":\"short\"}]}.\n"
        "A skill is used only if the reply follows its distinctive workflow or policy, not merely because it was retrieved."
    )
    payload = {"user_message": user_message, "assistant_reply": assistant_text, "retrieved_skills": hits}
    parsed = _parse_json_object(await side_query(system, json.dumps(payload, ensure_ascii=False)))
    raw_judgments = parsed.get("judgments") if isinstance(parsed, dict) else None
    names = {str(hit.get("name") or "") for hit in hits}
    if (not isinstance(raw_judgments, list) or len(raw_judgments) != len(names)
            or any(not isinstance(item, dict) or item.get("name") not in names
                   or type(item.get("relevant")) is not bool or type(item.get("used")) is not bool
                   or (item["used"] and not item["relevant"]) for item in raw_judgments)
            or len({item["name"] for item in raw_judgments}) != len(names)):
        raise ValueError("Invalid Skill usage judgments; statistics were not updated")
    by_name = {item["name"]: item for item in raw_judgments}
    judgments: list[dict[str, Any]] = []
    for hit in hits:
        name = str(hit.get("name") or "")
        raw = by_name.get(name, {})
        judgments.append(
            {
                "name": name,
                "source": hit.get("source", ""),
                "skill_dir": hit.get("skill_dir", ""),
                "retrieved": True,
                "relevant": bool(raw.get("relevant")),
                "used": bool(raw.get("used")),
                "score": float(hit.get("score", 0.0)),
                "reason": str(raw.get("reason") or ""),
            }
        )
    return judgments
