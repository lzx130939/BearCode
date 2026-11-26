from __future__ import annotations

import json
import math
import re
from base64 import encode
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .frontmatter import parse_frontmatter
from .skill_evolution import (
    create_skill_file,
    evolve_skill_file,
    format_skill_stats,
    record_online_skill_provenance,
    record_skill_feedback,
    record_skill_invocation,
    record_skill_usage_judgments,
)


@dataclass
class SkillDefinition:
    # 一个 skill 在程序内的统一表示，由 SKILL.md 的 frontmatter 和正文解析得到。
    name: str
    description: str
    when_to_use: str | None = None
    allowed_tools: list[str] | None = None
    user_invocable: bool = True
    context: str = "inline"  # "inline" or "fork"，context 默认 inline(把内容直接嵌在当前的上下文/输出里，fork表示Skill 会在一个全新的子代理（subagent）中运行)。
    prompt_template: str = "" # prompt_template 是 SKILL.md 的正文部分，包含了 skill 的使用说明和示例。
    source: str = "project"  # "project" or "user"
    skill_dir: str = ""


# skills 只在首次读取时扫描磁盘，后续复用缓存；修改 skill 后需要重启或 reset。
_cached_skills: list[SkillDefinition] | None = None
_cached_skill_signature = None


def execute_skill(skill_name:str, args:object)-> dict | None:
    """保存技能调用记录，并返回解析后的 prompt 和执行配置。"""
    # skill 工具的执行入口：按名字找到 skill，并返回解析后的 prompt 和执行配置。
    skill = get_skill_by_name(skill_name)
    if not skill:
        return None

    # 保存技能调用记录，持久化到get_evolution_dir() / USAGE_LOG中
    record_skill_invocation(
        skill_name=skill.name,
        source=skill.source,
        context=skill.context,
        args=args,
    )

    return {
        "prompt": resolve_skill_prompt(skill, args),
        "allowed_tools": skill.allowed_tools,
        "context": skill.context,
        "source": skill.source,
        "skill_dir": skill.skill_dir,
    }



def resolve_skill_prompt(skill: SkillDefinition, args: object) -> str:
    """把 skill 模板正文中的占位符（placeholder）替换成真实值，主要包括Argument+Dict，生成最终可用的 prompt 字符串"""
    import re
    # 获取skill的正文prompt
    prompt = skill.prompt_template
    # 支持在 SKILL.md 正文中使用 $ARGUMENTS 或 ${ARGUMENTS} 引用用户参数。
    # $ARGUMENTS / ${ARGUMENTS} → 替换成用户传入的参数
    prompt = re.sub(r"\$ARGUMENTS|\$\{ARGUMENTS\}", lambda _: str(args or ""), prompt)
    # 支持 skill 引用自己的目录，例如读取同目录下的 references/scripts。
    prompt = prompt.replace("${CLAUDE_SKILL_DIR}", skill.skill_dir)
    return prompt

def get_skill_by_name(skill_name:str)->SkillDefinition | None:
    """# 通过 name 查找 skill；name 来自 frontmatter，没有写时使用目录名。"""
    for s in discover_skills():
        if s.name == skill_name:
            return s
    return None

def discover_skills() -> list[SkillDefinition]:
    """从home()/.bear/skill 和 cwd()/.bear/skill 目录下扫描所有 SKILL.md 文件，解析成 SkillDefinition 列表返回。"""
    global _cached_skills, _cached_skill_signature
    roots = [Path.home() / ".bear" / "skills", Path.cwd() / ".bear" / "skills"]
    signature = []
    for root in roots:
        for path in sorted(root.glob("*/SKILL.md")):
            try:
                info = path.stat()
                # signature 是一个元组列表，每个元素包含 skill 文件的绝对路径、修改时间、大小和 inode
                signature.append((str(path.resolve()), info.st_mtime_ns, info.st_size, info.st_ino))
            except OSError:
                continue
    # 最终的signature是[cwd().resolve(), (skill_path, mtime_ns, size, inode), ...]，用于判断是否需要重新扫描
    signature = (str(Path.cwd().resolve()), tuple(signature))
    if _cached_skills is not None and signature == _cached_skill_signature:
        return _cached_skills

    skills: dict[str,SkillDefinition] = {}
    # 用户级 skills 优先级最高：~/.bear/skills/<name>/SKILL.md
    user_dir = Path.home() / ".bear" / "skills"
    _load_skills_from_dir(user_dir, "user", skills)
    # 项目级 skills 优先级较低：<cwd>/.bear/skills/<name>/SKILL.md
    project_dir = Path.cwd() / ".bear" / "skills"
    _load_skills_from_dir(project_dir, "project", skills, overwrite=False)

    _cached_skills = list(skills.values())
    _cached_skill_signature = signature
    return _cached_skills

def _load_skills_from_dir( base_dir: Path, source: str, skills:dict[str, SkillDefinition], overwrite: bool = True) -> None:
    """只加载目录形式的 skill，不加载 .bear/skills/foo.md 这种单文件形式。

    这里的overwrite表示的是解析Path的skill覆盖掉skills里面已经存在的重名skill项"""
    if not base_dir.is_dir():
        return
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        # 文件名必须是 SKILL.md，大小写要一致。
        skill_file = entry/  "SKILL.md"
        if not skill_file.exists():
            continue
        skill = _parse_skill_file(skill_file, source, str(entry))
        if skill:
            # 项目级加载时 overwrite=False，避免覆盖同名用户级 skill。
            if not overwrite and skill.name in skills:
                continue
            skills[skill.name] = skill

def _parse_skill_file(file_path: Path, source: str, skill_dir: str) -> SkillDefinition:
    """解析skill文件"""
    try:
        # SKILL.md = frontmatter 配置 + markdown 正文。
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        result = parse_frontmatter(raw)
        meta = result.meta

        # name 没写时用目录名；user-invocable(用户可调用) 默认 true；context 默认 inline(把内容直接嵌在当前的上下文/输出里，fork表示Skill 会在一个全新的子代理（subagent）中运行)。
        name = meta.get("name") or file_path.parent.name or "unknown"
        user_invocable = meta.get("user-invocable", "true") != "false"
        context = "fork" if meta.get("context") == "fork" else "inline"

        allowed_tools: list[str] | None = None
        if "allowed-tools" in meta:
            raw_tools = meta["allowed-tools"]
            # allowed-tools 支持 JSON 数组字符串，也支持逗号分隔。
            if raw_tools.startswith("["):
                try:
                    allowed_tools = json.loads(raw_tools)
                except Exception:
                    allowed_tools = [s.strip() for s in raw_tools.strip("[]").split(",")]
            else:
                allowed_tools = [s.strip() for s in raw_tools.split(",")]

        return SkillDefinition(
            name=name,
            description=meta.get("description", ""),
            when_to_use=meta.get("when_to_use") or meta.get("when-to-use"),
            allowed_tools=allowed_tools,
            user_invocable=user_invocable,
            context=context,
            prompt_template=result.body,
            source=source,
            skill_dir=skill_dir,
        )

    except Exception:
        return None


def build_skill_descriptions() -> str:
    """
    加载所有的 skill 信息，生成一个 markdown 格式的字符串，包含 skill 的名称、描述、使用场景等信息，供模型在系统提示词中使用。
    """

    # 把已加载的 skills 写进 system prompt，让模型知道哪些 skill 可用。
    skills = discover_skills()

    lines = ["# Available Skills", ""]
    if not skills:
        lines.append("(No skills are currently registered.)")
        lines.append("")
    # user_invocable=True 的 skill 主要给用户通过 /<name> 手动调用。
    invocable = [s for s in skills if s.user_invocable]
    # user_invocable=False 的 skill 作为自动调用候选，模型根据 when_to_use 决定是否调用 skill 工具。
    auto_only = [s for s in skills if not s.user_invocable]

    if invocable:
        lines.append("User-invocable skills (user types /<name> to invoke):")
        for s in invocable:
            lines.append(f"- **/{s.name}**: {s.description}")
            if s.when_to_use:
                lines.append(f"  When to use: {s.when_to_use}")
        lines.append("")

    if auto_only:
        lines.append("Auto-invocable skills:")
        lines.append("When the user's request matches a skill's When to use, call the `skill` tool with that skill name before continuing. Do not ask the user to invoke it manually.")
        for s in auto_only:
            lines.append(f"- **{s.name}**: {s.description}")
            if s.when_to_use:
                lines.append(f"  When to use: {s.when_to_use}")
        lines.append("")

    lines.append("To invoke a skill programmatically, use the `skill` tool with the skill name and optional arguments.")
    lines.append("")
    lines.append("# Skill Evolution")
    lines.append("Bear Code has an online skill evolution loop after each assistant response. Do not create or evolve skills during normal task execution unless the user explicitly asks for manual skill maintenance.")
    lines.append("If manual maintenance is explicitly requested, call `skill_evolve` only for durable reusable feedback on an existing skill, and call `skill_create` only when no suitable existing skill exists.")
    lines.append("Never create or evolve skills from one-off task content, private secrets, temporary project facts, or assistant-only guesses.")
    return "\n".join(lines)

# 连续的英文/数字算一个 token，而连续的中文按 1~2 个字为一组切分。
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{1,2}")
_STOP_TOKENS = {
    "请帮",
    "帮我",
    "我做",
    "做一",
    "一次",
    "一下",
    "这个",
    "那个",
    "一个",
    "用户",
    "问题",
    "回答",
    "生成",
    "使用",
    "需要",
}


def _tokens(text: str) -> set[str]:
    raw = str(text or "").lower().replace("_", " ").replace("-", " ")
    found = {m.group(0) for m in _TOKEN_RE.finditer(raw)}
    expanded = set(found)
    for token in found:
        if len(token) > 3 and token.endswith("s"):
            expanded.add(token[:-1])
    found = expanded
    cjk = re.findall(r"[\u4e00-\u9fff]+", raw)
    for chunk in cjk:
        if len(chunk) >= 2:
            found.update(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return {x for x in found if x.strip() and x not in _STOP_TOKENS}


def _token_list(text: str) -> list[str]:
    """将text进行分词操作"""
    raw = str(text or "").lower().replace("_", " ").replace("-", " ")
    tokens = [m.group(0) for m in _TOKEN_RE.finditer(raw)]
    for chunk in re.findall(r"[\u4e00-\u9fff]+", raw):
        if len(chunk) >= 2:
            tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    expanded: list[str] = []
    for token in tokens:
        if not token.strip() or token in _STOP_TOKENS:
            continue
        expanded.append(token)
        # 清除掉类似于"users"->"user"这种复数形式的token
        if len(token) > 3 and token.endswith("s"):
            expanded.append(token[:-1])
    return expanded


def _skill_search_text(skill: SkillDefinition) -> str:
    return "\n".join(
        [
            skill.name,
            skill.description,
            skill.when_to_use or "",
            skill.prompt_template[:4000],
        ]
    )


def retrieve_relevant_skills(
    query: str,
    *,
    limit: int = 3,
    min_score: float = 0.08,
) -> list[dict[str, Any]]:
    """根据查询条件检索相似的技能skill。
    基于 BM25 算法的技能（skill）检索函数：输入一段查询文本 query，从所有已发现的 skill 中找出最相关的若干个，按相关度排序返回

    TF(词频): 在文档中出现的次数越多，说明这个词对文档的描述能力越强，TF 越大。
    IDF(逆文档频率): 在所有文档中出现的次数越少，说明这个词对区分文档的能力越强，IDF 越大。
    """
    query_terms = _token_list(query)
    # 将query可能存在的重复token进行清除
    query_tokens = set(query_terms)
    if not query_tokens:
        return []

    docs: list[tuple[SkillDefinition, list[str]]] = []
    document_frequency: Counter[str] = Counter()
    for skill in discover_skills():
        meta_terms = _token_list("\n".join([skill.name, skill.description, skill.when_to_use or ""]))
        body_terms = _token_list(skill.prompt_template[:2500])
        # 加权拼接：meta_terms * 3 表示元信息 token 重复 3 遍，实现"名称/描述/使用场景比正文更重要"的字段加权
        terms = (meta_terms * 3) + body_terms
        if not terms:
            continue
        docs.append((skill, terms))
        # 统计文档频率 DF——每个 token 出现在多少个文档里（用 set 保证同一文档内重复 token 只计一次）
        document_frequency.update(set(terms))
    if not docs:
        return []

    # 平均每个doc的长度
    avg_doc_len = sum(len(terms) for _, terms in docs) / max(1, len(docs))
    doc_count = len(docs)
    k1 = 1.4  # 词频饱和参数——控制 TF 增长曲线，越大 TF 影响越持久
    b = 0.75  # 长度归一化强度——0 不归一，1 完全归一，0.75 是经典默认值
    hits: list[dict[str, Any]] = []
    for skill, terms in docs:
        # term_counts：terms中每个token的出现次数，Counter是一个字典，key是token，value是出现次数
        term_counts = Counter(terms)
        # overlap：query token 与文档 token 的交集——只有查询词在文档中出现过才参与打分
        overlap = query_tokens & set(term_counts)
        if not overlap:
            continue
        raw_score = 0.0
        doc_len = max(1, len(terms))
        for token in overlap:
            # ① TF（词频）：tf = term_counts[token]
            # ② IDF（逆文档频率）：
            # idf = log(1 + (N - df + 0.5) / (df + 0.5))
            # N = doc_count，df = document_frequency[token]
            # 出现文档越少的词，IDF 越大 → 越"稀有/有区分度"的词权重越高
            # 这是 BM25 常用的 Robertson-Sparck Jones IDF 变体（加 1 防止负值）
            tf = term_counts[token]
            idf = math.log(1 + (doc_count - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5))
            # ③ 长度归一化分母：
            # denom = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            # 文档越长，denom 越大 → 同样 tf 下得分被压低（惩罚长文档）
            # 这就是 b 的作用
            denom = tf + k1 * (1 - b + b * doc_len / max(1.0, avg_doc_len))
            # ④ 累加 BM25 项：
            # raw_score += idf * (tf * (k1 + 1)) / max(denom, 0.0001)
            # 标准 BM25 公式：IDF × (tf·(k1+1)) / (tf + k1·(1−b+b·|d|/avgdl))
            # max(denom, 0.0001) 防除零
            raw_score += idf * (tf * (k1 + 1)) / max(denom, 0.0001)
        # 名称加分：如果 skill 的 name（小写）作为子串出现在 query（小写）里 → 加 0.15。这是精确名称命中的强信号，BM25 之外的手工加成
        name_bonus = 0.15 if skill.name.lower() in str(query or "").lower() else 0.0
        score = min(1.0, (raw_score / max(3.0, len(query_tokens))) + name_bonus)
        if score < float(min_score):
            continue
        hits.append(
            {
                "score": float(score),
                "name": skill.name,
                "description": skill.description,
                "when_to_use": skill.when_to_use or "",
                "source": skill.source,
                "context": skill.context,
                "user_invocable": bool(skill.user_invocable),
                "skill_dir": skill.skill_dir,
            }
        )

    hits.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return hits[: max(1, int(limit or 1))]


def format_retrieved_skill_context(query: str, *, limit: int = 3) -> tuple[str, dict[str, Any] | None]:
    """格式化被检索到的技能信息，附加到上下文中(从这个函数可以知道为什么需要有_run_skill_usage_tracking这个方法了，因为这里被检索到的单纯是因为分数高并不一定是相关的)"""
    # 基于BM25 算法的技能（skill）检索函数检索到的高相似度skill列表，按相关度排序返回前3名
    hits = retrieve_relevant_skills(query, limit=limit)
    if not hits:
        return "", None
    lines = [
        "<retrieved_skills>",
        "These skills were retrieved for the current user request. Use a skill only if it directly matches the user's intent; otherwise ignore this block.",
    ]
    for idx, hit in enumerate(hits, start=1):
        lines.append(
            f"{idx}. {hit['name']} (score={float(hit['score']):.3f}, source={hit['source']}): {hit['description']}"
        )
        if hit.get("when_to_use"):
            lines.append(f"   When to use: {hit['when_to_use']}")
    lines.append("</retrieved_skills>")
    # 第一个元素是最高的相关度的skill信息，作为top返回
    top = dict(hits[0])
    # 同时绑定上一个all_hits属性，包含所有被检索到的skill信息
    top["all_hits"] = hits
    return "\n".join(lines), top


def reset_skill_cache() -> None:
    """通过清除skill_cache,实现重新刷新读取所有的skill信息
    本质是促使重新调用discover_skills()中的_load_skills_from_dir()
    """
    # 测试或运行中刷新 skills 时使用；普通用户通常重启程序即可。
    global _cached_skills
    _cached_skills = None


def evolve_skill(
    skill_name: str,
    lesson: str,
    rationale: str = "",
    target: str = "active",
    instructions: str = "",
    description: str = "",
    when_to_use: str = "",
    tags: list[str] | None = None,
    preserve_existing: bool = False,
) -> dict:
    """evolve_skill_file的封装函数，主要是为了在技能演化后刷新缓存"""
    skill = get_skill_by_name(skill_name)
    result = evolve_skill_file(
        skill_name=skill_name,
        lesson=lesson,
        rationale=rationale,
        target=target,
        active_dir=skill.skill_dir if skill else "",
        instructions=instructions,
        description=description,
        when_to_use=when_to_use,
        tags=tags,
        preserve_existing=preserve_existing,
    )
    if result.get("ok"):
        reset_skill_cache()
    return result


def create_skill(
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
) -> dict:
    result = create_skill_file(
        name=name,
        description=description,
        instructions=instructions,
        when_to_use=when_to_use,
        target=target,
        context=context,
        user_invocable=user_invocable,
        allowed_tools=allowed_tools,
        evidence=evidence,
        actor=actor,
        tags=tags,
    )
    if result.get("ok"):
        reset_skill_cache()
    return result


def record_online_provenance(
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
    record_online_skill_provenance(
        replay_context=replay_context,
        action=action,
        skill_name=skill_name,
        result=result,
        messages=messages,
        retrieved_reference=retrieved_reference,
        decision=decision,
        error=error,
    )


def record_feedback(skill_name: str, rating: str, note: str = "") -> None:
    record_skill_feedback(skill_name=skill_name, rating=rating, note=note)


def skill_stats() -> str:
    """获取所有skill的统计数据信息"""
    return format_skill_stats()


def record_usage_judgments(judgments: list[dict[str, Any]], *, allow_prune: bool = False) -> dict[str, Any]:
    """记录技能的使用判断结果，并根据需要刷新缓存"""
    result = record_skill_usage_judgments(judgments, allow_prune=allow_prune)
    # 判断是否pruned，是否被修剪。
    if result.get("pruned"):
        reset_skill_cache()
    return result
