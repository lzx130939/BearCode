"""
该文件的核心作用是进行messages的压缩工作
In-place tool-result compression without model, filesystem or UI dependencies."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

SNIP_THRESHOLD = 0.60
# 表示当前content已经被压缩过了，后续不再重复压缩
SNIP_PLACEHOLDER = "[Content snipped - re-read if needed]"
SNIPPABLE_TOOLS = {"read_file", "grep_search", "list_files", "run_shell"}
MICROCOMPACT_IDLE_S = 5 * 60
# 表示最多保留下来的结果
KEEP_RECENT_RESULTS = 3
OLD_RESULT_PLACEHOLDER = "[Old result cleared]"


def _text_results(messages: list[dict[str, Any]], *, use_openai: bool) -> Iterator[dict[str, Any]]:
    """
    返回一个迭代器，同时保持文本和非文本代码块不变。
    本质就是筛选出 messages 中的工具调用返回的包含 content 的原始字典结果，返回一个迭代器。

    Yield mutable result containers, leaving text and non-text blocks untouched."""
    for message in messages:
        if use_openai:
            if message.get("role") == "tool" and isinstance(message.get("content"), str):
                yield message
        elif message.get("role") == "user" and isinstance(message.get("content"), list):
            for block in message["content"]:
                if (isinstance(block, dict) and block.get("type") == "tool_result"
                        and isinstance(block.get("content"), str)):
                    yield block


def _anthropic_tool_index(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    找到anthropic消息中所有的工具调用，并返回一个字典，键是工具调用的id，值是工具调用的原始字典。
    只需索引一次，而无需为每个结果扫描整个对话
    Index once instead of scanning the full conversation for every result."""
    tools = {}
    for message in messages:
        if message.get("role") != "assistant" or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                tools.setdefault(block["id"], block)
    return tools


def compact_tool_results(messages: list[dict[str, Any]], *, use_openai: bool,
                         utilization: float, idle_seconds: float = 0) -> None:
    """
    筛选出重要结果，删除过时结果，然后清除闲置历史记录。
    保留既定的协议政策：
    1、OpenAI 会删除任何旧的文本工具结果；
    2、Anthropic 仅删除已知的可重复读取的工具结果和已被取代的文件读取结果。
    消息角色、调用标识符、非工具文本以及非文本内容均保持完整

    主要逻辑：
    取得所有纯文本工具结果
              ↓
    第一层：过长的结果保留头尾，截掉中间
              ↓
    第二层：上下文紧张时，把较旧结果替换成占位符
              ↓
    第三层：间隔较久时，把较旧的剩余结果替换成占位符

    Budget large results, snip stale results, then clear idle history.

    Preserve the established protocol policies: OpenAI snips any old text tool
    result; Anthropic snips only known rereadable tools and superseded file reads.
    Message roles, call identifiers, non-tool text and non-text content are intact.
    """
    # 返回messages中调用tool的content信息的原始dict
    results = list(_text_results(messages, use_openai=use_openai))
    if utilization >= 0.5:
        # 根据当前上下文窗口利用率，采用不同的预算策略，来决定保留多少内容
        budget = 15000 if utilization > 0.7 else 30000
        keep = (budget - 80) // 2
        for result in results:
            content = result["content"]
            if len(content) > budget:
                result["content"] = (content[:keep]
                    + f"\n\n[... budgeted: {len(content) - keep * 2} chars truncated ...]\n\n"
                    + content[-keep:])

    # 如果利用率超过了阈值，则对结果进行筛选，删除过时的结果
    if utilization >= SNIP_THRESHOLD:
        # 筛选出所有当前还没有被压缩过的调用工具调用结果content
        candidates = [result for result in results if result["content"] != SNIP_PLACEHOLDER]
        # OpenAI：不需要按工具名筛选，因此这里使用空字典。
        # Anthropic：构建工具调用索引，以便从结果的 tool_use_id 查到原始工具调用。
        tools = {} if use_openai else _anthropic_tool_index(messages)
        if not use_openai:
            # 如果是Anthropic协议，则只保留candidate中调用的tool是在SNIPPABLE_TOOLS中的工具调用结果
            # TODO：这里的区别对待我没理解到底为什么，跟gpt聊过了。给出的结论式作者认为设计的(我觉得没啥意义)
            #      另外只筛选SNIPPABLE_TOOLS中的方法，大概率式初代版本中对于文本读取等相关的工具调用结果，认为占有数据量大，压缩效益比较可观
            candidates = [result for result in candidates
                          if tools.get(result.get("tool_use_id"), {}).get("name") in SNIPPABLE_TOOLS]
        if len(candidates) > KEEP_RECENT_RESULTS:
            # 生成一个stale集合，即[0:len(candidates) - KEEP_RECENT_RESULTS)]
            # 这里认为在前面的就是messages更早之间的调用，所以需要被压缩掉
            stale = set(range(len(candidates) - KEEP_RECENT_RESULTS))
            if not use_openai:
                # 在Anthropic协议中，如果同一路径被 read_file 读取过多次，把前一次读取结果也标记为过时，因为后一次读取结果可以取代它。
                latest_reads = {}
                for index, result in enumerate(candidates):
                    tool = tools[result["tool_use_id"]]
                    arguments = tool.get("input") or {}
                    path = arguments.get("file_path") if isinstance(arguments, dict) else None
                    if tool.get("name") == "read_file" and isinstance(path, str) and path:
                        # 如果新调用中包含了之前的路径，则把之前的路径对应的调用index在stale中添加
                        if path in latest_reads:
                            stale.add(latest_reads[path])
                        # 同时更新path对应的索引
                        latest_reads[path] = index
            #  遍历stale，所有老的read_file操作content都被替换成SNIP_PLACEHOLDER(因为后面已经有新的读取内容了)，用以节约上下文空间
            for index in stale:
                candidates[index]["content"] = SNIP_PLACEHOLDER

    # 判断当前距离上次调用大模型的时间idle_seconds是否超过了MICROCOMPACT_IDLE_S(这里是5*60，5分钟)，
    # 如果超过了，则把剩余的结果中除了最近KEEP_RECENT_RESULTS条记录之外的其他记录的content都替换成OLD_RESULT_PLACEHOLDER
    # 也就是说超过5分钟没有调用大模型，就会把剩余的结果中除了最近KEEP_RECENT_RESULTS条记录之外的其他记录的content都替换成OLD_RESULT_PLACEHOLDER
    if idle_seconds >= MICROCOMPACT_IDLE_S:
        remaining = [result for result in results
                     if result["content"] not in {SNIP_PLACEHOLDER, OLD_RESULT_PLACEHOLDER}]
        for result in remaining[:-KEEP_RECENT_RESULTS]:
            result["content"] = OLD_RESULT_PLACEHOLDER
