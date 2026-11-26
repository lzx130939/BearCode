"""Shared YAML frontmatter parser for memory and skills files.
Handles simple `key: value` pairs between `---` delimiters."""

from dataclasses import dataclass, field


@dataclass
class FrontmatterResult:
    meta: dict[str, str] = field(default_factory=dict)
    body: str = ""


def parse_frontmatter(content: str) -> FrontmatterResult:
    """解析从记忆文件见读取出来的str，返回一个FrontmatterResult对象，包含meta和body两个属性

    预期的str格式如下(memory/skill/Agent都会共用这种格式)：
    ```markdown
    ---
    name: memory name
    description: one-line description
    type: user|feedback|project|reference
    XXX：xx
    ---
    Memory content here.<body>
    ```
    """
    lines = content.split("\n")
    # 表示没有了前面的header信息，直接返回body
    if not lines or lines[0].strip() != "---":
        return FrontmatterResult(body=content)

    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx == -1:
        return FrontmatterResult(body=content)

    # 获取meta信息
    meta: dict[str, str] = {}
    for i in range(1, end_idx):
        colon_idx = lines[i].find(":")
        if colon_idx == -1:
            continue
        key = lines[i][:colon_idx].strip()
        value = lines[i][colon_idx + 1:].strip()
        if key:
            meta[key] = value

    # 获取body信息
    body = "\n".join(lines[end_idx + 1:]).strip()
    return FrontmatterResult(meta=meta, body=body)


def format_frontmatter(meta: dict[str, str], body: str) -> str:
    lines = ["---"]
    for key, value in meta.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    return "\n".join(lines)
