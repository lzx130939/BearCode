"""Preview-first cleanup of disposable artifacts, preserving current champions."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import shutil
import time

from .skill_evolution import get_evolution_dir


def _referenced_results(directory: Path) -> set[str] | None:
    """
    _referenced_results是从.bear/sessions目录下的所有会话中的所有json文件中提取出所有被引用的工具结果文件名,格式<roots>/<filename>.txt，取<filename>.txt，并返回一个集合
    Scan all projects' sessions; an unreadable session makes deletion unsafe."""
    from .session import SESSION_DIR
    roots = {str(directory), str(directory.resolve())}
    # 先设置正则表达式的规则
    pattern = re.compile("(?:" + "|".join(re.escape(root) for root in roots) + r")[/\\]([A-Za-z0-9_.-]+\.txt)")
    referenced = set()
    try:
        for path in SESSION_DIR.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"Invalid session: {path}")
            pending = [data]
            while pending:
                item = pending.pop()
                if isinstance(item, dict):
                    pending.extend(item.keys())
                    pending.extend(item.values())
                elif isinstance(item, list):
                    pending.extend(item)
                elif isinstance(item, str):
                    # 在这里在item中选取通过正则表达式规律后的所有结果
                    referenced.update(pattern.findall(item))
    except (OSError, ValueError):
        logging.warning("Skipping tool-result cleanup: session references could not be verified", exc_info=True)
        return None
    return referenced


def cleanup_candidates(days: int = 30) -> list[Path]:
    """
    cleanup_candidates 逻辑
    1、先通过_referenced_results返回.bear/tool-results目录下所有被引用的工具结果文件名到referenced,格式<filename>.txt
    2、然后遍历所有目录 ~/.bear-code/tool-results/*.txt，查询不在引用信息（referenced）内+ 不是 symlink + 小于cutoff的Path，并append到paths中
    3、然后遍历所有目录 ~/.bear-code/tool-results/online-eval/runs/*/*中，查询不在冠军注册表下的run_id（champions.json → protected）+ 是目录结构 + 不是 symlink + 路径包含关系 + 小于cutoff的Path，并append到paths中
    """
    if days < 1:
        raise ValueError("retention days must be positive")
    cutoff = time.time() - days * 86400
    paths = []
    results = Path.home() / ".bear-code" / "tool-results"
    referenced = _referenced_results(results)
    for path in results.glob("*.txt"):
        if (referenced is not None and path.name not in referenced
                and not path.is_symlink() and path.stat().st_mtime < cutoff):
            paths.append(path)
    evaluation = get_evolution_dir() / "online-eval"
    registry = evaluation / "champions.json"
    champions = json.loads(registry.read_text(encoding="utf-8")).get("champions", {}) if registry.exists() else {}
    protected = {record.get("run_id") for record in champions.values()}
    for path in (evaluation / "runs").glob("*/*"):
        if (path.is_dir() and not path.is_symlink() and evaluation.resolve() in path.resolve().parents
                and path.name not in protected and path.stat().st_mtime < cutoff):
            paths.append(path)
    return paths


def cleanup_state(days: int = 30, *, apply: bool = False, expected_paths: list[str] | None = None) -> dict:
    """
    cleanup_state的本质是找到满足delay时间范围外和其他额外条件的~/.bear-code/tool-results/*.txt（一次工具调用的输出缓存文件）
    和~/.bear-code/tool-results/online-eval/runs/*/*目录下的文件（一次评测 run 的产物目录）的文件
    然后再跟expected_paths做对比，如果不一致就报错，最后根据apply参数决定是否真的删除这些文件"""
    paths = cleanup_candidates(days)
    if apply:
        if expected_paths is not None and set(map(str, paths)) != set(expected_paths):
            raise ValueError("cleanup candidates changed during approval; preview again")
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
    return {"applied": apply, "count": len(paths), "paths": [str(path) for path in paths]}
