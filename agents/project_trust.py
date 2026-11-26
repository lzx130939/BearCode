"""User-owned trust for an exact project configuration snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .storage import state_lock, write_json


def trust_file() -> Path:
    return Path.home() / ".bear" / "trusted-projects.json"


def project_fingerprint(project: Path | None = None) -> str:
    """project_fingerprint函数就是根据给定地址/cwd地址的地址Path 和
    .bear/settings.json的绝对地址 + 其中的内容 和
    .mcp.json的绝对地址 + 其中的内容 生成一个唯一的哈希值，作为该项目的指纹"""
    project = (project or Path.cwd()).resolve()
    digest = hashlib.sha256(str(project).encode())
    for relative in (".bear/settings.json", ".mcp.json"):
        path = project / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return digest.hexdigest()


def is_project_trusted(project: Path | None = None) -> bool:
    """is_project_trusted函数用于检查给定的项目是否已经被信任
    考虑project地址的指纹是否再trust_file()中存在，如果存在则说明该项目已经被信任，返回True，否则返回False"""
    project = (project or Path.cwd()).resolve()
    try:
        records = json.loads(trust_file().read_text())
        return records.get(str(project)) == project_fingerprint(project)
    except (OSError, ValueError, AttributeError):
        return False


def trust_project(project: Path | None = None) -> None:
    project = (project or Path.cwd()).resolve()
    path = trust_file()
    # 这里的with，如果观察内部可以发现其本质就是with lock 和 with path.with_name(path.name + ".lock").open("a+b") as stream
    # 当然同进程同线程的话重入不会再次进入with path.with_name(path.name + ".lock").open("a+b") as stream
    with state_lock(path):
        records = json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(records, dict):
            raise ValueError("Invalid project trust registry")
        records[str(project)] = project_fingerprint(project)
        write_json(path, records)
