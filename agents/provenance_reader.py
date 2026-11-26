"""Incremental, bounded local index of append-only evolution evidence."""
from __future__ import annotations

from collections import defaultdict, deque
from contextlib import closing
import json
from pathlib import Path
import sqlite3


def load_recent_sources(path: Path, limit: int, *, persist: bool = True) -> list[dict]:
    """根据path下面的文件加载每个skill最多limit条最近的记录，返回一个列表
    在persist为True时，会把path下面的文件内容增量加载到sqlite数据库中，并返回数据库中每个skill最多limit条最近的记录"""
    if not path.exists():
        return []
    # 如果不持久化，直接从文件中读取最近的limit条记录即可
    if not persist:
        # 创建一个字典 groups，当访问不存在的键时，会自动创建一个有最大长度限制的双端队列作为默认值。
        groups = defaultdict(lambda: deque(maxlen=limit))
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                name = row.get("skill") or (row.get("result") or {}).get("skill")
                if name:
                    groups[name].append(row)
        return [row for rows in groups.values() for row in rows]
    database = path.parent / "online-eval" / "provenance.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database, timeout=10)) as connection, connection:
        connection.execute("CREATE TABLE IF NOT EXISTS source (id INTEGER PRIMARY KEY, skill TEXT, payload TEXT)")
        connection.execute("CREATE INDEX IF NOT EXISTS source_skill ON source(skill, id)")
        connection.execute("CREATE TABLE IF NOT EXISTS cursor (key TEXT PRIMARY KEY, value TEXT)")
        # Serialize cursor advancement and inserts across concurrent evaluators.
        # BEGIN IMMEDIATE 立即获取写锁
        connection.execute("BEGIN IMMEDIATE")
        # cursor专门有一行key为position，value是一个json字符串，里面包含了inode、offset和mtime三个字段
        # 其作用是用来跟path的基本信息做校验
        previous = connection.execute("SELECT value FROM cursor WHERE key='position'").fetchone()
        state = json.loads(previous[0]) if previous else {}
        stat = path.stat()
        offset = state.get("offset", 0)
        # 拿state的字段和stat的属性进行比较，如果变化source清空，重新从path读取，offset重置为0
        # inode变了：文件被替换（如 logrotate 轮转）
        # st_size < offset：文件被截断
        # 大小没变但 mtime 变了：文件被重写（同样大小但内容不同）
        if (state.get("inode") != stat.st_ino or stat.st_size < offset
                or (stat.st_size == offset and state.get("mtime") != stat.st_mtime_ns)):
            connection.execute("DELETE FROM source")
            offset = 0
        # "rb": read+binary，二进制读取文件，避免编码问题
        with path.open("rb") as stream:
            # 重新设置读取位置为offset
            stream.seek(offset)
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                name = row.get("skill") or (row.get("result") or {}).get("skill")
                # 把path中读取到的每一行json数据插入到sqlite数据库中
                if name:
                    connection.execute("INSERT INTO source(skill,payload) VALUES(?,?)", (name, json.dumps(row, ensure_ascii=False)))
            offset = stream.tell()
        # 修剪旧数据，对每个skill，只保留id最大的limit条（即最新的），其余删除
        for (name,) in connection.execute("SELECT DISTINCT skill FROM source").fetchall():
            connection.execute("DELETE FROM source WHERE skill=? AND id NOT IN (SELECT id FROM source WHERE skill=? ORDER BY id DESC LIMIT ?)", (name, name, limit))
        # 记录新的读取位置、path的inode和mtime，供下次增量读取。事务结束（with 块退出）时自动提交。
        connection.execute("INSERT OR REPLACE INTO cursor VALUES('position',?)", (json.dumps({"inode": stat.st_ino, "offset": offset, "mtime": stat.st_mtime_ns}),))
        return [json.loads(row[0]) for row in connection.execute("SELECT payload FROM source ORDER BY id")]
