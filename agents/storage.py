"""Crash-safe state files and cooperative locks for local runtime writers."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat
import tempfile
import threading


_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()
_held = threading.local()

# @contextmanager这个装饰器就是需要根据装饰器要求的结构进行一个包装，包装的函数必须是一个生成器函数，
# yield前面的代码就是__enter__方法的内容，yield后面的final代码就是__exit__方法的内容
# 它可以配合with...as...语句使用，as后面是yield的返回值，with语句块执行完毕后会自动调用__exit__方法
# 它还可以配合next()/for()语句使用
@contextmanager
def state_lock(path: Path):
    """
    state_lock是一个上下文管理器函数，它的作用是对指定的文件路径进行加锁，确保在同一时间只有一个线程可以对该文件进行读/写操作，从而避免数据竞争和不一致的问题
    他采用了Lock+reentrant lock+ Os.lock(操作系统层面的文件锁)的方式，保证了同一线程可以多次获取锁而不会死锁，同时也保证了不同线程之间的互斥访问
    这里不仅仅采用Reentrant Lock(只能保证同一个进程内的多线程互斥)，但是多进程(同一个dir下多开程序)的互斥是无法实现的，因此需要使用操作系统层面的文件锁来实现跨进程的互斥访问

    Lock the entire read/modify/write operation, including across processes.

    Lock files are stable sidecars: never unlink them while another writer may
    hold or wait for the lock. All runtime writers must use the same path.
    """
    path = Path(path).resolve()
    key = str(path)
    with _locks_guard:
        lock = _locks.setdefault(key, threading.RLock())
    # 可重入锁，如果是同进程不同线程的在这里会被直接blocked住，直到获取到锁为止
    # lock的__enter__方法本质就是acquire()方法，__exit__方法本质就是release()方法
    # 所以with lock语句块执行前会进行__enter__方法，执行完毕后会自动调用__exit__方法释放锁
    with lock:
        held = getattr(_held, "paths", None)
        if held is None:
            held = _held.paths = set()
        # 这里如果在held里面表示的就是当前的path已经被当前线程的thread local持有了
        # 不需要去重复内部加锁.lock文件了
        if key in held:
            yield
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # 如果是多进程操作同一个文件，就用最后的.lock文件来进行跨进程的互斥访问
        with path.with_name(path.name + ".lock").open("a+b") as stream:
            if os.name == "nt":
                import msvcrt
                # msvcrt.locking函数是Windows平台下的文件锁定函数，它可以对文件进行加锁和解锁操作
                # stream.seek(0, os.SEEK_END)主要是将文件指针移动到文件末尾
                stream.seek(0, os.SEEK_END)
                # 如果文件为空，则写入一个空字节，确保文件存在并且有内容，这样才能进行锁定操作
                if not stream.tell():
                    stream.write(b"\0")
                    stream.flush()
                # 重新重置读写位置
                stream.seek(0)
                # stream.fileno()：拿到底层 OS 文件描述符（整数），msvcrt.locking 需要 fd 而不是文件对象，锁定1个字节
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            held.add(key)
            try:
                yield
            finally:
                held.remove(key)
                if os.name == "nt":
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def atomic_write_text(path: Path, content: str) -> None:
    """atomic_write_text本质上是在Path.parent目录下创建一个临时文件,并写入content内容，然后原子地将临时文件替换为最终文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 在目标文件所在目录里创建一个临时文件，返回文件描述符 fd 和临时文件路径 temporary。
    # prefix =  . + 目标文件名，是"隐藏临时文件"的惯例
    # 关键点：临时文件必须和目标文件在同一目录/同一文件系统，否则后面的 os.replace 会退化成跨设备拷贝，失去原子性。
    # fd是os.open()返回的文件描述符，temporary是临时文件的名字
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        # 把文件描述符 fd 包装成文本流（"w" 模式，UTF-8 编码）
        # hasattr(os, "fchmod") 是兼容性判断（Windows 上没有 fchmod），有才执行
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            if path.exists() and hasattr(os, "fchmod"):
                # 保留原文件权限：读它的权限位（stat.S_IMODE 只取权限部分，不含文件类型），应用到临时文件上。
                os.fchmod(stream.fileno(), stat.S_IMODE(path.stat().st_mode))
            stream.write(content)
            # 把python缓冲区的内容写入操作系统缓冲区
            stream.flush()
            # 强制操作系统把数据真正写到磁盘（持久化）
            os.fsync(stream.fileno())
        # 原子替换：把临时文件移动/重命名为目标文件，覆盖原文件。os.replace 在同一文件系统内是原子操作。
        os.replace(temporary, path)
        # os.replace 改变了目录内容（文件名映射），但"重命名"这个元数据操作本身可能还在页缓存里。对目录做 fsync，确保 rename 操作也落盘。
        # 这里只对非 Windows 系统做 fsync，因为 Windows 上的文件系统行为不同，通常不需要对目录做 fsync。
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2,
                                     sort_keys=True, default=str) + "\n")


def append_jsonl(path: Path, value) -> None:
    """append_jsonl函数用于将一个Python对象以JSON格式追加写入到指定的文件中，每个对象占一行，形成JSON Lines格式的文件"""
    with state_lock(path):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
