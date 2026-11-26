"""Bounded, recoverable Skill learning jobs owned by the main runtime."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import uuid
import hashlib


class BackgroundSkillTasks:
    def _capture_skill_feedback(self, feedback: str) -> None:
        """捕捉技能反馈并将_pending_skill_extraction_window中的消息+feedback添加到已经有反馈后的技能反馈列表_skill_feedback_outbox中"""
        window = self._pop_pending_skill_extraction_window(feedback)
        if window is not None:
            self._skill_feedback_outbox.append(window)
            # Save before requesting the next answer: its failure/cancellation
            # must not consume feedback for an already completed turn.
            # 涉及到session，记得保存
            self._auto_save(strict=True)

    def _flush_skill_feedback(self) -> None:
        """这个方法的本质就是把_skill_feedback_outbox中所有的已经评价好的反馈对话，分别在后台启动一个任务，然后让他们执行evolution的操作"""
        if self.permission_mode == "plan" or not self._online_evolution_enabled():
            return
        while self._skill_feedback_outbox:
            window = self._skill_feedback_outbox[0]
            identity = json.dumps(window, ensure_ascii=False, sort_keys=True)
            # 确定后台任务的json文件名称
            job_id = "feedback-" + hashlib.sha256(identity.encode()).hexdigest()
            # 开启一个后台任务来处理技能反馈，_run_online_skill_evolution是一个异步方法
            accepted = self._schedule_background_skill_task(
                self._run_online_skill_evolution(window),
                job={"kind": "evolution", "window": window}, job_id=job_id)
            if not accepted:
                break
            self._skill_feedback_outbox.pop(0)
            # 每一次开启一个后台任务后，都需要保存session文件，因为_skill_feedback_outbox是持久化在session文件中的
            self._auto_save(strict=True)

    def _background_jobs_dir(self) -> Path:
        from .skill_evolution import get_evolution_dir
        return get_evolution_dir() / "pending-jobs"

    def _resume_background_skill_jobs(self) -> None:
        """定义恢复后台 Skill 任务的方法。"""

        # 以下任意条件成立，就不恢复后台任务：
        # 1. 当前实例是子代理；
        # 2. 在线演进功能没有开启；
        # 3. 当前处于 plan（规划）模式。
        #
        # or 使用短路求值：
        # 例如 is_sub_agent 为 True，后面的条件就不会继续计算。
        if self.is_sub_agent or not self._online_evolution_enabled() or self.permission_mode == "plan":
            return
        # 从./.bear/skill-evolution/pending-jobs目录下读取所有的json文件，尝试恢复这些后台任务
        for path in sorted(self._background_jobs_dir().glob("*.json")):
            if len(self._background_skill_tasks) >= self.background_skill_queue_limit:
                break
            # 如果这个任务已经在当前Agent的后台任务列表中，则跳过
            if path.stem in self._background_job_ids:
                continue
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(job, dict):
                    raise ValueError("Invalid persisted Skill job")
                # 获取尝试次数；字段不存在时默认是 0。
                # 已经尝试至少 3 次的任务，本次不再恢复。
                if job.get("attempts", 0) >= 3:
                    continue
                # 如果任务记录了非假值的进程 ID，并且所属会话不是当前会话，就需要判断它是否仍被其他进程持有。
                # 这个判断想排除的是：已经被其他会话接管、可能还在执行的任务，避免同一个任务被重复执行。
                if job.get("owner_pid") and job.get("owner_session") != self.session_id:
                    # Windows：直接跳过这个任务。这里采取保守做法，只要记录属于其他会话，就暂时不接手。
                    # 也就是说windows上采取保守策略，避免跨会话的进程检查
                    if os.name == "nt":
                        continue
                    try:
                        # 在非 Windows 平台上，对指定进程发送信号 0。
                        # 信号 0 不终止进程，主要用于检查进程是否存在、以及当前进程是否有权限访问它。
                        # 进程存在 → continue，跳过任务，避免抢走正在处理的任务。
                        # 进程不存在 → pass，继续后面的恢复流程，接手遗留任务。
                        os.kill(job["owner_pid"], 0)
                        continue
                    except ProcessLookupError:
                        pass
                data = job["data"]
                # 如果发现当前的任务是技能演进任务，就调用 _run_online_skill_evolution 方法恢复它；
                if data["kind"] == "evolution":
                    # 这里的coro是一个协程对象，因为_run_online_skill_evolution是一个异步方法，调用它会返回一个协程对象。
                    # 如果想要结果需要await这个协程对象，或者把它交给事件循环去执行。
                    coro = self._run_online_skill_evolution(data["window"])
                # 如果发现当前的任务是技能使用追踪任务，就调用 _run_skill_usage_tracking 方法恢复它；
                elif data["kind"] == "usage":
                    # 这里的coro是一个协程对象，因为_run_online_skill_evolution是一个异步方法，调用它会返回一个协程对象。
                    # 如果想要结果需要await这个协程对象，或者把它交给事件循环去执行。
                    coro = self._run_skill_usage_tracking(data["user"], data["assistant"], hits=data["hits"])
                else:
                    continue
                from .storage import state_lock
                # 前面读json不上锁，这里上锁是为了保证当前阶段判断跟前者的时候不会有被再次接受的可能性
                with state_lock(path):
                    # Re-read ownership after acquiring the cross-process claim.
                    # 重新读取磁盘上的最新任务记录
                    latest = json.loads(path.read_text(encoding="utf-8"))
                    # 如果最新记录有持有进程，且最新所属会话与第一次读取时job持有的进程不同，
                    # 就认为任务在前后读取期间已被其他会话接管。
                    # 此时协程对象应该被关闭
                    if latest.get("owner_pid") and latest.get("owner_session") != job.get("owner_session"):
                        coro.close()
                        continue
                    # 二确后，发现还是没有变化，此时就可以把当前进程的pid和session_id写入任务记录，表示当前进程接管了这个任务
                    self._schedule_background_skill_task(coro, job=data, job_id=path.stem)
            except (OSError, ValueError, KeyError, TypeError):
                logging.exception("Unable to recover background Skill job %s", path)

    def _schedule_background_skill_task(self, coro, *, job: dict | None = None, job_id: str | None = None) -> bool:
        """根据提供的job_id以及配合的job参数创建一个新的后台的skill任务(当然有可能是后台重试/重启调用的skill，也可能是新生成的任务)"""
        # 如果当前处于计划模式，或者当前任务不为空且在线演进功能没有开启，则直接关闭协程对象并返回False，表示不接受这个后台任务。
        if self.permission_mode == "plan" or (job is not None and not self._online_evolution_enabled()):
            try:
                coro.close()
            except Exception:
                pass
            return False
        # 如果都不在后台任务id中就认为他是非法的
        if job_id and job_id in self._background_job_ids:
            coro.close()
            return True
        from .storage import write_json
        path = None
        record = None
        if job is not None:
            job_id = job_id or uuid.uuid4().hex
            path = self._background_jobs_dir() / f"{job_id}.json"
            try:
                # 如果是重启历史任务，那么原本的f"{job_id}.json"下面必然有历史信息
                # 反之则返回一个空消息
                previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                record = {"data": job, "status": "queued", "attempts": previous.get("attempts", 0),
                          "owner_pid": os.getpid(), "owner_session": self.session_id}
                # 在执行前更新信息,覆盖写
                write_json(path, record)
            except Exception:
                coro.close()
                raise
        # 如果当前的后台任务列表超过了上线，暂时把这个任务的coro关闭了
        # 警告后续会重新重试这个任务的
        if len(self._background_skill_tasks) >= self.background_skill_queue_limit:
            coro.close()
            logging.warning("Background Skill queue full; persisted jobs will retry on the next turn")
            return path is not None
        # 如果后台调度信号量变量是None，则根据并发量定义信号量变量
        if self._background_semaphore is None:
            self._background_semaphore = asyncio.Semaphore(self.background_skill_concurrency)

        # 定义任务执行的模板函数
        async def bounded():
            try:
                # 用信号量来控制并发数
                async with self._background_semaphore:
                    if path:
                        # 更新执行状态，并增加尝试次数
                        record.update(status="running", attempts=record["attempts"] + 1)
                        write_json(path, record)
                    # 执行传入的协程对象，等待其完成，并获取结果
                    result = await coro
                    if path:
                        # 如果执行失败，更新record信息
                        if isinstance(result, dict) and result.get("ok") is False:
                            record.update(status="failed", error=str(result.get("error", "operation failed")))
                            write_json(path, record)
                        else:
                            # 把path对应的json文件删除掉，表示这个后台任务已经完成了
                            path.unlink(missing_ok=True)
            except asyncio.CancelledError:
                if path:
                    record["status"] = "queued"
                    record["owner_pid"] = None
                    write_json(path, record)
                raise
            except Exception as exc:
                if path:
                    record.update(status="failed", error=str(exc))
                    write_json(path, record)
                raise
            finally:
                coro.close()
        # 创建一个新的任务注册到循环事件中来执行bounded函数，
        task = asyncio.create_task(bounded())
        # 并把这个任务加入到后台技能任务集合中
        self._background_skill_tasks.add(task)
        if job_id:
            # 把后台任务的文件名放到后台任务id集合中，表示当前这个后台任务已经被接管了
            self._background_job_ids.add(job_id)

        def _done(done_task: asyncio.Task) -> None:
            # 对刚刚结束的任务做收尾处理。
            # 剔除后台的done_task，表示这个后台任务已经结束了
            # 剔除后台的任务文件名job_id，表示这个后台任务已经结束了
            self._background_skill_tasks.discard(done_task)
            if job_id:
                self._background_job_ids.discard(job_id)
            try:
                # 返回done_task的结果，如果done_task在执行过程中抛出了异常，这里会捕获到异常并打印日志
                done_task.result()
            except asyncio.CancelledError:
                logging.info("Background Skill task cancelled")
            except Exception:
                logging.exception("Background Skill task failed")

        # 将回调注册到这个任务上。
        # 结束时，调用 _done，并把你自己作为参数传进去。
        task.add_done_callback(_done)
        return True

    async def drain_background_skill_tasks(self, *, cancel_on_timeout: bool = False) -> None:
        """
        等待Agent后台的所有技能任务都完成
        Wait for evaluation inputs; only shutdown may cancel unfinished work."""
        tasks = [task for task in self._background_skill_tasks if not task.done()]
        if tasks:
            if not cancel_on_timeout:
                # 如果cancel_on_timeout为False，表示不需要在超时后取消任务，而是等待所有任务完成
                # 这种情况如果外部例如/skill-eval被取消了，理论上也不应该影响gather操作
                await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
                return
            # 没有timeout，就直接按照shutdown_timeout时间进行有限时间wait
            # pending是剩下来不及完成的任务，可能是因为shutdown_timeout时间不够长，或者是因为这些任务本身需要等待外部的资源才能完成
            _, pending = await asyncio.wait(tasks, timeout=self.shutdown_timeout)
            if pending:
                logging.warning("Cancelling %s background Skill tasks after %.1fs", len(pending), self.shutdown_timeout)
                # 把所有pending任务全部取消掉
                for task in pending:
                    task.cancel()
                # task.cancel()只是下了cancel的命令，并不会立即取消任务，所以这里需要等待所有pending任务都被取消掉
                await asyncio.gather(*pending, return_exceptions=True)

