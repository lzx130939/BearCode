"""CLI entry point and interactive REPL — mirrors cli.ts."""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import threading
from pathlib import Path

from .agent import Agent
from .config import (
    load_env_file as _load_env_file,
    resolve_api_config as _resolve_api_config,
    resolve_agent_options,
)
from .ui import (
    print_welcome,
    print_user_prompt,
    print_error,
    print_info,
    print_plan_for_approval,
    print_plan_approval_options,
    print_goodbye,
    print_interrupted,
    print_memory_entries,
    print_skill_entries,
    print_warning,
)
from .session import load_session, get_latest_session_id
from .skill_evolution import restore_pruned_skill
from .tools import check_permission
from .memory import list_memories
from .session_detail import build_session_detail, print_session_detail
from .skills import (
    create_skill,
    discover_skills,
    evolve_skill,
    execute_skill,
    get_skill_by_name,
    record_feedback,
    skill_stats,
)
from .online_skill_eval import format_online_skill_eval_async


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mini-claude",
        description="Bear Code — a minimal coding agent",
        add_help=False,
    )
    parser.add_argument("prompt", nargs="*", help="One-shot prompt")
    parser.add_argument("--yolo", "-y", action="store_true", help="Skip all confirmation prompts")
    parser.add_argument("--plan", action="store_true", help="Plan mode: read-only")
    parser.add_argument("--accept-edits", action="store_true", help="Auto-approve file edits")
    parser.add_argument("--dont-ask", action="store_true", help="Auto-deny confirmations (for CI)")
    parser.add_argument("--thinking", action="store_true", help="Enable extended thinking")
    parser.add_argument("--model", "-m", default=None, help="Model to use")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--resume", action="store_true", help="Resume latest session in this project")
    parser.add_argument("--resume-id", help="Resume a specific session in this project")
    parser.add_argument("--input-price", type=float, default=None, help="Estimated USD per million input tokens")
    parser.add_argument("--output-price", type=float, default=None, help="Estimated USD per million output tokens")
    parser.add_argument("--max-cost", type=float, default=None, help="Max USD spend")
    parser.add_argument("--max-turns", type=int, default=None, help="Max agentic turns")
    parser.add_argument("--context-window", type=int, default=None, help="Model context capacity in tokens")
    parser.add_argument("--no-folding", action="store_true", help="Disable full session folding")
    parser.add_argument("--fold-threshold", type=float, default=0.70, help="Auto-fold at this fraction of the context window")
    parser.add_argument("--help", "-h", action="store_true", help="Show help")
    return parser.parse_args()


def _resolve_permission_mode(args: argparse.Namespace) -> str:
    """
    bypassPermissions（完全放行）
    plan（只读规划）
    acceptEdits（自动接受编辑）（自动创建和修改文件，无需逐次确认）
    dontAsk（自动拒绝确认）
    default（兜底默认）
    """
    if args.yolo:
        return "bypassPermissions"
    if args.plan:
        return "plan"
    if args.accept_edits:
        return "acceptEdits"
    if args.dont_ask:
        return "dontAsk"
    return "default"

# asyncio.Future本质是"一个将来会有结果的占位符"，如果某处尝试获取他的值的时候如果没有，那么就会等待获取
# 因此他存在挂起/唤醒等操作，所以他是跟Event loop对象强制绑定在一起的
_pending_input: asyncio.Future | None = None


async def _read_repl_input(prompt: str = "") -> str:
    """ _read_repl_input函数就是按照prompt参数，通过stdio的形式展示，并接收到user的返回的答复
    但是由于存在多个tasks需要同一个输入进行交互，所以考虑了多个协程复用Future产生的一系列问题
    包括使用Future的shield方法保护Future不被取消，使用Event loop的call_soon_threadsafe方法注册回调函数等
    Let background tasks run while stdin waits; don't keep exit alive on stdin."""
    global _pending_input
    # asyncio.get_running_loop()返回当前执行该协程的线程对应的`Event loop`对象
    # 就是理解为一个线程里面通过一个while循环调度所有的任务，并且他还会处理sleep、恢复挂起的任务等
    loop = asyncio.get_running_loop()

    # 如果pending_input不为None并且与当前的Event loop相同
    # 其实找的就是同一个event loop调用的不同协程之间的共享的Future对象
    # 比如这种情况：同一协程函数体内的 await，其目标协程都在同一个 event loop 上执行。
    if _pending_input is not None and _pending_input.get_loop() is loop:
        pending = _pending_input
        try:
            # 这里的await asyncio.shield(arg)的意思是：arg是一个协程对象或者Future对象，shield的意思是保护这个Future对象不被取消
            # 这里不被取消的意思是，即使外部的async函数被cannel了，正常情况下内部await的arg也会被取消，但是shield保护了arg，所以arg不会被取消
            # 他依然能够从pending这里获取到value值。
            # 这么设计的逻辑是因为可能存在超时或者其他原因导致外部的async函数被取消，但是我们希望这个输入的Future对象不被取消，依然能够获取到用户的输入
            # 虽然即使外部可能早已停止了
            value = await asyncio.shield(pending)
        except asyncio.CancelledError:
            # raise表示原样输出异常
            raise
        except BaseException:
            _pending_input = None
            raise
        _pending_input = None
        return value
    # 从这个Event loop上创建一个Future对象，这个Future对象是和这个Event loop绑定的
    result = _pending_input = loop.create_future()

    def complete(value: str | None, error: BaseException | None) -> None:
        if not result.done():
            if error is not None:
                result.set_exception(error)
            else:
                result.set_result(value)

    def read() -> None:
        value, error = None, None
        try:
            # 从标准输入读取一个字符串。末尾的换行符将被去除。
            # 如果指定了提示字符串，则在读取输入之前，将其打印到标准输出，且不带末尾换行符。
            value = input(prompt) if prompt else input()
        except (EOFError, OSError) as exc:
            error = exc
        try:
            # 像这个Event loop对象中注册一个complete(value, error)函数
            loop.call_soon_threadsafe(complete, value, error)
        except RuntimeError:
            pass  # The loop may already have closed after cancellation/exit.

    # 调用子线程异步执行read()函数，本质就是阻塞读取stdio的数据流
    # daemon=True表示这个线程是守护线程，主线程退出的时候这个线程也会退出,不会因为等待输入而阻塞主线程关闭
    threading.Thread(target=read, daemon=True).start()
    try:
        value = await asyncio.shield(result)
    except asyncio.CancelledError:
        raise
    except BaseException:
        _pending_input = None
        raise
    _pending_input = None
    return value


async def run_repl(agent: Agent) -> None:
    # 这个操作是希望先把signal.SIGINT这个信号(本质上是ctrl+c)的处理函数保存起来，
    # 等到REPL结束后再恢复原来的处理函数。
    previous_handler = signal.getsignal(signal.SIGINT)
    try:
        await _run_repl(agent)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        await agent.aclose()


async def _run_repl(agent: Agent) -> None:
    """Interactive REPL loop."""

    async def confirm_fn(message: str) -> bool:
        try:
            answer = await _read_repl_input("  Allow? (y/n): ")
            return answer.lower().startswith("y")
        except EOFError:
            return False

    agent.set_confirm_fn(confirm_fn)

    async def plan_approval_fn(plan_content: str) -> dict:
        print_plan_for_approval(plan_content)
        print_plan_approval_options()
        while True:
            try:
                choice = (await _read_repl_input("  Enter choice (1-4): ")).strip()
            except EOFError:
                return {"choice": "cancel"}
            if choice == "1":
                return {"choice": "clear-and-execute"}
            elif choice == "2":
                return {"choice": "execute"}
            elif choice == "3":
                return {"choice": "manual-execute"}
            elif choice == "4":
                try:
                    feedback = (await _read_repl_input("  Feedback (what to change): ")).strip()
                except EOFError:
                    feedback = ""
                return {"choice": "keep-planning", "feedback": feedback or None}
            else:
                print_warning("Invalid choice. Enter 1, 2, 3, or 4.")

    agent.set_plan_approval_fn(plan_approval_fn)

    sigint_count = 0

    # 上面在run_repl函数中保存了signal.SIGINT的处理函数，这里是重新注册一个新的处理函数
    def handle_sigint(sig, frame):
        nonlocal sigint_count
        if agent.is_processing:
            # Agent is processing
            agent.abort()
            print_interrupted()
            sigint_count = 0
            print_user_prompt()
        else:
            sigint_count += 1
            if sigint_count >= 2:
                print_goodbye()
                sys.exit(0)
            print_warning("Press Ctrl+C again to exit.")
            print_user_prompt()

    signal.signal(signal.SIGINT, handle_sigint)
    print_welcome()

    while True:
        print_user_prompt()
        try:
            line = await _read_repl_input()
        except (EOFError, KeyboardInterrupt):
            print_goodbye()
            break

        inp = line.strip()
        sigint_count = 0

        if not inp:
            continue
        if inp in ("exit", "quit"):
            print_goodbye()
            break

        # REPL commands
        # clear命令就是清除相关的参数，具体还没看
        if inp == "/clear":
            agent.clear_history()
            continue
        # trust-project就是把增加cwd与之对应的指纹信息到~/.bear/trusted-projects.json文件中
        if inp == "/trust-project":
            from .project_trust import trust_project
            try:
                trust_project()
                print_info(f"Trusted project configuration: {Path.cwd().resolve()}")
            except (ValueError, OSError) as exc:
                print_error(str(exc))
            continue
        # cleanup-state命令就是清理.bear/state目录下的文件，默认是30天前的文件，--apply表示真的执行删除操作
        if inp.startswith("/cleanup-state"):
            from .maintenance import cleanup_state
            try:
                parts = inp.split()[1:]
                apply = "--apply" in parts
                days = next((int(part) for part in parts if part != "--apply"), 30)
                # preview的格式： {"applied": apply, "count": len(paths), "paths": [str(path) for path in paths]}
                preview = cleanup_state(days)
                print_info(str(preview))
                # 只有apply为True并且授权通过才会真正执行删除操作
                if apply and await agent.authorize_operation("state_cleanup", preview):
                    # 可以看到这里的expected_paths本质上是先运行了一次没有删文件的cleanup_state，然后在确认后二次执行删除
                    print_info(str(cleanup_state(days, apply=True, expected_paths=preview["paths"])))
            except (ValueError, OSError) as exc:
                print_error(str(exc))
            continue
        # 切换plan模式的开关
        if inp == "/plan":
            agent.toggle_plan_mode()
            continue
        # 显示当前进程Agent开销
        if inp == "/cost":
            agent.show_cost()
            continue
        if inp == "/session-detail" or inp.startswith("/session-detail "):
            options = inp.split()[1:]
            if options not in ([], ["-d"], ["--full"]):
                print_error("Usage: /session-detail [-d]")
                continue
            try:
                print_session_detail(build_session_detail(agent), full=bool(options))
            except (OSError, ValueError, TypeError) as exc:
                print_error(f"Cannot generate report: {exc}")
            continue
        # 压缩会话(这里本质上是手动调用)
        if inp == "/compact":
            try:
                await agent.compact()
            except Exception as e:
                print_error(str(e))
            continue
        # 展示cwd下的所有记忆文件(除索引MEMORY.md之外)
        if inp == "/memory":
            memories = list_memories()
            if not memories:
                print_info("No memories saved yet.")
            else:
                print_memory_entries(memories)
            continue
        # 展示当先cwd + 系统下的skill
        if inp == "/skills":
            skills = discover_skills()
            if not skills:
                print_info("No skills found. Add skills to .bear/skills/<name>/SKILL.md")
            else:
                print_skill_entries(skills)
            continue
        # skill-candidate（技能候选）指的是一个由 Agent 自动生成的、尚未正式生效的“技能提案”——它处于“候选”状态，等待被审核、采纳或丢弃。
        if inp == "/skill-candidates":
            from .skill_candidates import list_candidates
            try:
                for proposal in list_candidates():
                    print_info(f"{proposal['id']} | {proposal['status']} | {proposal['action']} {proposal['skill']}\n"
                               f"{proposal['write_input'].get('instructions', '')}")
            except (ValueError, OSError, KeyError, TypeError) as exc:
                print_error(str(exc))
            continue
        # candidate的基本格式(猜测):
        # {
        #     "id": "0123456789abcdef01234567",
        #     "status": "pending",
        #     "action": "add",
        #     "skill": "python_error_analysis",
        #     "write_input": {
        #         "name": "python_error_analysis",
        #         "description": "分析 Python 异常并定位原因",
        #         "instructions": "先读取完整堆栈，再检查函数调用链和参数类型。",
        #         "when_to_use": "用户需要排查 Python 异常时",
        #         "target": "project",
        #         "context": "inline",
        #         "user_invocable": false,
        #         "evidence": "用户要求分析异常时追踪上游调用，而不只检查报错行。",
        #         "actor": "online",
        #         "tags": ["python", "debugging"]
        #     },
        #     "base_hash": null,
        #     "snapshot_date": "2026-11-06",
        #     "snapshot": {
        #         "name": "python_error_analysis",
        #         "description": "分析 Python 异常并定位原因",
        #         "instructions": "# Skill Instructions\n\n先读取完整堆栈，再检查函数调用链和参数类型。\n\n## Creation Evidence\n\n用户要求分析异常时追踪上游调用，而不只检查报错行。",
        #         "candidate_id": "0123456789abcdef01234567",
        #         "proposal_hash": "<程序计算的64位SHA-256哈希>"
        #     }
        # }
        # skill-apply命令就是手动的把在skill_candidates目录下的proposal文件应用到当前的技能目录下，成为一个正式的技能，
        # 所以一般来说/skill-apply <id> [--manual-review]
        # --manual-review 表示由你人工审核候选技能，跳过“必须有匹配的自动评测通过记录”这项要求，也就是不需要考虑champion.json了。
        if inp.startswith("/skill-apply "):
            from .skill_candidates import read_candidate, apply_candidate
            try:
                candidate_id = inp[len("/skill-apply "):].strip()
                parts = candidate_id.split()
                if not parts or len(parts) > 2 or (len(parts) == 2 and parts[1] != "--manual-review"):
                    raise ValueError("Usage: /skill-apply <id> [--manual-review]")
                candidate_id = parts[0]
                proposal = read_candidate(candidate_id)
                import hashlib
                import json
                # 计算proposal对象的哈希值，作为proposal_hash
                proposal_hash = hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()
                print_info(f"Publish proposed Skill {proposal['skill']}:\n{proposal['write_input'].get('instructions', '')}")
                # 确定这个候选者到底是merge还是add操作，merge就是在原有的技能上进行修改，add就是新增一个技能
                operation = "skill_evolve" if proposal["action"] == "merge" else "skill_create"
                if await agent.authorize_operation(operation, proposal["write_input"]):
                    result = apply_candidate(candidate_id, manual_review="--manual-review" in parts, expected_hash=proposal_hash)
                    print_info(str(result))
                    # 不管怎么样，skill的状态发生了变化，所以需要刷新系统提示词
                    agent._refresh_runtime_system_prompt()
                else:
                    print_info("Skill publication denied.")
            except (ValueError, OSError) as exc:
                print_error(str(exc))
            continue
        #  skill-rollback命令就是把当前的技能回滚到上一个快照版本
        if inp.startswith("/skill-rollback "):
            from .skill_candidates import rollback_skill
            name = inp[len("/skill-rollback "):].strip()
            if await agent.authorize_operation("skill_evolve", {"skill_name": name}):
                try:
                    print_info(str(rollback_skill(name)))
                    # skill变化，重置系统提示词
                    agent._refresh_runtime_system_prompt()
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    print_error(str(exc))
            else:
                print_info("Skill rollback denied.")
            continue
        # skill-stats命令就是展示当前的技能统计信息
        if inp == "/skill-stats":
            print_info(skill_stats())
            continue
        # skill-eval是评估技能演化质量的命令：它从历史记录中提取经过审核的任务，让当前技能和候选改进版本重新作答，再比较结果，判断改进是否有效。
        # 看了内部，只能用在online skill下面
        if inp == "/skill-eval":
            try:
                # Finish pending writes before reading provenance and active snapshots.
                await agent.drain_background_skill_tasks()
                side_query = agent._build_side_query(max_tokens=2400, require_complete=True)
                # 专门设定评估的专有模型，用judge_model再做一次评估，如果judge_model没有设置，则judge_query = None
                judge_model = os.environ.get("BEAR_SKILL_JUDGE_MODEL")
                judge_query = agent._build_side_query(max_tokens=2400, require_complete=True, model_override=judge_model) if judge_model else None
                print_info(await format_online_skill_eval_async(side_query=side_query, judge_side_query=judge_query))
            except (ValueError, OSError) as exc:
                print_error(f"Skill evaluation could not complete: {exc}")
            continue
        # 立刻提取技能，从pending_skill_extraction_window中获取原始对话信息
        if inp.startswith("/extract_now"):
            hint = inp[len("/extract_now") :].strip()
            result = await agent.extract_now(hint)
            if result.get("ok"):
                print_info("Ran online skill extraction for the current pending window.")
            else:
                print_error(str(result.get("error") or result))
            continue
        # 处理用户输入的 /skill-feedback 命令，从中提取技能名称、评分和备注，然后记录反馈。
        if inp.startswith("/skill-feedback "):
            _, rest = inp.split(" ", 1)
            parts = rest.strip().split(" ", 2)
            if len(parts) < 2:
                print_error("Usage: /skill-feedback <skill-name> <rating> [note]")
                continue
            note = parts[2] if len(parts) > 2 else ""
            record_feedback(parts[0], parts[1], note)
            print_info(f"Recorded feedback for skill: {parts[0]}")
            continue
        #  技能演化命令：/skill-evolve <skill-name> <durable lesson>，其中durable lesson是一个持久化的技能改进建议
        if inp.startswith("/skill-evolve "):
            _, rest = inp.split(" ", 1)
            parts = rest.strip().split(" ", 1)
            if len(parts) < 2:
                print_error("Usage: /skill-evolve <skill-name> <durable lesson>")
                continue
            if not await agent.authorize_operation("skill_evolve", {"skill_name": parts[0], "lesson": parts[1]}):
                print_info("Skill evolution denied.")
                continue
            # 调用evolve_skill函数进行技能演化，传入技能名称、lesson、理由(手动REPL模式调用演化)和目标(active，默认遍历所有默认skill的地址位置)
            result = evolve_skill(parts[0], parts[1], rationale="Manual REPL evolution", target="active")
            if result.get("ok"):
                print_info(f"Evolved skill {result.get('skill')} to version {result.get('version')}")
            else:
                print_error(str(result.get("error") or result))
            continue
        # /skill-restore 用来恢复之前被归档或移除的技能
        if inp.startswith("/skill-restore "):
            name = inp[len("/skill-restore "):].strip()
            # 这里的逻辑本质上就是agent.authorize_operation("skill_restore", {"skill_name": name})
            permission = check_permission("skill_restore", {"skill_name": name}, agent.permission_mode)
            if permission["action"] == "deny" or (permission["action"] == "confirm" and
                    not await agent._confirm_tool_call("skill_restore", permission)):
                print_info("Skill restoration denied.")
                continue
            try:
                result = restore_pruned_skill(name)
                # 恢复原本的技能后，需要重新更新当前在线的skill缓存，并刷新系统提示词
                if result.get("ok"):
                    from .skills import reset_skill_cache
                    reset_skill_cache()
                    agent._refresh_runtime_system_prompt()
                    print_info(f"Restored skill: {name}")
                else:
                    print_error(str(result.get("error")))
            except (ValueError, OSError) as exc:
                print_error(f"Skill restoration failed: {exc}")
            continue
        # 创建一个新的技能，需要正确满足四个参数：name、description、when-to-use、instructions，并且需要授权通过
        if inp.startswith("/skill-create "):
            _, rest = inp.split(" ", 1)
            parts = [part.strip() for part in rest.split("|", 3)]
            if len(parts) < 4 or not all(parts[:4]):
                print_error("Usage: /skill-create <name> | <description> | <when-to-use> | <instructions>")
                continue
            if not await agent.authorize_operation("skill_create", {"name": parts[0], "description": parts[1],
                    "when_to_use": parts[2], "instructions": parts[3], "target": "project"}):
                print_info("Skill creation denied.")
                continue
            result = create_skill(
                name=parts[0],
                description=parts[1],
                when_to_use=parts[2],
                instructions=parts[3],
                target="project",
                context="inline",
                user_invocable=False,
                # evidence参数是为了记录技能创建的证据来源，这里是手动REPL技能创建
                evidence="Manual REPL skill creation",
            )
            if result.get("ok"):
                print_info(f"Created skill {result.get('skill')} at {result.get('file')}")
            else:
                print_error(str(result.get("error") or result))
            continue

        # Skill invocation: /<skill-name> [args]
        if inp.startswith("/"):
            space_idx = inp.find(" ")
            cmd_name = inp[1:space_idx] if space_idx > 0 else inp[1:]
            cmd_args = inp[space_idx + 1:] if space_idx > 0 else ""
            # 查找当前的技能列表中是否存在这个技能名称
            skill = get_skill_by_name(cmd_name)
            if skill and skill.user_invocable:
                print_info(f"Invoking skill: {skill.name}")
                try:
                    # 如果技能的上下文是fork，那么就是让当前 Agent 通过 skill 工具调用技能
                    # 不调用execute_skill，而是把一句"请用 skill 工具调用 X"的自然语言指令塞进当前对话
                    # 具体怎么调用、是否创建子Agent、是否隔离，完全交给LLM和skill工具去决定
                    if skill.context == "fork":
                        await agent.chat(f'Use the skill tool to invoke "{skill.name}" with args: {cmd_args or "(none)"}')
                    else:
                        # inline把技能提示词交给当前 Agent
                        result = execute_skill(skill.name, cmd_args)
                        if not result:
                            print_error(f"Unknown skill: {skill.name}")
                            continue
                        await agent.chat(result["prompt"])
                except Exception as e:
                    if "abort" not in str(e).lower():
                        print_error(str(e))
                continue

        # Normal chat
        try:
            await agent.chat(inp)
        except Exception as e:
            if "abort" not in str(e).lower():
                print_error(str(e))


async def run_one_shot(agent: Agent, prompt: str) -> None:
    if agent.permission_mode == "plan":
        async def plan_only_approval_fn(plan_content: str) -> dict:
            print_plan_for_approval(plan_content)
            return {"choice": "plan-only"}

        agent.set_plan_approval_fn(plan_only_approval_fn)

    try:
        await agent.chat(prompt)
    finally:
        await agent.aclose()


def main() -> None:
    """CLI 程序入口：准备运行配置，创建 Agent，并按参数选择一次性执行或交互模式。"""
    # 解析命令行参数，例如 --plan、--resume、--model，以及可选的一次性 prompt。
    args = parse_args()
    _load_env_file()

    if args.help:
        # 自定义帮助文本，展示 Bear Code 支持的启动参数和 REPL 内置命令。
        print("""
Usage: bear-code [options] [prompt]

Options:
  --yolo, -y          Skip all confirmation prompts (bypassPermissions mode)
  --plan              Plan mode: read-only, describe changes without executing
  --accept-edits      Auto-approve file edits, still confirm every shell command
  --dont-ask          Auto-deny anything needing confirmation (for CI)
  --thinking          Enable extended thinking (Anthropic only)
  --model, -m         Model to use (default: deepseek-chat, or MODEL env)
  --api-base URL      Override API base URL from CLI or .env
  --resume            Resume the latest session in the current project
  --resume-id ID      Resume a specific session in the current project
  --input-price USD   Estimated price per million input tokens (default: 3)
  --output-price USD  Estimated price per million output tokens (default: 15)
  --max-cost USD      Stop when estimated cost exceeds this amount
  --max-turns N       Stop after N agentic turns
  --context-window N Model context capacity in tokens
  --no-folding        Disable full session folding
  --fold-threshold X  Auto-fold threshold (0 < X < 1, default: 0.70)
  --help, -h          Show this help

REPL commands:
  /clear              Clear conversation history
  /plan               Toggle plan mode (read-only <-> normal)
  /cost               Show token usage and cost
  /session-detail     Show statistics, checkpoints and memory; add -d for messages and tool details
  /compact            Manually compact conversation
  /memory             List saved memories
  /skills             List available skills
  /skill-stats        Show skill usage and evolution stats
  /skill-eval         Evaluate online skill evolution quality
  /extract_now        Extract the current pending online skill window: /extract_now [hint]
  /skill-feedback     Record feedback: /skill-feedback <skill> <rating> [note]
  /skill-evolve       Evolve a skill: /skill-evolve <skill> <durable lesson>
  /skill-restore      Restore an archived skill: /skill-restore <skill>
  /skill-create       Create a skill: /skill-create <name> | <description> | <when-to-use> | <instructions>
  /skill-candidates   Review isolated online Skill proposals
  /skill-apply        Publish a passed candidate: /skill-apply <id> [--manual-review]
  /skill-rollback     Restore the most recent snapshot: /skill-rollback <skill>
  /trust-project      Trust current project permission/MCP configuration
  /cleanup-state      Preview old artifacts: /cleanup-state [days] [--apply]
  /<skill-name>       Invoke a skill (e.g. /commit "fix types")

Examples:
  mini-claude "fix the bug in src/app.ts"
  mini-claude --yolo "run all tests and fix failures"
  mini-claude --plan "how would you refactor this?"
  mini-claude --max-cost 0.50 --max-turns 20 "implement feature X"
  MODEL=deepseek-chat APIKEY=sk-xxx API=https://api.deepseek.com/anthropic mini-claude "hello"
  MODEL=gpt-4o OPENAI_API_KEY=sk-xxx OPENAI_BASE_URL=https://aihubmix.com/v1 mini-claude "hello"
  mini-claude --resume
  mini-claude  # starts interactive REPL
""")
        sys.exit(0)

    # 将命令行布尔开关统一转换成 Agent 内部使用的权限模式。
    permission_mode = _resolve_permission_mode(args)
    api_config = _resolve_api_config(args.api_base)

    # 没有可用 API key 时无法调用模型，直接提示配置方式并退出。
    if not api_config[1]:
        print_error(
            "API key is required.\n"
            "  Set APIKEY (+ optional API) in .env for generic config,\n"
            "  or use ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL,\n"
            "  or use OPENAI_API_KEY / OPENAI_BASE_URL."
        )
        sys.exit(1)

    # 创建主 Agent。OpenAI-compatible 和 Anthropic 原生接口使用不同的 base URL 参数名传入。
    agent = Agent(
        **resolve_agent_options(
            api_config=api_config,
            model=args.model,
            input_price=args.input_price,
            output_price=args.output_price,
            context_window=args.context_window,
        ),
        permission_mode=permission_mode,
        thinking=args.thinking,
        max_cost_usd=args.max_cost,
        max_turns=args.max_turns,
        enable_folding=not args.no_folding,
        auto_compact_threshold=args.fold_threshold,
    )

    # Resume session
    # --resume 会加载最近一次会话，把历史消息恢复到新建的 Agent 中。
    if args.resume or args.resume_id:
        session_id = args.resume_id or get_latest_session_id()
        if session_id:
            session = load_session(session_id)
            if session:
                try:
                    agent.restore_session(session)
                    agent.session_id = session_id
                except ValueError as exc:
                    print_error(str(exc))
                    return
            else:
                print_info("No session found to resume.")
        else:
            print_info("No previous sessions found.")

    # 如果命令行后面带了普通文本参数，就拼成一次性 prompt；否则进入交互式 REPL。
    prompt = " ".join(args.prompt) if args.prompt else None

    if prompt:
        # One-shot mode
        # 一次性模式：执行完用户 prompt 后进程结束。
        try:
            asyncio.run(run_one_shot(agent, prompt))
        except Exception as e:
            print_error(str(e))
            sys.exit(1)
    else:
        # Interactive REPL
        # 交互模式：启动循环读取用户输入，直到用户退出。
        asyncio.run(run_repl(agent))


if __name__ == "__main__":
    main()
