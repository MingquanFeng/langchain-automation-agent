
"""
Agent 调用封装。调度器和 Python 调用方都走这里，保证行为一致。

对应课件第 09 章（courseware/langChain/尚硅谷-09-上下文与记忆.pdf）：记忆与状态持久化。

主要职责：
1. 维护进程内唯一的 Agent 单例。
2. 把「指令 + 上下文 + 会话 ID」组装成一次 invoke。
3. 把执行结果整理成统一的字典返回。

设计要点：
调度器和 Python 调用方都从这个同步入口走，WebSocket 使用流消费器。
两种执行方式共用会话占用表与取消中间件，
这样触发出来的行为（上下文构造、thread_id 规则、
写权限默认值）完全一致，不会出现「某个入口忘了传 context」的情况。
"""

from __future__ import annotations

# uuid：标准库，用来生成随机的会话 ID。
import uuid
import logging
import time
import threading

# Any：typing 里的任意类型标注。
from typing import Any

# HumanMessage：LangChain 消息类型，代表用户输入。
from langchain_core.messages import HumanMessage

# RUNTIME：应用运行配置，这里主要取通知渠道。
from agent.config import RUNTIME

# AgentContext：单次调用的运行时上下文结构。
# Trigger：触发来源的字面量类型（schedule/api/cli/lab）。
from agent.context import AgentContext, Trigger, TaskCancelledError, check_cancelled

# build_agent：Agent 装配入口，见 agent/graph.py。
from agent.graph import build_agent
from runtime.storage import get_runtime_store
from runtime.concurrency import THREAD_RUNS

# 复用同一个 agent 实例，保留 checkpointer 里的记忆
#
# 为什么必须复用：
# build_agent() 内部会创建 Checkpointer，
# SQLite 后端下它是连着数据库的。
# 每建一个 Agent 就多开一条连接、多建一套中间件，
# 而且不同 Agent 的 thread_id 记忆是隔离的，
# 换 Agent 等于把之前的会话记忆丢掉。
#
# 所以这里用模块级变量做惰性单例：
# 第一次调用 get_agent() 时才真正创建。
_AGENT = None
_AGENT_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


# ============================================================
# 1. 获取 Agent 单例
# ============================================================

def get_agent():
    """
    获取进程内唯一的 Agent 实例。

    返回：
        build_agent() 创建的 Agent。

    注意：
    惰性创建：模块导入时不建 Agent，
    避免仅仅 import runtime 就连上数据库、初始化模型客户端。

    线程安全：
    用锁串行化首次初始化，避免并发建立多套模型客户端和数据库连接。
    """
    global _AGENT
    if _AGENT is None:
        with _AGENT_LOCK:
            if _AGENT is None:
                _AGENT = build_agent()
    return _AGENT


# ============================================================
# 2. 执行一次任务
# ============================================================

def run_task(
    instruction: str,
    task_name: str | None = None,
    trigger: Trigger = "cli",
    allow_write: bool = False,
    thread_id: str | None = None,
    *,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """执行一次任务。

    参数：
        instruction：
            给 Agent 的自然语言指令。

        task_name：
            任务名，用于日志和通知标题。
            不传时截取指令前 30 个字符兜底。

        trigger：
            触发来源，见 agent/context.py 的 Trigger 定义。
            调度器传 schedule，Python 调用方一般用默认的 cli。

        allow_write：
            是否允许写操作。
            默认 False，即默认只读。
            写权限最终由 TaskGateMiddleware 强制检查。

        thread_id：
            会话 ID。
            相同则共享记忆（对应课件 09 章短期记忆，见 courseware/langChain/尚硅谷-09-上下文与记忆.pdf）。
            不传时生成一个一次性的随机 ID。
            这只影响会话记忆的复用；每次执行仍会进入本地运行历史。

        cancel_event：可选的协作式取消信号；取消后抛 TaskCancelledError。
            已经开始的同步 I/O 等待自身返回，期间会话仍被占用。

    抛出：
        RunBusyError：同进程中其他入口正在使用该会话，本次不会登记运行记录。

    返回：
        包含 run_id、任务名、thread_id、最终输出、消息条数的字典。

    注意：
    thread_id 的命名规则决定了记忆的隔离边界：
        cron-{任务名}：定时任务跨天保留记忆。
        task-{随机}：一次性任务，不复用会话记忆；运行记录仍会持久化。
    """
    # --------------------------------------------------------
    # 2.1 构造运行时上下文
    # --------------------------------------------------------

    # notify_channel 从全局配置取，
    # 保证所有入口的通知走同一个渠道。
    ctx = AgentContext(
        task_name=task_name or instruction[:30],
        trigger=trigger,
        allow_write=allow_write,
        notify_channel=RUNTIME.notify_channel,
        cancel_event=cancel_event,
    )

    # --------------------------------------------------------
    # 2.2 构造 config
    # --------------------------------------------------------

    # thread_id 必须放在 config["configurable"] 里，
    # 这是 Checkpointer 识别会话的依据。
    #
    # 不传时用 task-{8位随机} 生成一次性的会话 ID，
    # 保证不同任务之间不会串记忆。
    config = {"configurable": {"thread_id": thread_id or f"task-{uuid.uuid4().hex[:8]}"}}
    resolved_thread_id = config["configurable"]["thread_id"]
    run_id = f"run-{uuid.uuid4().hex}"
    with THREAD_RUNS.claim(resolved_thread_id):
        return _execute_task(instruction, ctx, config, run_id, resolved_thread_id)


def _execute_task(instruction, ctx, config, run_id, resolved_thread_id) -> dict[str, Any]:
    """在调用方持有会话占用期间登记、执行并保存终态。"""
    started_at = time.time()
    ctx.run_id = run_id
    store = get_runtime_store()
    recorded = False
    try:
        store.start_run(
            run_id=run_id,
            thread_id=resolved_thread_id,
            task_name=ctx.task_name,
            trigger=ctx.trigger,
            instruction=instruction,
            started_at=started_at,
        )
        recorded = True
    except Exception:
        # 本地历史库异常不应阻止 Agent 正常执行；服务端会记录诊断日志。
        logger.exception("无法登记 Agent 运行记录：%s", run_id)

    # --------------------------------------------------------
    # 2.3 调用 Agent
    # --------------------------------------------------------

    # context 是 invoke 的独立参数，不在 config 里
    #
    # 这是最容易踩的坑之一：
    #
    # agent.invoke(msgs, {"configurable": {...}}, context=ctx)   # ✅
    # agent.invoke(msgs, {"configurable": {..., "context": ctx}})  # ❌
    #
    # 写错的话 invoke 不报错，
    # 但 runtime.context 会是 None，
    # 中间件里的 _ctx() 兜底会走到默认值，
    # 表现为「任务名和写权限都丢了」这种很难查的问题。
    try:
        check_cancelled(ctx)
        agent = get_agent()
        check_cancelled(ctx)
        result = agent.invoke(
            {"messages": [HumanMessage(content=instruction)]}, config, context=ctx
        )
        check_cancelled(ctx)
        # Agent 跑完后最后一条是 AIMessage（最终回答）。
        output = result["messages"][-1].content
        message_count = len(result["messages"])
    except Exception as exc:
        cancelled = isinstance(exc, TaskCancelledError) or (
            ctx.cancel_event is not None and ctx.cancel_event.is_set()
        )
        if recorded:
            try:
                store.finish_run(
                    run_id,
                    status="cancelled" if cancelled else "failed",
                    error="用户取消" if cancelled else f"{type(exc).__name__}: {exc}",
                    duration_ms=int((time.time() - started_at) * 1000),
                )
            except Exception:
                logger.exception("无法保存 Agent 运行错误：%s", run_id)
        if cancelled and not isinstance(exc, TaskCancelledError):
            raise TaskCancelledError("任务已请求取消") from exc
        raise

    # --------------------------------------------------------
    # 2.4 整理返回值
    # --------------------------------------------------------

    if recorded:
        try:
            store.finish_run(
                run_id,
                status="succeeded",
                output=output if isinstance(output, str) else str(output),
                duration_ms=int((time.time() - started_at) * 1000),
                message_count=message_count,
            )
        except Exception:
            logger.exception("无法保存 Agent 运行结果：%s", run_id)

    return {
        "task_name": ctx.task_name,
        "run_id": run_id,
        "thread_id": resolved_thread_id,
        "output": output,
        # 消息条数：一条用户消息 + 每轮的工具消息 + 最终回答。
        # 可以用来粗略判断这次任务来回了多少轮。
        "messages": message_count,
    }
