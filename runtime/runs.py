"""
Run Manager：run 的生命周期管理。

对应桌面版改造计划的**阶段 4**。

职责单一：**run 的生命周期**。不碰 agent、不碰事件格式。

**run_id 与 thread_id 是两回事**，别混：

| | 含义 | 生命周期 | 生成规则 |
|---|---|---|---|
| `run_id` | 一次执行的标识 | 只在本次执行内有效，结束即注销 | `run-{uuid8}`，每次执行唯一 |
| `thread_id` | 会话标识，记忆的隔离键 | 跨多次执行复用 | 由客户端指定；不指定则一次性 |

一个会话（thread）里可以连续跑多个 run，每个 run 有独立的 run_id 和独立的取消标志，
但共享同一份记忆。这正是桌面 Agent 的常态：用户在同一个对话里连问三件事。

**并发限制：单连接单任务，同一会话跨入口互斥。**
每个连接（owner）同时只允许一个活跃 run。收到 run.start 时若已有活跃 run ——
**拒绝并回一条 `run.error`**（说明「上一轮还在运行」），而不是静默取消旧的。
理由：静默取消会让用户以为自己看到的是新一轮的结果。

同一 thread_id 由进程共享的 ThreadRegistry 保护，覆盖其他连接、同步任务和删除操作。
取消后仍持有会话占用，直到同步工作线程真正退出；不同会话可并发执行。
"""

from __future__ import annotations

# asyncio：起后台任务。
import asyncio

# threading：取消标志要跨线程用。
import threading

# time：记录开始时刻。
import time

# uuid：生成 run_id 与一次性 thread_id。
import uuid
import weakref

# dataclass：RunHandle 的数据结构。
from dataclasses import dataclass, field

# Any：typing 的任意类型标注。
from typing import Any, Callable

# AgentContext：运行时上下文。
from agent.context import AgentContext

# StreamConsumer：真正地执行体（阶段 3）。
from runtime.consumer import StreamConsumer
from runtime.approvals import ApprovalBroker
from runtime.events import BaseEvent
from runtime.concurrency import THREAD_RUNS, RunBusyError, ThreadLease


# ============================================================
# 1. RunHandle
# ============================================================

@dataclass
class RunHandle:
    """一次执行的登记。

    字段：
        run_id：这次执行的唯一标识。
        owner：发起方（第一版是 WebSocket 连接的 id）——
            单连接单任务的限制就按它判断。
        thread_id：会话 id，决定共享哪份记忆。
        instruction：原始指令。
        task_name：任务名，出现在通知标题里。
        cancel_event：取消标志。消费器每轮检查它。
        started_at：开始时刻（秒级浮点）。
        task：跑消费器的 asyncio 任务。
    """

    run_id: str
    owner: str
    thread_id: str
    instruction: str
    task_name: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    started_at: float = 0.0
    task: asyncio.Task | None = None
    approval_broker: ApprovalBroker | None = None
    consumer: StreamConsumer | None = None
    lease: ThreadLease | None = None
    # 只有真实工作线程退出后才允许注销；取消 asyncio 等待器不满足这个条件。
    worker_finished: threading.Event = field(default_factory=threading.Event)


# ============================================================
# 2. Run Manager
# ============================================================

class RunManager:
    """管理所有活跃 run 的生命周期。

    线程安全：
    登记、注销会被 WebSocket 端点（事件循环线程）调用，
    也可能被别处（信号处理、测试）调用，所以用锁保护。
    """

    def __init__(self) -> None:
        # run_id → RunHandle。只放活跃的 run。
        self._runs: dict[str, RunHandle] = {}

        # 保护 _runs。
        self._lock = threading.Lock()
        # 已结束的 run 也可能还有等待入队的事件。弱引用不延长消费者生命周期，
        # 但连接断开时仍能关闭这些残留投递；仅靠 active() 会漏掉它们。
        self._consumers: dict[str, weakref.WeakSet[StreamConsumer]] = {}

    # ---------- 启动 ----------

    def start(
        self,
        *,
        owner: str,
        instruction: str,
        task_name: str,
        thread_id: str | None,
        agent: Any,
        queue: "asyncio.Queue",
        loop: asyncio.AbstractEventLoop,
        ctx: AgentContext,
        on_started: Callable[[RunHandle], None] | None = None,
        on_terminal: Callable[[BaseEvent], None] | None = None,
    ) -> RunHandle:
        """登记并启动一次执行。

        参数：
            owner：发起方标识（第一版是连接 id）。同一 owner 只允许一个活跃 run。
            instruction：给 Agent 的指令。
            task_name：任务名。
            thread_id：会话 id。不指定则生成一次性的（不留痕）。
            agent：编译好的 Agent。
            queue：事件队列。
            loop：事件循环。
            ctx：运行时上下文。
            on_started：可选的运行登记回调，在消费线程启动前调用。
            on_terminal：可选的结束事件回调，用于保存执行结果。

        返回：
            登记好的 RunHandle。

        抛出：
            RunBusyError：同一 owner 或同一 thread_id 已有活跃 run。

        注意：
            任务跑在**工作线程**里（`run_in_executor`）——
            消费器是同步的，不能占住事件循环。
        """
        with self._lock:
            # 单连接单任务：同一 owner 上一次还没跑完就拒绝。
            if any(h.owner == owner for h in self._runs.values()):
                raise RunBusyError("上一轮还在运行")

            run_id = f"run-{uuid.uuid4().hex[:8]}"
            # 不指定 thread_id 就用一次性的 —— 跟 runner.run_task() 的规则一致。
            resolved_thread_id = thread_id or f"task-{uuid.uuid4().hex[:8]}"
            lease = THREAD_RUNS.claim(resolved_thread_id)

            handle = RunHandle(
                run_id=run_id,
                owner=owner,
                thread_id=resolved_thread_id,
                instruction=instruction,
                task_name=task_name,
                cancel_event=threading.Event(),
                started_at=time.time(),
                lease=lease,
            )
            self._runs[run_id] = handle

        if on_started is not None:
            try:
                on_started(handle)
            except Exception:
                # 登记失败时不启动任务，也不能在活跃表中留下孤儿 run。
                handle.worker_finished.set()
                self.finish(run_id)
                raise

        consumer = StreamConsumer(
            agent=agent,
            run_id=run_id,
            queue=queue,
            loop=loop,
            cancel_event=handle.cancel_event,
            on_terminal=on_terminal,
        )
        broker = ApprovalBroker(
            run_id=run_id,
            emit=consumer.emit_event,
            cancel_event=handle.cancel_event,
        )
        handle.approval_broker = broker
        handle.consumer = consumer
        with self._lock:
            self._consumers.setdefault(owner, weakref.WeakSet()).add(consumer)
        ctx.approval_handler = broker.request
        ctx.cancel_event = handle.cancel_event

        # executor 执行同步消费器；仅工作线程的 finally 能释放会话。
        # 消费器结束时自己会发结束事件（run.end / run.cancelled / run.error），
        # 这里的任务只是等它跑完。
        def work() -> None:
            try:
                consumer.run(instruction, ctx, resolved_thread_id)
            finally:
                handle.worker_finished.set()
                self.finish(run_id)

        async def wait_for_worker(worker) -> None:
            # 外层 task 的取消不能取消 executor future，更不能提前释放会话占用。
            await asyncio.shield(worker)

        try:
            worker = loop.run_in_executor(None, work)
        except Exception:
            handle.worker_finished.set()
            self.finish(run_id)
            raise
        task = loop.create_task(wait_for_worker(worker))
        handle.task = task
        task.add_done_callback(
            lambda done: self.cancel(run_id, disconnect=True) if done.cancelled() else None
        )

        return handle

    # ---------- 查询 ----------

    def get(self, run_id: str) -> RunHandle | None:
        """按 run_id 查活跃的 run。查不到返回 None。"""
        with self._lock:
            return self._runs.get(run_id)

    def active(self) -> list[RunHandle]:
        """列出所有活跃的 run。"""
        with self._lock:
            return list(self._runs.values())

    def active_for(self, owner: str) -> list[RunHandle]:
        """列出某个 owner 的活跃 run。"""
        with self._lock:
            return [h for h in self._runs.values() if h.owner == owner]

    # ---------- 取消与注销 ----------

    def cancel(self, run_id: str, *, disconnect: bool = False) -> bool:
        """请求取消一次执行。

        返回：
            True 表示命中了活跃的 run；False 表示没有这个 run 或它已经结束了。

        注意：
            这是**协作式取消** —— 流循环和每次模型 / 工具尝试前后都会检查。
            正在跑的同步调用等待返回或自身超时，已经发生的副作用不会回滚。
            取消信号最终体现为一条 `run.cancelled` 事件。
            disconnect=True 表示客户端已断开，同时清理队列投递与背压等待。
        """
        handle = self.get(run_id)
        if handle is None:
            return False
        handle.cancel_event.set()
        if disconnect and handle.consumer is not None:
            handle.consumer.close()
        if handle.approval_broker is not None:
            handle.approval_broker.cancel_pending()
        return True

    def respond_approval(self, run_id: str, owner: str, call_id: str, approved: bool) -> bool:
        """仅允许该 run 的所属连接响应仍待处理的授权请求。"""
        handle = self.get(run_id)
        if handle is None or handle.owner != owner or handle.approval_broker is None:
            return False
        return handle.approval_broker.respond(call_id, approved)

    def disconnect(self, owner: str) -> None:
        """关闭连接的所有消费者，包括 run 已结束但事件尚未发完的情况。"""
        for handle in self.active_for(owner):
            self.cancel(handle.run_id, disconnect=True)
        with self._lock:
            consumers = list(self._consumers.pop(owner, ()))
        for consumer in consumers:
            consumer.close()

    def finish(self, run_id: str) -> None:
        """注销已退出的 run；仍执行中的任务不会提前释放会话占用。"""
        with self._lock:
            handle = self._runs.get(run_id)
            if handle is None or not handle.worker_finished.is_set():
                return
            handle = self._runs.pop(run_id, None)
            # 与移除活跃登记处于同一临界区，防止新任务见到半释放状态。
            if handle.lease is not None:
                handle.lease.release()
        if handle.approval_broker is not None:
            handle.approval_broker.cancel_pending()
