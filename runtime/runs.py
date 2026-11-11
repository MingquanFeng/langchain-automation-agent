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

**并发限制：第一版单连接单任务。**
每个连接（owner）同时只允许一个活跃 run。收到 run.start 时若已有活跃 run ——
**拒绝并回一条 `run.error`**（说明「上一轮还在运行」），而不是静默取消旧的。
理由：静默取消会让用户以为自己看到的是新一轮的结果。

将来要做多任务并发，Run Manager 按 run_id 管理已经能承接，
届时再去掉 owner 维度的限制、再加「事件按 run_id 路由」。
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

# dataclass：RunHandle 的数据结构。
from dataclasses import dataclass, field

# Any：typing 的任意类型标注。
from typing import Any

# AgentContext：运行时上下文。
from agent.context import AgentContext

# StreamConsumer：真正地执行体（阶段 3）。
from runtime.consumer import StreamConsumer
from runtime.approvals import ApprovalBroker


# ============================================================
# 1. 异常
# ============================================================

class RunBusyError(RuntimeError):
    """同一个 owner 已经有活跃 run 了。

    调用方（阶段 5 的 WebSocket 端点）应该捕获它，
    然后给客户端回一条 `run.error` 说明「上一轮还在运行」，
    而不是静默取消旧的 run。
    """


# ============================================================
# 2. RunHandle
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


# ============================================================
# 3. Run Manager
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

        返回：
            登记好的 RunHandle。

        抛出：
            RunBusyError：同一 owner 已有活跃 run。

        注意：
            任务跑在**工作线程**里（`asyncio.to_thread`）——
            消费器是同步的，不能占住事件循环。
        """
        with self._lock:
            # 单连接单任务：同一 owner 上一次还没跑完就拒绝。
            if any(h.owner == owner for h in self._runs.values()):
                raise RunBusyError("上一轮还在运行")

            run_id = f"run-{uuid.uuid4().hex[:8]}"
            # 不指定 thread_id 就用一次性的 —— 跟 runner.run_task() 的规则一致。
            resolved_thread_id = thread_id or f"task-{uuid.uuid4().hex[:8]}"

            handle = RunHandle(
                run_id=run_id,
                owner=owner,
                thread_id=resolved_thread_id,
                instruction=instruction,
                task_name=task_name,
                cancel_event=threading.Event(),
                started_at=time.time(),
            )
            self._runs[run_id] = handle

        consumer = StreamConsumer(
            agent=agent,
            run_id=run_id,
            queue=queue,
            loop=loop,
            cancel_event=handle.cancel_event,
        )
        broker = ApprovalBroker(
            run_id=run_id,
            emit=consumer.emit_event,
            cancel_event=handle.cancel_event,
        )
        handle.approval_broker = broker
        ctx.approval_handler = broker.request

        # to_thread 把同步的消费器放到工作线程里跑。
        # 消费器结束时自己会发结束事件（run.end / run.cancelled / run.error），
        # 这里的任务只是等它跑完。
        task = loop.create_task(
            asyncio.to_thread(consumer.run, instruction, ctx, resolved_thread_id)
        )
        handle.task = task

        # 跑完自动注销 —— 幂等，重复调用无害。
        task.add_done_callback(lambda _: self.finish(run_id))

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

    def cancel(self, run_id: str) -> bool:
        """请求取消一次执行。

        返回：
            True 表示命中了活跃的 run；False 表示没有这个 run 或它已经结束了。

        注意：
            这是**协作式取消** —— 置位标志后，消费循环在**下一次迭代**时
            检查到并 break。正在跑的那一步（模型调用或工具执行）**会做完**。
            取消信号最终体现为一条 `run.cancelled` 事件。
        """
        handle = self.get(run_id)
        if handle is None:
            return False
        handle.cancel_event.set()
        if handle.approval_broker is not None:
            handle.approval_broker.cancel_pending()
        return True

    def respond_approval(self, run_id: str, owner: str, call_id: str, approved: bool) -> bool:
        """仅允许该 run 的所属连接响应仍待处理的授权请求。"""
        handle = self.get(run_id)
        if handle is None or handle.owner != owner or handle.approval_broker is None:
            return False
        return handle.approval_broker.respond(call_id, approved)

    def finish(self, run_id: str) -> None:
        """注销一个 run。幂等 —— 重复调用无害。"""
        with self._lock:
            handle = self._runs.pop(run_id, None)
        if handle is not None and handle.approval_broker is not None:
            handle.approval_broker.cancel_pending()
