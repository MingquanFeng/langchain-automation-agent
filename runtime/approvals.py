"""WebSocket 工具调用的单次授权协调器。

工具的同步 handler 跑在工作线程里，但用户决定由 asyncio WebSocket 事件循环接收，
两边不能直接用普通函数返回值传递结果。本模块用 `threading.Event` 搭桥：

1. 工具中间件调用 `ApprovalBroker.request()`，登记待确认请求并发出
   `approval.request` 事件，然后在工作线程等待。
2. WebSocket 收到客户端的 `approval.respond` 后，RunManager 按 `run_id`、
   连接 owner 和 `call_id` 找到对应请求，再调用 `respond()`。
3. `respond()` 保存决定并唤醒工作线程；只有明确批准时 `request()` 才返回 True，
   工具中间件才会继续执行实际工具。

授权等待必须有时限，也必须响应 run 取消。无论批准、拒绝、超时、取消还是事件发送失败，
请求最终都要从待确认表移除，避免残留状态影响后续工具调用。
"""

from __future__ import annotations

# threading：Event 用于跨工作线程与事件循环传递决定，Lock 保护待确认表。
import threading

# time：用单调时钟计算超时，避免系统时间调整导致等待时间变长或变短。
import time

# dataclass / field：定义每个待确认请求，并为每条请求创建独立的线程事件。
from dataclasses import dataclass, field

# Any / Callable：描述工具参数及线程安全事件发送函数的类型。
from typing import Any, Callable

# ApprovalRequest / BaseEvent：构造发给客户端的授权事件，并标注事件发送接口。
from runtime.events import ApprovalRequest, BaseEvent

# TOOL_RISK_LEVELS：为授权事件附上该工具登记的风险等级。
from agent.tools.risk import TOOL_RISK_LEVELS


# 用户未作决定时，最多等待一分钟；超时后安全地按拒绝处理。
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 60.0


@dataclass
class _PendingApproval:
    """一个尚未完成的工具授权请求在内存中的同步状态。

    decided：工作线程等待的信号。收到用户应答或 run 被取消时置位。
    approved：用户决定；None 表示还没有决定，True / False 分别表示允许 / 拒绝。
    """

    # 每个 call_id 都有独立 Event，确保某个请求的决定不会唤醒其他工具调用。
    decided: threading.Event = field(default_factory=threading.Event)

    # 先保存决定，再置位 decided；等待方被唤醒后即可读取完整结果。
    approved: bool | None = None

    # 单调时钟下的过期时间；用于拒绝刚好在清理前到达的迟到应答。
    expires_at: float = 0.0


class ApprovalBroker:
    """在工具工作线程与 WebSocket 事件循环之间协调逐次授权。

    待确认请求按工具的 `call_id` 索引。锁只保护字典和单条请求的决定状态；
    等待用户期间不持锁，因此 WebSocket 仍能登记或处理其他请求。
    """

    def __init__(
        self,
        *,
        run_id: str,
        emit: Callable[[BaseEvent], None],
        cancel_event: threading.Event,
        timeout: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    ) -> None:
        # run_id：授权属于哪一次 Agent 执行，客户端应在应答中原样返回。
        self.run_id = run_id

        # emit：把 BaseEvent 送入该 run 的事件队列；由 StreamConsumer 提供线程安全实现。
        self.emit = emit

        # cancel_event：与 RunHandle 共用；run 取消时请求不应继续等待用户。
        self.cancel_event = cancel_event

        # timeout：本次授权的最长等待秒数，便于测试时传入较短时限。
        self.timeout = timeout

        # call_id → _PendingApproval。只保留尚未完成的请求，不保存历史决定。
        self._pending: dict[str, _PendingApproval] = {}

        # request / respond / cancel_pending 可能从不同线程同时进入，统一保护共享状态。
        self._lock = threading.Lock()

    def request(self, name: str, call_id: str, args: dict[str, Any]) -> bool:
        """发出授权请求并等待用户对该 call_id 作出决定。

        返回 True 只表示用户明确批准；拒绝、超时、取消、重复 call_id
        或事件发送失败都返回 False。
        """
        # 无 id 就无法与客户端应答配对；已取消的 run 也不能再发起等待。
        if not call_id or self.cancel_event.is_set():
            return False

        # 从开始处理请求时就计算截止时间，避免事件发送期间耗时被排除在等待上限外。
        deadline = time.monotonic() + max(0.0, self.timeout)

        # 先登记，再发事件：客户端可能很快应答，先登记可避免应答早于请求入表。
        pending = _PendingApproval(expires_at=deadline)
        with self._lock:
            # 同一个 run 内 call_id 必须唯一；重复 id 无法安全地区分两次决定。
            if call_id in self._pending:
                return False
            self._pending[call_id] = pending

        try:
            # 事件里包含工具名、参数和风险等级，供用户判断具体操作。
            self.emit(ApprovalRequest(
                run_id=self.run_id,
                call_id=call_id,
                name=name,
                args=args,
                risk_level=TOOL_RISK_LEVELS[name],
            ))
        except Exception:
            # 事件无法送达时没有可用的授权界面，因此按拒绝处理并清除登记。
            self._remove_if_current(call_id, pending)
            return False

        try:
            # Event.wait 会阻塞工作线程，不会占住 WebSocket 的 asyncio 事件循环。
            while not self.cancel_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # 没有及时收到决定时拒绝工具调用，避免超时后意外产生副作用。
                    with self._lock:
                        if not pending.decided.is_set():
                            pending.approved = False
                            pending.decided.set()
                    return False
                # 短间隔轮询取消标志，保证取消无需等满整个授权超时。
                if pending.decided.wait(min(0.1, remaining)):
                    # 只有显式 True 才算批准；拒绝或未设置都不能放行工具。
                    return pending.approved is True
            return False
        finally:
            # 所有完成路径都清理请求，未知或过期 call_id 后续无法再次应答。
            self._remove_if_current(call_id, pending)

    def respond(self, call_id: str, approved: bool) -> bool:
        """完成一个仍在等待的请求；未知或已处理的 call_id 返回 False。"""
        # 严格限定 JSON 布尔值；不把 0/1 或其他真值转换成授权决定。
        if type(approved) is not bool:
            return False
        with self._lock:
            # 只接受当前仍在等待的请求；过期、未知和重复应答都不改变状态。
            pending = self._pending.get(call_id)
            if pending is None or pending.decided.is_set():
                return False

            # request 线程可能还没来得及清理刚过期的条目；在这里再次核对截止时间，
            # 防止迟到的允许应答抢在 finally 清理前进入。
            if time.monotonic() >= pending.expires_at:
                pending.approved = False
                pending.decided.set()
                return False

            # 先写入决定，再通知工作线程，确保被唤醒时状态已经完整。
            pending.approved = approved
            pending.decided.set()
            return True

    def cancel_pending(self) -> None:
        """唤醒所有等待中的工具请求，并以拒绝结束它们。"""
        with self._lock:
            # 取消和正常应答共用同一信号；将结果设为拒绝可确保工具不会误放行。
            for pending in self._pending.values():
                if not pending.decided.is_set():
                    pending.approved = False
                    pending.decided.set()

    def _remove_if_current(self, call_id: str, pending: _PendingApproval) -> None:
        """仅移除仍指向本次请求的条目，避免误删同 id 的后续请求。"""
        with self._lock:
            # 对象身份检查让清理操作具备幂等性，也保护可能复用的 call_id。
            if self._pending.get(call_id) is pending:
                del self._pending[call_id]
