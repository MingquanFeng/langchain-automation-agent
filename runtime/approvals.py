"""WebSocket 工具调用的单次授权协调器。

工具的同步 handler 跑在工作线程里，但用户决定由 asyncio WebSocket 事件循环接收，
两边不能直接用普通函数返回值传递结果。本模块用条件变量协调排队与应答：

1. 工具中间件调用 `ApprovalBroker.request()`，按到达顺序登记请求。轮到队首时
   才发出 `approval.request` 并开始计时，用户每次只需要核对一项。
2. WebSocket 收到客户端的 `approval.respond` 后，RunManager 按 `run_id`、
   连接 owner 和 `call_id` 找到对应请求，再调用 `respond()`。
3. `respond()` 保存决定并唤醒工作线程；只有明确批准时 `request()` 才返回 True，
   工具中间件才会继续执行实际工具。

授权等待必须有时限，也必须响应 run 取消。无论批准、拒绝、超时、取消还是事件发送失败，
请求最终都要从待确认表移除，避免残留状态影响后续工具调用。
"""

from __future__ import annotations

# threading：条件变量在工作线程间交接队首，同时接收事件循环送来的决定。
import threading

# time：用单调时钟计算超时，避免系统时间调整导致等待时间变长或变短。
import time

# dataclass：保存每个调用独立的决定和截止时间。
from dataclasses import dataclass

# Any / Callable：描述工具参数及线程安全事件发送函数的类型。
from typing import Any, Callable

# 审批超时由 broker 直接通知界面，不必等并行工具执行完才收起卡片。
from runtime.events import ApprovalExpired, ApprovalRequest, BaseEvent

# TOOL_RISK_LEVELS：为授权事件附上该工具登记的风险等级。
from agent.tools.risk import TOOL_RISK_LEVELS


# 用户未作决定时，最多等待一分钟；超时后安全地按拒绝处理。
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 60.0


@dataclass
class _PendingApproval:
    """一个尚未完成的工具授权请求在内存中的同步状态。

    approved：用户决定；None 表示还没有决定，True / False 分别表示允许 / 拒绝。
    expires_at：轮到此调用时写入单调时钟的截止时间，排队期间保持为空。
    """

    approved: bool | None = None
    expires_at: float | None = None


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

        # dict 保持登记顺序。决定、取消和队首出队都唤醒等待者重新检查自己的状态。
        self._turn = threading.Condition()

    def request(self, name: str, call_id: str, args: dict[str, Any]) -> bool:
        """发出授权请求并等待用户对该 call_id 作出决定。

        返回 True 只表示用户明确批准；拒绝、超时、取消、重复 call_id
        或事件发送失败都返回 False。
        """
        # 无 id 就无法与客户端应答配对；已取消的 run 也不能再发起等待。
        if not call_id or self.cancel_event.is_set():
            return False

        pending = _PendingApproval()
        with self._turn:
            # 同一个 run 内 call_id 必须唯一；重复 id 无法安全地区分两次决定。
            if call_id in self._pending:
                return False
            self._pending[call_id] = pending

        try:
            with self._turn:
                while next(iter(self._pending)) != call_id:
                    if self.cancel_event.is_set() or pending.approved is not None:
                        return False
                    self._turn.wait(0.1)
                if self.cancel_event.is_set() or pending.approved is not None:
                    return False
                deadline = time.monotonic() + max(0.0, self.timeout)
                pending.expires_at = deadline

            # 事件里包含工具名、参数和风险等级，供用户判断具体操作。
            self.emit(ApprovalRequest(
                run_id=self.run_id,
                call_id=call_id,
                name=name,
                args=args,
                risk_level=TOOL_RISK_LEVELS[name],
            ))
            with self._turn:
                while pending.approved is None and not self.cancel_event.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._turn.wait(min(0.1, remaining))
                if pending.approved is not None or self.cancel_event.is_set():
                    return pending.approved is True and not self.cancel_event.is_set()
            self.emit(ApprovalExpired(run_id=self.run_id, call_id=call_id))
            return False
        except Exception:
            # 事件未能送达时按拒绝处理，finally 仍会交接到下一项。
            return False
        finally:
            with self._turn:
                del self._pending[call_id]
                self._turn.notify_all()

    def respond(self, call_id: str, approved: bool) -> bool:
        """完成一个仍在等待的请求；未知或已处理的 call_id 返回 False。"""
        # 严格限定 JSON 布尔值；不把 0/1 或其他真值转换成授权决定。
        if type(approved) is not bool:
            return False
        with self._turn:
            if self.cancel_event.is_set():
                return False
            # 只接受当前仍在等待的请求；过期、未知和重复应答都不改变状态。
            pending = self._pending.get(call_id)
            if pending is None or pending.approved is not None or pending.expires_at is None:
                return False

            # request 线程可能还没来得及清理刚过期的条目；在这里再次核对截止时间，
            # 防止迟到的允许应答抢在 finally 清理前进入。
            if time.monotonic() >= pending.expires_at:
                return False

            # 先写入决定，再通知工作线程，确保被唤醒时状态已经完整。
            pending.approved = approved
            self._turn.notify_all()
            return True

    def cancel_pending(self) -> None:
        """唤醒所有等待中的工具请求，并以拒绝结束它们。"""
        with self._turn:
            # 取消和正常应答共用同一信号；将结果设为拒绝可确保工具不会误放行。
            for pending in self._pending.values():
                if pending.approved is None:
                    pending.approved = False
            self._turn.notify_all()
