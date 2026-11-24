"""逐次授权 broker 的并发、超时与清理测试。"""

from __future__ import annotations

# queue / threading：工作线程发出请求事件，测试线程等待事件并返回决定。
import queue
import threading

import pytest

from runtime.approvals import ApprovalBroker


def _broker(*, timeout: float = 1.0, emitted: queue.Queue | None = None):
    """创建一个不依赖 WebSocket 的 broker，并可收集发出的授权事件。"""
    cancel_event = threading.Event()
    event_queue = emitted if emitted is not None else queue.Queue()
    broker = ApprovalBroker(
        run_id="run-test",
        emit=event_queue.put,
        cancel_event=cancel_event,
        timeout=timeout,
    )
    return broker, cancel_event, event_queue


def _start_request(broker: ApprovalBroker, call_id: str):
    """在独立工作线程启动阻塞式授权请求，返回结果容器与线程。"""
    result: dict[str, bool] = {}

    def request() -> None:
        result["approved"] = broker.request("write_file", call_id, {"path": call_id})

    thread = threading.Thread(target=request)
    thread.start()
    return result, thread


def test_reject_and_duplicate_response_do_not_approve_request():
    """用户拒绝后返回 False；未知或重复应答不能改变该决定。"""
    broker, _, emitted = _broker()
    result, thread = _start_request(broker, "call-1")

    event = emitted.get(timeout=1)
    assert event.call_id == "call-1"
    assert broker.respond("unknown", True) is False
    assert broker.respond("call-1", False) is True
    assert broker.respond("call-1", True) is False

    thread.join(timeout=1)
    assert not thread.is_alive()
    assert result["approved"] is False
    assert broker._pending == {}


def test_expired_request_is_cleaned_and_late_response_is_rejected():
    """超时按拒绝处理，过期 call_id 不能再被应答放行。"""
    broker, _, emitted = _broker(timeout=0.01)

    assert broker.request("write_file", "call-timeout", {}) is False
    event = emitted.get(timeout=1)
    assert event.call_id == "call-timeout"
    assert broker.respond("call-timeout", True) is False
    assert broker._pending == {}


def test_cancel_wakes_waiting_request_and_clears_it():
    """run 取消时唤醒正在等待的请求，且工具授权结果为拒绝。"""
    broker, cancel_event, emitted = _broker(timeout=5)
    result, thread = _start_request(broker, "call-cancel")
    emitted.get(timeout=1)
    queued_result, queued_thread = _start_request(broker, "call-queued")

    # RunManager 先置取消标志，再唤醒 broker 中的等待者。
    cancel_event.set()
    broker.cancel_pending()
    thread.join(timeout=1)
    queued_thread.join(timeout=1)

    assert not thread.is_alive()
    assert result["approved"] is False
    assert queued_result["approved"] is False
    assert emitted.empty()
    assert broker._pending == {}


def test_concurrent_calls_keep_decisions_separate():
    """并行工具按登记顺序逐项审批；提前应答队尾不能放行。"""
    broker, _, emitted = _broker(timeout=2)
    first_result, first_thread = _start_request(broker, "call-a")
    assert emitted.get(timeout=1).call_id == "call-a"
    second_result, second_thread = _start_request(broker, "call-b")

    with pytest.raises(queue.Empty):
        emitted.get(timeout=0.05)
    assert broker.respond("call-b", True) is False
    assert broker.respond("call-a", True) is True
    assert emitted.get(timeout=1).call_id == "call-b"
    assert broker.respond("call-b", False) is True
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert first_result["approved"] is True
    assert second_result["approved"] is False
    assert broker._pending == {}


def test_queued_approval_gets_its_own_timeout():
    """第一项耗完整个审批时限，第二项展示后仍能获批。"""
    broker, _, emitted = _broker(timeout=0.25)
    first, first_thread = _start_request(broker, "slow")
    assert emitted.get(timeout=1).call_id == "slow"
    second, second_thread = _start_request(broker, "next")

    expired = emitted.get(timeout=2)
    assert (expired.type, expired.call_id) == ("approval.expired", "slow")
    assert emitted.get(timeout=1).call_id == "next"
    assert broker.respond("slow", True) is False
    assert broker.respond("next", True) is True
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)
    assert first["approved"] is False
    assert second["approved"] is True


def test_event_emission_failure_cleans_pending_request():
    """授权事件无法发布时立即拒绝，并移除已登记的请求。"""
    cancel_event = threading.Event()

    def fail_emit(_event) -> None:
        raise RuntimeError("event loop closed")

    broker = ApprovalBroker(
        run_id="run-test",
        emit=fail_emit,
        cancel_event=cancel_event,
    )

    assert broker.request("write_file", "call-fail", {}) is False
    assert broker._pending == {}
