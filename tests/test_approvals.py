"""逐次授权 broker 的并发、超时与清理测试。"""

from __future__ import annotations

# queue / threading：工作线程发出请求事件，测试线程等待事件并返回决定。
import queue
import threading

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

    # RunManager 先置取消标志，再唤醒 broker 中的等待者。
    cancel_event.set()
    broker.cancel_pending()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert result["approved"] is False
    assert broker._pending == {}


def test_concurrent_calls_keep_decisions_separate():
    """两个并发 call_id 各自等待，允许一个不会误放行另一个。"""
    broker, _, emitted = _broker(timeout=2)
    first_result, first_thread = _start_request(broker, "call-a")
    second_result, second_thread = _start_request(broker, "call-b")

    calls = {emitted.get(timeout=1).call_id, emitted.get(timeout=1).call_id}
    assert calls == {"call-a", "call-b"}

    assert broker.respond("call-a", True) is True
    assert broker.respond("call-b", False) is True
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert first_result["approved"] is True
    assert second_result["approved"] is False
    assert broker._pending == {}


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
