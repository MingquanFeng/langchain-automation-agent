"""
WebSocket 事件流端点的测试。

主要职责：
1. 守住协议 —— 命令的形状、事件的顺序。
2. 守住「异常路径必须闭合」—— 客户端断开、命令非法、run 被占用，
   每一种都要有一条结束事件或明确的错误，客户端不能一直转圈。
3. 守住「单连接单任务」—— 第二次 run.start 要被拒绝，
   而不是静默取消上一轮。

注意：
用 FastAPI 的 TestClient 跑真实的 WebSocket 握手，
不是 mock —— 端点的价值全在「连接生命周期」上，mock 掉就测不到东西了。
"""

from __future__ import annotations

# 标准库：sys / pathlib。
import sys
import time
import uuid
from pathlib import Path

# 把项目根目录与 tests/ 插到 sys.path 最前面。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# pytest：测试框架与 fixture。
import pytest

# fastapi.testclient：跑真实的 WebSocket 握手。
from fastapi.testclient import TestClient

# 项目内部。
import runtime.server as server_mod
from runtime.server import app


# ---------- fixtures ----------


@pytest.fixture(autouse=True)
def _no_scheduler(monkeypatch):
    """测试期间不要真启动调度器。

    调度器会到点触发真实的定时任务（打真实 API），
    测试里既慢又不确定。
    """
    monkeypatch.setattr(server_mod, "start_scheduler", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "stop_scheduler", lambda *a, **k: None)


@pytest.fixture
def fake_agent(monkeypatch):
    """让端点用假模型。

    端点内部调 `get_agent()` 拿进程内的 Agent 单例 ——
    不替换的话测试会**真的调模型 API**（花钱、还不确定、模型可能一直说话）。

    端点模块里的 `get_agent` 是名字绑定，
    所以要 patch `runtime.server.get_agent` 而不是 `runtime.runner.get_agent`。
    """
    from agent.graph import build_agent
    from fakes import FakeModel
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver

    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "list_files", "args": {}, "id": "c1", "type": "tool_call"}],
    )
    agent = build_agent(
        model=FakeModel(messages=iter([tool_call, AIMessage(content="有 3 个文件")])),
        checkpointer=InMemorySaver(),
    )
    monkeypatch.setattr(server_mod, "get_agent", lambda: agent)
    return agent


@pytest.fixture
def client(fake_agent):
    """一个 TestClient（已注入假模型）。"""
    with TestClient(app) as c:
        yield c


def _drain_until(ws, stop_types, limit=30):
    """收事件直到遇到指定类型之一。

    返回收到的事件列表。
    """
    events = []
    for _ in range(limit):
        data = ws.receive_json()
        events.append(data)
        if data.get("type") in stop_types:
            return events
    raise AssertionError(f"没等到 {stop_types}，收到的是 {[e.get('type') for e in events]}")


def _install_write_agent(monkeypatch, *, call_id="write-1"):
    """让 WebSocket run 请求写入一个唯一的沙箱文件，并返回目标路径。"""
    from agent.graph import build_agent
    from agent.tools.files import SANDBOX
    from fakes import FakeModel
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver

    relative = f"approval-test-{uuid.uuid4().hex}.txt"
    target = SANDBOX / relative
    tool_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "write_file",
            "args": {"path": relative, "content": "approved"},
            "id": call_id,
            "type": "tool_call",
        }],
    )
    agent = build_agent(
        model=FakeModel(messages=iter([tool_call, AIMessage(content="完成")])),
        checkpointer=InMemorySaver(),
    )
    monkeypatch.setattr(server_mod, "get_agent", lambda: agent)
    return target


# ---------- 正常流程 ----------


def test_run_start_produces_events(client):
    """run.start 之后应该依次收到 run.start 与 run.end。"""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "run.start", "instruction": "用一句话介绍你自己"})
        events = _drain_until(ws, {"run.end", "run.cancelled", "run.error"})

    types = [e["type"] for e in events]
    assert types[0] == "run.start"
    assert "model.text" in types
    assert types[-1] == "run.end"


def test_every_event_has_the_common_fields(client):
    """每条事件都要带 version / run_id / timestamp / type。"""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "run.start", "instruction": "用一句话介绍你自己"})
        events = _drain_until(ws, {"run.end", "run.cancelled", "run.error"})

    for event in events:
        assert event["version"] == 1
        assert event["run_id"]
        assert isinstance(event["timestamp"], float)
        assert event["type"]

    # 同一批事件共享一个 run_id。
    assert len({e["run_id"] for e in events}) == 1


def test_run_start_echoes_the_run_metadata(client):
    """run.start 事件要带回 run_id / task_name / thread_id / instruction。"""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({
            "type": "run.start",
            "instruction": "列出数据目录下的文件",
            "task_name": "列文件",
            "thread_id": "my-chat",
        })
        events = _drain_until(ws, {"run.end", "run.cancelled", "run.error"})

    start = events[0]
    assert start["type"] == "run.start"
    assert start["task_name"] == "列文件"
    assert start["thread_id"] == "my-chat"
    assert start["instruction"] == "列出数据目录下的文件"


# ---------- 取消 ----------


def test_cancel_produces_cancelled_event(client):
    """取消后应该收到 run.cancelled（不是 run.end）。"""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "run.start", "instruction": "用一句话介绍你自己"})
        events = _drain_until(ws, {"run.start"})
        ws.send_json({"type": "cancel"})
        events = _drain_until(ws, {"run.cancelled", "run.end", "run.error"})

    types = [e["type"] for e in events]
    assert "run.cancelled" in types
    assert "run.end" not in types


# ---------- 拒绝与错误 ----------


def test_second_run_on_same_connection_is_rejected(client):
    """同一连接已有活跃 run 时，第二次 run.start 要被拒绝。"""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "run.start", "instruction": "用一句话介绍你自己"})
        _drain_until(ws, {"run.start"})
        ws.send_json({"type": "run.start", "instruction": "再来一次"})
        events = _drain_until(ws, {"run.error", "run.end", "run.cancelled"})

    errors = [e for e in events if e["type"] == "run.error"]
    assert errors, f"应该收到 run.error，实际 {[e['type'] for e in events]}"
    assert "上一轮还在运行" in errors[0]["message"]


def test_unknown_command_gets_run_error(client):
    """未知命令要回一条 run.error，不能静默忽略。"""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "noop"})
        events = _drain_until(ws, {"run.error"})

    assert "未知命令" in events[0]["message"]


def test_invalid_json_gets_run_error(client):
    """不是合法 JSON 要回一条 run.error。"""
    with client.websocket_connect("/ws") as ws:
        ws.send_text("这不是 JSON")
        events = _drain_until(ws, {"run.error"})

    assert "JSON" in events[0]["message"]


def test_missing_instruction_gets_run_error(client):
    """缺 instruction 要回一条 run.error。"""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "run.start"})
        events = _drain_until(ws, {"run.error"})

    assert "instruction" in events[0]["message"]


def test_cancel_without_active_run_gets_run_error(client):
    """没有活跃 run 时取消，要回一条 run.error。"""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "cancel"})
        events = _drain_until(ws, {"run.error"})

    assert "没有正在运行" in events[0]["message"]


def test_write_tool_waits_for_matching_approval(client, monkeypatch):
    """写工具在匹配的 approval.respond 前不能执行，批准后才继续。"""
    # 用随机文件名隔离并行或重复测试，避免覆盖数据目录里的现有文件。
    target = _install_write_agent(monkeypatch)

    try:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "run.start", "instruction": "写入文件"})
            approval = _drain_until(ws, {"approval.request"})[-1]
            assert approval["call_id"] == "write-1"
            assert approval["name"] == "write_file"
            assert not target.exists()
            ws.send_json({"type": "approval.respond", "run_id": approval["run_id"], "call_id": "wrong", "approved": True})
            error = _drain_until(ws, {"run.error"})[-1]
            assert error["recoverable"] is True
            assert not target.exists()
            ws.send_json({"type": "approval.respond", "run_id": approval["run_id"], "call_id": "write-1", "approved": True})
            # run.error 可能是重复应答的可恢复错误；继续读取直到整轮任务结束。
            events = _drain_until(ws, {"run.end", "run.cancelled"})
        assert target.read_text() == "approved"
        assert any(event["type"] == "tool.start" and event["call_id"] == "write-1" for event in events)
    finally:
        target.unlink(missing_ok=True)


def test_rejecting_approval_prevents_write_and_repeated_response(client, monkeypatch):
    """拒绝时不执行写入；同一 call_id 的第二次应答返回可恢复错误。"""
    target = _install_write_agent(monkeypatch)
    try:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "run.start", "instruction": "写入文件"})
            approval = _drain_until(ws, {"approval.request"})[-1]
            ws.send_json({
                "type": "approval.respond",
                "run_id": approval["run_id"],
                "call_id": approval["call_id"],
                "approved": False,
            })
            ws.send_json({
                "type": "approval.respond",
                "run_id": approval["run_id"],
                "call_id": approval["call_id"],
                "approved": True,
            })
            # run.error 可能是重复应答的可恢复错误；继续读取直到整轮任务结束。
            events = _drain_until(ws, {"run.end", "run.cancelled"})

        assert not target.exists()
        assert not any(event["type"] == "tool.start" for event in events)
        assert any(event["type"] == "tool.error" for event in events)
        assert any(event["type"] == "run.error" and event["recoverable"] for event in events)
    finally:
        target.unlink(missing_ok=True)


def test_cancel_during_approval_prevents_write(client, monkeypatch):
    """等待授权时仍可处理 cancel，并且取消后写工具保持未执行。"""
    target = _install_write_agent(monkeypatch)
    try:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "run.start", "instruction": "写入文件"})
            approval = _drain_until(ws, {"approval.request"})[-1]
            ws.send_json({"type": "cancel"})
            events = _drain_until(ws, {"run.cancelled", "run.end", "run.error"})

        assert events[-1]["type"] == "run.cancelled"
        assert not target.exists()
        assert not any(event["type"] == "tool.start" for event in events)
    finally:
        target.unlink(missing_ok=True)


def test_disconnect_during_approval_cleans_pending_request(client, monkeypatch):
    """WebSocket 断开会取消等待中的授权，并清理 run 与 pending 记录。"""
    target = _install_write_agent(monkeypatch)
    manager = app.state.run_manager
    try:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "run.start", "instruction": "写入文件"})
            approval = _drain_until(ws, {"approval.request"})[-1]
            handle = manager.get(approval["run_id"])
            assert handle is not None
            assert "write-1" in handle.approval_broker._pending

        # 退出 websocket 上下文会触发服务端断连清理；等待工作线程结束后检查状态。
        for _ in range(50):
            if manager.get(approval["run_id"]) is None:
                break
            time.sleep(0.05)

        assert manager.get(approval["run_id"]) is None
        assert handle.approval_broker._pending == {}
        assert not target.exists()
    finally:
        target.unlink(missing_ok=True)


def test_malformed_approval_response_is_recoverable(client):
    """approved 不是布尔值时返回可恢复错误，不中断 WebSocket。"""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "approval.respond", "run_id": "r", "call_id": "c", "approved": "yes"})
        event = ws.receive_json()
    assert event["type"] == "run.error"
    assert event["recoverable"] is True


# ---------- 连接生命周期 ----------


def test_disconnect_cancels_the_run(client):
    """连接断开时，这个连接上的 run 要被取消并注销。"""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "run.start", "instruction": "用一句话介绍你自己"})
        _drain_until(ws, {"run.start"})
        manager = app.state.run_manager
        assert len(manager.active()) == 1

    # 退出 with 块 = 断开连接。
    # 断开后的清理是异步的，给它一点时间。
    import time

    for _ in range(50):
        if not manager.active():
            break
        time.sleep(0.05)

    assert manager.active() == [], "连接断开后 run 应该被取消并注销"
