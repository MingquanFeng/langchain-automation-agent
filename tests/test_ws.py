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
from starlette.websockets import WebSocketDisconnect

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


def _install_edit_agent(monkeypatch, tmp_path, *, call_id="edit-1"):
    """让 WebSocket run 编辑临时工作区文件，并把备份也隔离到临时目录。"""
    from agent.graph import build_agent
    from agent.tools import edits as edits_mod
    from agent.tools import files as files_mod
    from fakes import FakeModel
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver

    workspace = tmp_path / "edit-workspace"
    backups = tmp_path / "edit-backups"
    workspace.mkdir()
    target = workspace / "settings.txt"
    target.write_text("mode=old\n", encoding="utf-8")
    monkeypatch.setattr(files_mod, "SANDBOX", workspace.resolve())
    monkeypatch.setattr(edits_mod, "BACKUP_DIR", backups.resolve())
    tool_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "edit_file",
            "args": {
                "path": "settings.txt",
                "old_text": "mode=old",
                "new_text": "mode=new",
            },
            "id": call_id,
            "type": "tool_call",
        }],
    )
    agent = build_agent(
        model=FakeModel(messages=iter([tool_call, AIMessage(content="完成")])),
        checkpointer=InMemorySaver(),
    )
    monkeypatch.setattr(server_mod, "get_agent", lambda: agent)
    return target, backups


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


# ---------- 长期记忆管理 API ----------

def test_memory_api_supports_explicit_crud(client):
    """管理入口可以查看、新建、修改和删除一条明确的长期记忆。"""
    created_response = client.post("/memories", json={
        "scope": "project",
        "memory_type": "preference",
        "content": "默认使用简体中文",
    })
    assert created_response.status_code == 201
    created = created_response.json()
    assert created["id"].startswith("mem_")
    assert created["content"] == "默认使用简体中文"

    listed = client.get("/memories?scope=project").json()
    assert listed["items"] == [created]

    updated_response = client.put(f"/memories/{created['id']}", json={
        "scope": "project",
        "memory_type": "fact",
        "content": "当前项目使用 SQLite",
    })
    assert updated_response.status_code == 200
    updated = updated_response.json()
    assert updated["memory_type"] == "fact"
    assert updated["content"] == "当前项目使用 SQLite"

    deleted = client.delete(
        f"/memories/{created['id']}?scope=project"
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert client.get("/memories?scope=project").json()["items"] == []


def test_memory_api_rejects_invalid_or_missing_records(client):
    """非法类别、空内容、错误 ID 和不存在的更新都有明确状态码。"""
    invalid_kind = client.post("/memories", json={
        "scope": "project",
        "memory_type": "conversation",
        "content": "不要保存整段对话",
    })
    assert invalid_kind.status_code == 422

    empty_content = client.post("/memories", json={
        "scope": "project",
        "memory_type": "fact",
        "content": "   ",
    })
    assert empty_content.status_code == 422

    invalid_id = client.delete("/memories/not-a-memory?scope=project")
    assert invalid_id.status_code == 422

    missing_id = "mem_00000000000000000000000000000000"
    missing = client.put(f"/memories/{missing_id}", json={
        "scope": "project",
        "memory_type": "fact",
        "content": "不存在",
    })
    assert missing.status_code == 404


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


def test_edit_approval_exposes_diff_then_writes_and_backs_up(client, monkeypatch, tmp_path):
    """编辑审批只传安全 diff；批准后才写入，并保留可恢复备份。"""
    target, backups = _install_edit_agent(monkeypatch, tmp_path)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "run.start", "instruction": "修改设置"})
        approval = _drain_until(ws, {"approval.request"})[-1]
        assert approval["name"] == "edit_file"
        assert approval["args"]["path"] == "settings.txt"
        assert "-mode=old" in approval["args"]["diff"]
        assert "+mode=new" in approval["args"]["diff"]
        assert "old_text" not in approval["args"]
        assert "new_text" not in approval["args"]
        assert target.read_text(encoding="utf-8") == "mode=old\n"
        assert not backups.exists()

        ws.send_json({
            "type": "approval.respond",
            "run_id": approval["run_id"],
            "call_id": approval["call_id"],
            "approved": True,
        })
        events = _drain_until(ws, {"run.end", "run.cancelled", "run.error"})

    assert events[-1]["type"] == "run.end"
    assert target.read_text(encoding="utf-8") == "mode=new\n"
    assert len(list(backups.glob("*.json"))) == 1


def test_only_the_run_owner_can_approve_a_tool_call(client, monkeypatch):
    """其他 WebSocket 连接不能替 run 所属连接批准写操作。"""
    target = _install_write_agent(monkeypatch)
    try:
        with client.websocket_connect("/ws") as owner, client.websocket_connect("/ws") as other:
            owner.send_json({"type": "run.start", "instruction": "写入文件"})
            approval = _drain_until(owner, {"approval.request"})[-1]

            # 他人知道 run_id / call_id 也不够，服务端还要核对 WebSocket owner。
            other.send_json({
                "type": "approval.respond",
                "run_id": approval["run_id"],
                "call_id": approval["call_id"],
                "approved": True,
            })
            error = other.receive_json()
            assert error["type"] == "run.error"
            assert error["recoverable"] is True
            assert not target.exists()

            owner.send_json({
                "type": "approval.respond",
                "run_id": approval["run_id"],
                "call_id": approval["call_id"],
                "approved": True,
            })
            _drain_until(owner, {"run.end", "run.cancelled"})

        assert target.read_text() == "approved"
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
        assert any(event["type"] == "run.progress" for event in events)
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


# ---------- 桌面托管连接与逐次审批 ----------


@pytest.fixture
def desktop_access(monkeypatch):
    """为当前测试启用桌面托管模式，使用可控的测试凭证。"""
    access = server_mod.DesktopAccess("a" * 64, "b" * 64)
    monkeypatch.setattr(app.state, "desktop_access", access)
    return access


def test_desktop_http_requires_token_but_allows_health_and_preflight(client, desktop_access):
    """HTTP 越权请求在路由执行之前被拒绝，WebView 预检可以通过。"""
    origin = "tauri://localhost"
    assert client.get("/health").json() == {
        "status": "ok", "scheduler": False, "readiness": desktop_access.readiness,
    }
    for method, path in (
        ("GET", "/runs"),
        ("GET", "/threads"),
        ("GET", "/memories"),
        ("POST", "/memories"),
        ("DELETE", "/runs"),
        ("DELETE", "/threads"),
    ):
        response = client.request(method, path, headers={"Origin": origin})
        assert response.status_code == 401
        assert response.headers["access-control-allow-origin"] == origin
    assert client.get("/runs", headers={"Authorization": "Bearer wrong"}).status_code == 401
    preflight = client.options("/runs", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    })
    assert preflight.status_code == 200

    headers = {"Authorization": f"Bearer {desktop_access.token}"}
    response = client.get("/runs", headers=headers)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert client.delete("/tasks/example", headers=headers).status_code == 403


def test_desktop_websocket_requires_token_and_checks_origin(client, desktop_access):
    """握手缺凭证、凭证错误或网页来源不符时，均在接收命令前拒绝。"""
    protocol = server_mod.DESKTOP_WS_PROTOCOL
    credentials = f"bearer.{desktop_access.token}"
    for protocols, origin in (
        ([protocol], "tauri://localhost"),
        ([protocol, "bearer.wrong"], "tauri://localhost"),
        ([protocol, credentials], "https://example.test"),
    ):
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect("/ws", subprotocols=protocols, headers={"Origin": origin}):
                pytest.fail("拒绝的握手不应进入 WebSocket 会话")
        assert error.value.code == 1008


def test_desktop_websocket_validates_commands_and_requires_approval(client, desktop_access, monkeypatch):
    """托管连接认证成功后，客户端仍无法授予整轮写权限。"""
    target = _install_write_agent(monkeypatch)
    protocol = server_mod.DESKTOP_WS_PROTOCOL
    try:
        with client.websocket_connect("/ws", subprotocols=[protocol, f"bearer.{desktop_access.token}"]) as ws:
            assert ws.accepted_subprotocol == protocol
            for invalid_field in ({"instruction": 12}, {"thread_id": 123}, {"allow_write": "false"}, {"allow_write": True}):
                ws.send_json({"type": "run.start", "instruction": "写入文件", **invalid_field})
                assert ws.receive_json()["type"] == "run.error"
                assert not target.exists()

            ws.send_json({"type": "run.start", "instruction": "写入文件"})
            events = _drain_until(ws, {"approval.request", "run.error"})
            approval = events[-1]
            assert approval["type"] == "approval.request"
            assert not target.exists()
            ws.send_json({"type": "approval.respond", "run_id": approval["run_id"], "call_id": approval["call_id"], "approved": False})
            assert _drain_until(ws, {"run.end", "run.error"})[-1]["type"] == "run.end"
            assert not target.exists()
    finally:
        target.unlink(missing_ok=True)


# ---------- 连接生命周期 ----------


def test_two_connections_cannot_run_or_delete_same_thread(client, monkeypatch):
    """真实 WebSocket 连接共享互斥，重复命令不改变旧任务的状态。"""
    target = _install_write_agent(monkeypatch)
    thread_id = f"shared-{uuid.uuid4().hex}"
    try:
        with client.websocket_connect("/ws") as owner, client.websocket_connect("/ws") as other:
            owner.send_json({"type": "run.start", "instruction": "写文件", "thread_id": thread_id})
            approval = _drain_until(owner, {"approval.request"})[-1]
            other.send_json({"type": "run.start", "instruction": "重复", "thread_id": thread_id})
            error = other.receive_json()
            assert error["type"] == "run.error"
            assert "该会话" in error["message"]
            assert client.delete("/threads", params={"thread_id": thread_id}).status_code == 409
            owner.send_json({"type": "run.start", "instruction": "重复"})
            assert owner.receive_json()["recoverable"] is True
            owner.send_text("invalid json")
            assert owner.receive_json()["recoverable"] is True
            owner.send_json({"type": "cancel"})
            events = _drain_until(owner, {"run.cancelled", "run.error", "run.end"})
            assert events[-1]["type"] == "run.cancelled"
            records = client.get("/runs").json()
            record = next(r for r in records if r["run_id"] == approval["run_id"])
            assert record["status"] == "cancelled"
            assert not target.exists()
    finally:
        target.unlink(missing_ok=True)


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
