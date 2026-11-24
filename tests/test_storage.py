"""运行时 SQLite 持久化的单元与集成测试。

本模块验证：
1. 默认任务只在新数据库中种入一次，动态改动跨 RuntimeStore 实例保留。
2. Agent run 的成功、失败、取消和进程中断状态能正确写入并分页读取。
3. 同步 runner 与 WebSocket 服务都使用运行时存储记录实际执行结果。
4. 工作台历史分页、消息恢复和运行记录/会话删除语义保持一致。

所有数据库写入都使用 pytest 临时目录；WebSocket 集成测试使用假模型，不调用真实模型 API。
"""

from __future__ import annotations

# sys / pathlib：把项目源码目录和测试替身目录加入模块搜索路径。
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TESTS_DIR))

# pytest：提供临时目录、monkeypatch 和异常断言。
import pytest

# TestClient：通过真实 HTTP 与 WebSocket 生命周期验证服务端接线。
from fastapi.testclient import TestClient

# AIMessage：构造不依赖网络的模型结果。
from langchain_core.messages import AIMessage, HumanMessage

# InMemorySaver：集成测试中的会话状态只放内存，不触碰真实记忆库。
from langgraph.checkpoint.memory import InMemorySaver

# FakeModel：支持 agent.stream() 的假模型，适用于 WebSocket 流式路径。
from fakes import FakeModel

# RuntimeStore：本模块要验证的 SQLite 运行时数据访问层。
from runtime.storage import RuntimeStore


# ============================================================
# 1. 任务注册表持久化
# ============================================================

def _default_tasks() -> list[dict]:
    """返回一组隔离测试用的默认任务定义。"""
    return [
        {
            "name": "每日汇总",
            "cron": "0 9 * * *",
            "instruction": "整理每日摘要",
            "allow_write": False,
            "enabled": True,
        },
        {
            "name": "目录巡检",
            "cron": "0 18 * * *",
            "instruction": "检查工作目录",
            "allow_write": True,
            "enabled": False,
        },
    ]


def test_default_tasks_are_seeded_once_and_survive_store_recreation(tmp_path):
    """默认项首次写入；之后更新和删除都不会被下次启动重新覆盖。"""
    database = tmp_path / "runtime.sqlite3"
    defaults = _default_tasks()
    first_store = RuntimeStore(database)

    assert first_store.load_or_seed_tasks(defaults) == defaults

    # 同名任务执行 upsert；删除默认任务后保留初始化标记，不能再次被种回去。
    updated = {**defaults[0], "cron": "30 9 * * *", "instruction": "更新后的摘要"}
    first_store.save_task(updated)
    first_store.delete_task("目录巡检")

    restored = RuntimeStore(database).load_or_seed_tasks(defaults)
    assert restored == [updated]


def test_task_module_restores_dynamic_changes_from_sqlite(tmp_path, monkeypatch):
    """tasks.py 启动恢复、动态新增与删除都会同步到 SQLite。"""
    import runtime.tasks as tasks_mod

    # configure_persistence 会替换模块级注册表；测试结束后恢复全局状态。
    monkeypatch.setattr(tasks_mod, "_registry", tasks_mod._registry.copy())
    monkeypatch.setattr(tasks_mod, "_store", None)

    database = tmp_path / "runtime.sqlite3"
    tasks_mod.configure_persistence(RuntimeStore(database))
    tasks_mod.add_task(tasks_mod.ScheduledTask(
        name="临时巡检",
        cron="15 10 * * *",
        instruction="检查临时工作区",
        allow_write=False,
    ))
    assert tasks_mod.remove_task("每日待办汇总") is True

    # 新实例模拟一次服务重启：动态项保留，已删除默认项不会复活。
    tasks_mod.configure_persistence(RuntimeStore(database))
    restored = {task.name: task for task in tasks_mod.all_tasks()}
    assert set(restored) == {"数据目录巡检", "临时巡检"}
    assert restored["临时巡检"].cron == "15 10 * * *"


# ============================================================
# 2. Agent run 状态、结果与分页
# ============================================================

def _start_record(
    store: RuntimeStore,
    run_id: str,
    *,
    started_at: float,
    trigger: str = "api",
) -> None:
    """用固定时间戳创建一条测试运行，便于稳定断言分页顺序。"""
    store.start_run(
        run_id=run_id,
        thread_id=f"thread-{run_id}",
        task_name=f"任务-{run_id}",
        trigger=trigger,
        instruction=f"处理 {run_id}",
        started_at=started_at,
    )


def test_run_records_store_terminal_fields_and_page_newest_first(tmp_path):
    """终态字段持久化，分页按开始时间倒序，时间输出含 UTC 时区。"""
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    _start_record(store, "run-old", started_at=100.0, trigger="schedule")
    store.finish_run(
        "run-old",
        status="succeeded",
        output="任务完成",
        duration_ms=321,
        message_count=4,
        finished_at=101.0,
    )
    _start_record(store, "run-new", started_at=200.0)
    store.finish_run(
        "run-new",
        status="failed",
        error="RuntimeError: provider unavailable",
        duration_ms=50,
        finished_at=201.0,
    )

    newest = store.list_runs(limit=1, offset=0)
    older = store.list_runs(limit=1, offset=1)

    assert newest[0]["run_id"] == "run-new"
    assert newest[0]["status"] == "failed"
    assert newest[0]["error"] == "RuntimeError: provider unavailable"
    assert newest[0]["trigger"] == "api"
    assert older[0]["run_id"] == "run-old"
    assert older[0]["status"] == "succeeded"
    assert older[0]["output"] == "任务完成"
    assert older[0]["trigger"] == "schedule"
    assert older[0]["duration_ms"] == 321
    assert older[0]["message_count"] == 4
    assert older[0]["started_at"].endswith("+00:00")
    assert older[0]["finished_at"].endswith("+00:00")


def test_service_restart_marks_only_running_records_interrupted(tmp_path):
    """服务启动恢复只关闭未完成 run，不覆盖已成功的历史记录。"""
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    _start_record(store, "run-running", started_at=100.0)
    _start_record(store, "run-finished", started_at=200.0)
    store.finish_run("run-finished", status="succeeded", output="done")

    assert store.mark_interrupted_runs() == 1
    rows = {row["run_id"]: row for row in store.list_runs()}
    assert rows["run-running"]["status"] == "interrupted"
    assert rows["run-running"]["error"] == "服务重启时任务仍在运行"
    assert rows["run-running"]["finished_at"] is not None
    assert rows["run-finished"]["status"] == "succeeded"


def test_first_terminal_state_wins_over_a_late_terminal_event(tmp_path):
    """取消、错误等迟到终态不能覆盖已经写入的最终结果。"""
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    _start_record(store, "run-final-once", started_at=100.0)

    assert store.finish_run(
        "run-final-once", status="cancelled", error="用户取消", finished_at=101.0
    ) is True
    assert store.finish_run(
        "run-final-once", status="failed", error="迟到的工作线程错误", finished_at=102.0
    ) is False

    row = store.list_runs(limit=1)[0]
    assert row["status"] == "cancelled"
    assert row["error"] == "用户取消"


def test_run_records_redact_credentials_but_keep_diagnostic_context(tmp_path):
    """历史记录不可保存凭据，但仍保留错误类型和可操作的上下文。"""
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.start_run(
        run_id="run-redaction",
        thread_id="thread-redaction",
        task_name="脱敏测试",
        trigger="test",
        instruction="检查 api_key=sk-secret-value 是否可用",
    )
    store.finish_run(
        "run-redaction",
        status="failed",
        output="Authorization: Bearer token-secret-value",
        error="ProviderError: password=hunter2 请求失败",
    )

    row = store.list_runs(limit=1)[0]
    serialized = "\n".join(str(row[key]) for key in ("instruction", "output", "error"))
    assert "sk-secret-value" not in serialized
    assert "token-secret-value" not in serialized
    assert "hunter2" not in serialized
    assert "ProviderError" in row["error"]
    assert "[已隐藏]" in serialized


def test_database_file_is_private_on_posix(tmp_path):
    """POSIX 环境下运行记录数据库权限限制为当前用户读写。"""
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize()

    if store.path.stat().st_mode & 0o777 != 0o600:
        pytest.skip("当前文件系统不支持 POSIX 权限模式")
    assert store.path.stat().st_mode & 0o777 == 0o600


def test_command_audit_records_only_the_first_terminal_state(tmp_path):
    """开发命令审计关联 run/call，且终态不能被迟到更新覆盖。"""
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    execution_id = store.start_command_execution(
        run_id="run-command", call_id="call-command", category="python_test",
        argv='["python", "-m", "pytest"]', cwd=".", timeout_seconds=60,
        started_at=1.0,
    )
    store.finish_command_execution(
        execution_id, status="succeeded", exit_code=0, output_summary="ok", finished_at=2.0,
    )
    store.finish_command_execution(
        execution_id, status="failed", exit_code=1, output_summary="late", finished_at=3.0,
    )

    with store._connect() as connection:
        row = connection.execute(
            "SELECT run_id, call_id, status, exit_code, output_summary, finished_at "
            "FROM command_executions WHERE id = ?", (execution_id,),
        ).fetchone()
    assert dict(row) == {
        "run_id": "run-command", "call_id": "call-command", "status": "succeeded",
        "exit_code": 0, "output_summary": "ok", "finished_at": 2.0,
    }


# ============================================================
# 3. 同步 runner 持久化
# ============================================================

def test_run_task_persists_success_and_returns_run_id(tmp_path, monkeypatch):
    """同步入口将最终回答写入历史，并向调用方返回可查询的 run_id。"""
    import runtime.runner as runner_mod

    store = RuntimeStore(tmp_path / "runtime.sqlite3")

    class FakeAgent:
        def invoke(self, *_args, **_kwargs):
            return {"messages": [AIMessage(content="同步任务完成")]}

    monkeypatch.setattr(runner_mod, "get_runtime_store", lambda: store)
    monkeypatch.setattr(runner_mod, "get_agent", lambda: FakeAgent())

    result = runner_mod.run_task(
        instruction="运行同步持久化检查",
        task_name="同步检查",
        trigger="schedule",
        thread_id="cron-storage-check",
    )

    record = store.list_runs(limit=1)[0]
    assert result["run_id"] == record["run_id"]
    assert result["output"] == "同步任务完成"
    assert record["thread_id"] == "cron-storage-check"
    assert record["trigger"] == "schedule"
    assert record["status"] == "succeeded"
    assert record["output"] == "同步任务完成"
    assert record["message_count"] == 1


def test_run_task_persists_agent_failure(tmp_path, monkeypatch):
    """Agent 初始化或 invoke 失败时，同步入口记录失败后仍抛出原异常。"""
    import runtime.runner as runner_mod

    store = RuntimeStore(tmp_path / "runtime.sqlite3")

    class FailingAgent:
        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(runner_mod, "get_runtime_store", lambda: store)
    monkeypatch.setattr(runner_mod, "get_agent", lambda: FailingAgent())

    with pytest.raises(RuntimeError, match="provider unavailable"):
        runner_mod.run_task(instruction="运行失败路径检查")

    record = store.list_runs(limit=1)[0]
    assert record["status"] == "failed"
    assert record["error"] == "RuntimeError: provider unavailable"
    assert record["finished_at"] is not None


# ============================================================
# 4. WebSocket run 与 `/runs` 接口持久化
# ============================================================

@pytest.fixture
def isolated_server_store(tmp_path, monkeypatch):
    """把服务 lifespan、任务表和调度器隔离到临时 SQLite 数据库。"""
    import runtime.server as server_mod
    import runtime.tasks as tasks_mod

    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    monkeypatch.setattr(server_mod, "get_runtime_store", lambda: store)
    monkeypatch.setattr(server_mod, "start_scheduler", lambda *args, **kwargs: None)
    monkeypatch.setattr(server_mod, "stop_scheduler", lambda: None)

    # Lifespan 会调用 configure_persistence 重新绑定这两个全局值，测试后恢复原状态。
    monkeypatch.setattr(tasks_mod, "_registry", tasks_mod._registry.copy())
    monkeypatch.setattr(tasks_mod, "_store", tasks_mod._store)
    return store


def _receive_until_terminal(websocket, terminal_types: set[str]) -> list[dict]:
    """接收 WebSocket 事件直到出现终态，防止测试漏掉持久化完成时机。"""
    events = []
    for _ in range(30):
        event = websocket.receive_json()
        events.append(event)
        if event["type"] in terminal_types:
            return events
    raise AssertionError(f"未收到终态事件：{[event['type'] for event in events]}")


def test_websocket_success_is_queryable_from_runs_endpoint(
    isolated_server_store, monkeypatch
):
    """真实 WebSocket 生命周期结束后，`GET /runs` 能读到对应记录。"""
    import runtime.server as server_mod

    # 不读取真实单例；下面显式注入一次性假模型 Agent。
    from agent.graph import build_agent

    test_agent = build_agent(
        model=FakeModel(messages=iter([AIMessage(content="流式任务完成")])),
        checkpointer=InMemorySaver(),
    )
    monkeypatch.setattr(server_mod, "get_agent", lambda: test_agent)

    with TestClient(server_mod.app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({
                "type": "run.start",
                "instruction": "运行 WebSocket 持久化检查",
                "task_name": "流式检查",
                "thread_id": "thread-storage-ws",
            })
            events = _receive_until_terminal(websocket, {"run.end", "run.error"})

        assert events[0]["type"] == "run.start"
        assert events[-1]["type"] == "run.end"
        assert events[-1]["output"] == "流式任务完成"

        response = client.get("/runs?limit=10&offset=0")
        assert response.status_code == 200
        assert client.get("/runs?limit=101").status_code == 422
        records = response.json()

    assert len(records) == 1
    assert records[0]["run_id"] == events[0]["run_id"]
    assert records[0]["thread_id"] == "thread-storage-ws"
    assert records[0]["status"] == "succeeded"
    assert records[0]["output"] == "流式任务完成"


def test_websocket_cancel_is_persisted_as_cancelled(isolated_server_store, monkeypatch):
    """WebSocket 取消终态与持久化状态一致。"""
    import runtime.server as server_mod

    class SlowAgent:
        def stream(self, *_args, **_kwargs):
            # 第一帧让 run 进入消费循环；延迟第二帧给 WebSocket 发出 cancel 的机会。
            yield "values", {"messages": []}
            time.sleep(0.2)
            yield "values", {"messages": []}

    monkeypatch.setattr(server_mod, "get_agent", lambda: SlowAgent())

    with TestClient(server_mod.app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({
                "type": "run.start",
                "instruction": "运行 WebSocket 取消持久化检查",
                "thread_id": "thread-storage-cancel",
            })
            start_events = _receive_until_terminal(websocket, {"run.start", "run.error"})
            assert start_events[-1]["type"] == "run.start"
            websocket.send_json({"type": "cancel"})
            terminal_events = _receive_until_terminal(
                websocket, {"run.cancelled", "run.end", "run.error"}
            )

        assert terminal_events[-1]["type"] == "run.cancelled"
        records = client.get("/runs?limit=10&offset=0").json()

    assert len(records) == 1
    assert records[0]["run_id"] == start_events[0]["run_id"]
    assert records[0]["status"] == "cancelled"


def test_websocket_does_not_start_when_run_record_cannot_be_written(
    isolated_server_store, monkeypatch
):
    """运行记录起始行无法写入时，端点返回错误并清理 run，不调用 Agent。"""
    import runtime.server as server_mod

    class AgentMustNotRun:
        def stream(self, *_args, **_kwargs):
            raise AssertionError("记录写入失败后不应启动 Agent")

    def fail_start(**_kwargs):
        raise OSError("runtime database is unavailable")

    monkeypatch.setattr(server_mod, "get_agent", lambda: AgentMustNotRun())
    monkeypatch.setattr(isolated_server_store, "start_run", fail_start)

    with TestClient(server_mod.app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({
                "type": "run.start",
                "instruction": "运行持久化失败边界检查",
            })
            events = _receive_until_terminal(websocket, {"run.error"})
        assert "无法保存运行记录" in events[-1]["message"]
        assert client.app.state.run_manager.active() == []

    assert isolated_server_store.list_runs() == []


def test_checkpoint_thread_history_lists_sessions_without_run_rows(
    isolated_server_store, monkeypatch
):
    """历史会话直接从 Checkpointer 列出，不依赖新版运行记录表。"""
    import runtime.server as server_mod

    from agent.graph import build_agent

    memory = InMemorySaver()
    test_agent = build_agent(
        model=FakeModel(messages=iter([
            AIMessage(content="第一段回答"),
            AIMessage(content="第二段回答"),
        ])),
        checkpointer=memory,
    )
    for thread_id, question in (
        ("thread-history-old", "第一段历史提问"),
        ("thread-history-new", "第二段历史提问"),
    ):
        test_agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            {"configurable": {"thread_id": thread_id}},
        )
    monkeypatch.setattr(server_mod, "get_agent", lambda: test_agent)

    with TestClient(server_mod.app) as client:
        first_page = client.get("/threads?limit=1&offset=0")
        second_page = client.get("/threads?limit=1&offset=1")

    assert first_page.status_code == 200
    assert second_page.status_code == 200
    first_payload = first_page.json()
    second_payload = second_page.json()
    assert first_payload["has_more"] is True
    assert second_payload["has_more"] is False
    sessions = first_payload["threads"] + second_payload["threads"]
    assert {session["thread_id"] for session in sessions} == {
        "thread-history-old",
        "thread-history-new",
    }
    assert {session["title"] for session in sessions} == {
        "第一段历史提问",
        "第二段历史提问",
    }
    assert {session["preview"] for session in sessions} == {
        "第一段回答",
        "第二段回答",
    }
    assert isolated_server_store.list_runs() == []


def _seed_finished_run(
    store: RuntimeStore,
    run_id: str,
    thread_id: str,
    *,
    started_at: float,
) -> None:
    """为历史删除语义测试创建一条已完成运行记录。"""
    store.start_run(
        run_id=run_id,
        thread_id=thread_id,
        task_name=f"任务-{run_id}",
        trigger="api",
        instruction=f"处理 {run_id}",
        started_at=started_at,
    )
    store.finish_run(
        run_id,
        status="succeeded",
        output=f"完成 {run_id}",
        finished_at=started_at + 1,
    )


def test_desktop_history_follows_workspace_and_preserves_unfiled_threads(
    isolated_server_store, tmp_path, monkeypatch
):
    """两个同名目录分别恢复历史；清空运行记录不会抹掉归属；旧会话续聊时才归档。"""
    import runtime.server as server
    from agent.graph import build_agent

    agent = build_agent(
        model=FakeModel(messages=iter(AIMessage(content="已记录") for _ in range(5))),
        checkpointer=InMemorySaver(),
    )
    agent.invoke({"messages": [HumanMessage(content="旧版对话")]}, {"configurable": {"thread_id": "old"}})
    monkeypatch.setattr(server, "get_agent", lambda: agent)
    access = server.DesktopAccess("a" * 64, "b" * 64)
    monkeypatch.setattr(server.app.state, "desktop_access", access, raising=False)
    project_a, project_b = tmp_path / "甲" / "项目", tmp_path / "乙" / "项目"
    headers = {"Authorization": f"Bearer {access.token}"}
    protocols = [server.DESKTOP_WS_PROTOCOL, f"bearer.{access.token}"]

    with TestClient(server.app, headers=headers) as client:
        def run(instruction, thread_id=None):
            with client.websocket_connect("/ws", subprotocols=protocols) as websocket:
                websocket.send_json({"type": "run.start", "instruction": instruction, "thread_id": thread_id})
                return _receive_until_terminal(websocket, {"run.end", "run.error"})

        def titles(scope="current", **params):
            page = client.get("/threads", params={"scope": scope, **params}).json()
            return [item["title"] for item in page["threads"]]

        monkeypatch.setattr(server, "DATA_DIR", project_a)
        events_a = run("甲的对话")
        assert events_a[-1]["type"] == "run.end"
        thread_a, run_a = events_a[0]["thread_id"], events_a[0]["run_id"]
        assert titles() == ["甲的对话"]
        assert titles("legacy") == ["旧版对话"]

        monkeypatch.setattr(server, "DATA_DIR", project_b)
        assert titles() == []
        assert client.get("/runs").json() == []
        assert client.get("/threads/messages", params={"thread_id": thread_a}).status_code == 404
        assert client.delete("/threads", params={"thread_id": thread_a}).status_code == 404
        assert client.delete(f"/runs/{run_a}").json()["deleted"] is False
        assert run("错误目录续聊", thread_a)[-1]["type"] == "run.error"
        assert run("乙的对话")[-1]["type"] == "run.end"
        assert titles() == ["乙的对话"]
        assert client.delete("/runs").json() == {"deleted": 1}
        assert titles() == ["乙的对话"]

        monkeypatch.setattr(server, "DATA_DIR", project_a)
        assert [row["run_id"] for row in client.get("/runs").json()] == [run_a]
        previous = client.get("/threads/messages", params={"thread_id": thread_a}).json()
        assert previous["messages"][0]["content"] == "甲的对话"
        assert run("继续旧版对话", "old")[-1]["type"] == "run.end"
        assert titles("legacy") == []
        assert titles(limit=1, offset=1) == ["甲的对话"]
        assert client.delete("/threads", params={"thread_id": "old"}).json()["deleted_runs"] == 1
        assert "old" not in RuntimeStore(isolated_server_store.path).thread_workspaces()


def test_thread_messages_restore_and_continue_the_same_thread(
    isolated_server_store, monkeypatch
):
    """消息接口过滤内部提示，恢复后的下一轮沿用原 thread_id 和完整记忆。"""
    import runtime.server as server_mod

    from agent.graph import build_agent

    thread_id = "thread-history-resume"
    test_agent = build_agent(
        model=FakeModel(messages=iter([
            AIMessage(content="第一轮回答"),
            AIMessage(content="第二轮回答"),
        ])),
        checkpointer=InMemorySaver(),
    )
    monkeypatch.setattr(server_mod, "get_agent", lambda: test_agent)

    with TestClient(server_mod.app) as client:
        for instruction, expected_output in (
            ("第一轮问题", "第一轮回答"),
            ("继续这个会话", "第二轮回答"),
        ):
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({
                    "type": "run.start",
                    "instruction": instruction,
                    "thread_id": thread_id,
                })
                events = _receive_until_terminal(websocket, {"run.end", "run.error"})
            assert events[-1]["type"] == "run.end"
            assert events[-1]["output"] == expected_output

        response = client.get("/threads/messages", params={"thread_id": thread_id})
        session = client.get("/threads?limit=10&offset=0").json()["threads"][0]

    assert response.status_code == 200
    payload = response.json()
    assert payload["thread_id"] == thread_id
    assert payload["truncated"] is False
    assert [(item["role"], item["content"]) for item in payload["messages"]] == [
        ("user", "第一轮问题"),
        ("assistant", "第一轮回答"),
        ("user", "继续这个会话"),
        ("assistant", "第二轮回答"),
    ]
    assert session["thread_id"] == thread_id
    assert session["title"] == "第一轮问题"
    assert session["preview"] == "第二轮回答"
    assert session["message_count"] == 4


def test_deleting_runs_and_clearing_history_preserve_checkpoint(
    isolated_server_store, monkeypatch
):
    """删除单条或清空运行记录都不能删除对应的会话记忆。"""
    import runtime.server as server_mod

    from agent.context import AgentContext
    from agent.graph import build_agent

    thread_id = "thread-history-keep"
    test_agent = build_agent(
        model=FakeModel(messages=iter([AIMessage(content="会话记忆仍在")])),
        checkpointer=InMemorySaver(),
    )
    test_agent.invoke(
        {"messages": [HumanMessage(content="需要保留的提问")]},
        {"configurable": {"thread_id": thread_id}},
        context=AgentContext(allow_write=True),
    )
    monkeypatch.setattr(server_mod, "get_agent", lambda: test_agent)
    _seed_finished_run(isolated_server_store, "run-delete-one", thread_id, started_at=10)
    _seed_finished_run(isolated_server_store, "run-clear-rest", thread_id, started_at=20)

    with TestClient(server_mod.app) as client:
        deleted = client.delete("/runs/run-delete-one")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True
        assert [row["run_id"] for row in client.get("/runs").json()] == ["run-clear-rest"]
        assert client.get("/threads/messages", params={"thread_id": thread_id}).json()[
            "message_count"
        ] == 2

        cleared = client.delete("/runs")
        assert cleared.status_code == 200
        assert cleared.json()["deleted"] == 1
        assert client.get("/runs").json() == []
        messages = client.get("/threads/messages", params={"thread_id": thread_id}).json()
        threads = client.get("/threads").json()["threads"]

    assert messages["message_count"] == 2
    assert any(item["thread_id"] == thread_id for item in threads)


def test_running_history_cannot_be_deleted_or_cleared(
    isolated_server_store, monkeypatch
):
    """仍在运行的记录不能单独删除，也不能通过清空历史绕过保护。"""
    import runtime.server as server_mod

    from agent.graph import build_agent

    test_agent = build_agent(
        model=FakeModel(messages=iter([AIMessage(content="不会实际调用")])),
        checkpointer=InMemorySaver(),
    )
    monkeypatch.setattr(server_mod, "get_agent", lambda: test_agent)

    with TestClient(server_mod.app) as client:
        # lifespan 会把启动前遗留的 running 标记为 interrupted，所以必须在服务
        # 启动后创建当前进程内的运行记录，才能验证在线保护语义。
        isolated_server_store.start_run(
            run_id="run-live",
            thread_id="thread-live",
            task_name="当前运行任务",
            trigger="api",
            instruction="不能删除",
            started_at=20,
        )
        assert client.delete("/runs/run-live").status_code == 409
        assert client.delete("/runs").status_code == 409

        isolated_server_store.finish_run("run-live", status="cancelled")
        assert client.delete("/runs/run-live").status_code == 200


def test_deleting_thread_removes_checkpoint_and_only_its_runs(
    isolated_server_store, monkeypatch
):
    """删除整个会话清理其 Checkpointer 和关联记录，但不影响其他会话。"""
    import runtime.server as server_mod

    from agent.context import AgentContext
    from agent.graph import build_agent

    target_thread = "thread-history-delete"
    other_thread = "thread-history-other"
    test_agent = build_agent(
        model=FakeModel(messages=iter([
            AIMessage(content="待删除回答"),
            AIMessage(content="保留回答"),
        ])),
        checkpointer=InMemorySaver(),
    )
    for thread_id, question in (
        (target_thread, "待删除问题"),
        (other_thread, "保留问题"),
    ):
        test_agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            {"configurable": {"thread_id": thread_id}},
            context=AgentContext(allow_write=True),
        )
    monkeypatch.setattr(server_mod, "get_agent", lambda: test_agent)
    _seed_finished_run(isolated_server_store, "run-target-1", target_thread, started_at=10)
    _seed_finished_run(isolated_server_store, "run-target-2", target_thread, started_at=20)
    _seed_finished_run(isolated_server_store, "run-other", other_thread, started_at=30)

    with TestClient(server_mod.app) as client:
        deleted = client.delete("/threads", params={"thread_id": target_thread})
        assert deleted.status_code == 200
        assert deleted.json() == {"thread_id": target_thread, "deleted_runs": 2}

        target_messages = client.get(
            "/threads/messages", params={"thread_id": target_thread}
        ).json()
        other_messages = client.get(
            "/threads/messages", params={"thread_id": other_thread}
        ).json()
        threads = client.get("/threads").json()["threads"]
        runs = client.get("/runs").json()

    assert target_messages["message_count"] == 0
    assert other_messages["message_count"] == 2
    assert {item["thread_id"] for item in threads} == {other_thread}
    assert [row["run_id"] for row in runs] == ["run-other"]

