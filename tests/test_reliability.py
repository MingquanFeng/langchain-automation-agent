"""运行可靠性回归：真实网络超时、取消传播、跨入口互斥和终态持久化。

所有网络请求仅发往测试创建的 loopback HTTP 服务，模型推理使用假模型。
用 Event 明确控制竞态顺序，避免靠固定 sleep 猜测任务执行到了哪一步。
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import multiprocessing
import sqlite3
import threading

import pytest
from langchain.tools import tool
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.config import ModelConfig
from agent.context import AgentContext, TaskCancelledError
from agent.graph import build_agent
from fakes import FakeModel
from runtime import runner
from runtime.consumer import StreamConsumer
from runtime.runs import RunBusyError, RunManager
from runtime.storage import RuntimeStore


def _hold_sqlite_write_lock(database_path: str, locked, release) -> None:
    """在独立进程中持有写锁，模拟另一实例正在提交运行记录。"""
    connection = sqlite3.connect(database_path, timeout=1)
    try:
        connection.execute("BEGIN IMMEDIATE")
        locked.set()
        assert release.wait(5)
        connection.commit()
    finally:
        connection.close()


# ============================================================
# 1. 模型传输配置与实际超时
# ============================================================

@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="LLM_REQUEST_TIMEOUT"):
        ModelConfig(request_timeout=timeout)


def test_model_request_times_out_without_sdk_retries(monkeypatch):
    """无响应的本地 HTTP 服务触发真实 SDK 超时，且请求不会被 SDK 再发。"""
    from openai import APITimeoutError

    # 开发机可能设置了全局代理；本测试只允许请求自己的 loopback 服务。
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    requests = []
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(self.path)
            release.wait(3)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        model = ModelConfig(
            provider="deepseek", model="deepseek-chat", api_key="test-only",
            base_url=f"http://127.0.0.1:{server.server_port}/v1", request_timeout=0.3,
        ).build()
        # 先验证目的地，防止适配器忽略地址参数后意外访问供应商服务器。
        assert str(model.root_client.base_url) == f"http://127.0.0.1:{server.server_port}/v1/"
        with pytest.raises(APITimeoutError):
            model.invoke("test")
        assert requests == ["/v1/chat/completions"]
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join(2)


# ============================================================
# 2. 取消不能触发下一次模型 / 工具调用
# ============================================================

@pytest.mark.parametrize("fail", [False, True])
def test_cancel_during_model_prevents_tool_and_retry(fail):
    """阻塞模型在取消后返回工具调用或异常，都只产生取消终态。"""
    entered, release, cancel = threading.Event(), threading.Event(), threading.Event()
    calls, tool_calls, recorded = [], [], []

    class BlockingModel(FakeModel):
        def _stream(self, *args, **kwargs):
            calls.append("model")
            entered.set()
            assert release.wait(3)
            if fail:
                raise TimeoutError("late network timeout")
            yield from super()._stream(*args, **kwargs)

    @tool
    def list_files() -> str:
        """测试工具是否实际执行。"""
        tool_calls.append("tool")
        return "不应执行"

    model = BlockingModel(messages=iter([AIMessage(content="", tool_calls=[
        {"name": "list_files", "args": {}, "id": "c1", "type": "tool_call"},
    ])]))
    agent = build_agent(model=model, tools=[list_files], checkpointer=InMemorySaver())

    async def scenario():
        queue = asyncio.Queue()
        consumer = StreamConsumer(
            agent, run_id="cancel-model", queue=queue, loop=asyncio.get_running_loop(),
            cancel_event=cancel, on_terminal=recorded.append,
        )
        task = asyncio.create_task(asyncio.to_thread(consumer.run, "测试", AgentContext(), "cancel-model"))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            cancel.set()
        finally:
            release.set()
        await asyncio.wait_for(task, 3)
        assert [event.type for event in recorded] == ["run.cancelled"]
        assert calls == ["model"]
        assert tool_calls == []

    asyncio.run(scenario())


def test_cancel_during_tool_does_not_retry_and_persists_cancelled(monkeypatch, tmp_path):
    """同步入口中的工具在取消后报错，不再重试，并保存 cancelled。"""
    cancel, calls = threading.Event(), []

    @tool
    def list_files() -> str:
        """模拟执行途中被取消的工具。"""
        calls.append("tool")
        cancel.set()
        raise OSError("late tool error")

    model = FakeModel(messages=iter([AIMessage(content="", tool_calls=[
        {"name": "list_files", "args": {}, "id": "c1", "type": "tool_call"},
    ])]))
    agent = build_agent(model=model, tools=[list_files], checkpointer=InMemorySaver())
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    monkeypatch.setattr(runner, "get_agent", lambda: agent)
    monkeypatch.setattr(runner, "get_runtime_store", lambda: store)
    with pytest.raises(TaskCancelledError):
        runner.run_task("测试", thread_id="cancel-tool", cancel_event=cancel)
    assert calls == ["tool"]
    assert store.list_runs()[0]["status"] == "cancelled"


# ============================================================
# 3. 同步入口占用与初始化锁
# ============================================================

def test_sync_run_blocks_websocket_until_exit(monkeypatch, tmp_path):
    """同步任务先占用会话时，WebSocket 和第二个同步任务都必须被拒绝。"""
    entered, release = threading.Event(), threading.Event()

    class Agent:
        def invoke(self, *args, **kwargs):
            entered.set()
            assert release.wait(3)
            return {"messages": [AIMessage(content="完成")]}

    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    monkeypatch.setattr(runner, "get_agent", lambda: Agent())
    monkeypatch.setattr(runner, "get_runtime_store", lambda: store)

    async def scenario():
        task = asyncio.create_task(asyncio.to_thread(runner.run_task, "测试", thread_id="sync-lock"))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            with pytest.raises(RunBusyError):
                runner.run_task("重复", thread_id="sync-lock")
            with pytest.raises(RunBusyError):
                RunManager().start(
                    owner="ws", instruction="重复", task_name="t", thread_id="sync-lock",
                    agent=None, queue=asyncio.Queue(), loop=asyncio.get_running_loop(),
                    ctx=AgentContext(),
                )
            assert len(store.list_runs()) == 1
        finally:
            release.set()
        await asyncio.wait_for(task, 3)
        assert store.list_runs()[0]["status"] == "succeeded"

    asyncio.run(scenario())


def test_agent_initialization_is_singleton_under_concurrency(monkeypatch):
    """并发首次调用只创建一个模型与 Checkpointer 实例。"""
    entered, release = threading.Event(), threading.Event()
    built = []

    def build():
        built.append(object())
        entered.set()
        assert release.wait(2)
        return built[-1]

    monkeypatch.setattr(runner, "_AGENT", None)
    monkeypatch.setattr(runner, "build_agent", build)
    with ThreadPoolExecutor(max_workers=4) as pool:
        first = pool.submit(runner.get_agent)
        try:
            assert entered.wait(1)
            others = [pool.submit(runner.get_agent) for _ in range(3)]
        finally:
            release.set()
        assert all(f.result() is first.result() for f in others)
    assert len(built) == 1


def test_parallel_tasks_keep_sqlite_checkpoints_and_runs_separate(monkeypatch, tmp_path):
    """不同会话并行使用同一个 SQLite saver，消息和运行记录不能串线。"""
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langchain_core.messages import HumanMessage

    connection = sqlite3.connect(tmp_path / "memory.sqlite", check_same_thread=False)
    try:
        saver = SqliteSaver(connection)
        model = FakeModel(messages=iter([AIMessage(content="完成") for _ in range(8)]))
        agent = build_agent(model=model, checkpointer=saver)
        store = RuntimeStore(tmp_path / "runtime.sqlite3")
        monkeypatch.setattr(runner, "get_agent", lambda: agent)
        monkeypatch.setattr(runner, "get_runtime_store", lambda: store)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(runner.run_task, f"指令-{i}", thread_id=f"parallel-{i}") for i in range(8)]
            results = [future.result(timeout=5) for future in futures]
        assert len({result["run_id"] for result in results}) == 8
        records = store.list_runs()
        assert len(records) == 8
        assert all(record["status"] == "succeeded" for record in records)
        for i in range(8):
            state = agent.get_state({"configurable": {"thread_id": f"parallel-{i}"}})
            instructions = [m.content for m in state.values["messages"] if isinstance(m, HumanMessage)]
            assert instructions == [f"指令-{i}"]
    finally:
        connection.close()


# ============================================================
# 4. 跨进程 SQLite 锁竞争与磁盘故障
# ============================================================

def test_runtime_store_waits_for_a_cross_process_sqlite_write_lock(tmp_path):
    """另一进程短暂持有写锁时，运行记录写入等待后成功而不是报 locked。"""
    database = tmp_path / "runtime.sqlite3"
    store = RuntimeStore(database)
    store.initialize()

    # spawn 避免 macOS fork 继承 pytest 运行时状态，也更接近两个独立桌面进程。
    context = multiprocessing.get_context("spawn")
    locked = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_sqlite_write_lock,
        args=(str(database), locked, release),
    )
    holder.start()
    try:
        assert locked.wait(3), "持锁子进程未能取得 SQLite 写锁"
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                store.start_run,
                run_id="run-cross-process-lock",
                thread_id="thread-cross-process-lock",
                task_name="跨进程锁竞争",
                trigger="test",
                instruction="等待另一进程提交",
            )
            # 锁仍未释放时，调用应处于 SQLite 的 busy 等待，而非立即失败。
            with pytest.raises(TimeoutError):
                future.result(timeout=0.2)
            release.set()
            future.result(timeout=3)
    finally:
        release.set()
        holder.join(5)
        if holder.is_alive():
            holder.terminate()
            holder.join(2)

    assert holder.exitcode == 0
    assert store.list_runs(limit=1)[0]["run_id"] == "run-cross-process-lock"


def test_sqlite_checkpointer_waits_for_a_cross_process_write_lock(tmp_path):
    """会话 Checkpointer 遇到另一进程的短暂写锁后仍可完成同一会话写入。"""
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.sqlite import SqliteSaver

    database = tmp_path / "memory.sqlite"
    connection = sqlite3.connect(database, timeout=3, check_same_thread=False)
    saver = SqliteSaver(connection)
    # 生产路径在 _build_checkpointer() 中主动调用 setup()；先复现该初始化顺序。
    saver.setup()
    agent = build_agent(
        model=FakeModel(messages=iter([AIMessage(content="检查点写入完成")])),
        checkpointer=saver,
    )

    context = multiprocessing.get_context("spawn")
    locked = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_sqlite_write_lock,
        args=(str(database), locked, release),
    )
    holder.start()
    try:
        assert locked.wait(3), "持锁子进程未能取得 SQLite 写锁"
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                agent.invoke,
                {"messages": [HumanMessage(content="检查跨进程锁")]},
                {"configurable": {"thread_id": "checkpoint-cross-process-lock"}},
                context=AgentContext(),
            )
            with pytest.raises(TimeoutError):
                future.result(timeout=0.2)
            release.set()
            result = future.result(timeout=3)
    finally:
        release.set()
        holder.join(5)
        if holder.is_alive():
            holder.terminate()
            holder.join(2)
        connection.close()

    assert holder.exitcode == 0
    assert result["messages"][-1].content == "检查点写入完成"


def test_runtime_store_rolls_back_when_record_write_hits_disk_error(monkeypatch, tmp_path):
    """磁盘写入异常必须回滚事务，不能留下 status=running 的半条运行记录。"""
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize()
    original_connect = sqlite3.connect

    class FailingConnection:
        def __init__(self, connection):
            object.__setattr__(self, "_connection", connection)

        def __setattr__(self, name, value):
            # RuntimeStore 会设置 row_factory；代理必须把它传给真实连接。
            setattr(self._connection, name, value)

        def execute(self, statement, parameters=()):
            if "INSERT INTO agent_runs" in statement:
                raise sqlite3.OperationalError("disk I/O error")
            return self._connection.execute(statement, parameters)

        def __getattr__(self, name):
            return getattr(self._connection, name)

    monkeypatch.setattr(
        "runtime.storage.sqlite3.connect",
        lambda *args, **kwargs: FailingConnection(original_connect(*args, **kwargs)),
    )

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        store.start_run(
            run_id="run-disk-error",
            thread_id="thread-disk-error",
            task_name="磁盘故障",
            trigger="test",
            instruction="模拟运行记录写入失败",
        )

    assert RuntimeStore(store.path).list_runs() == []
