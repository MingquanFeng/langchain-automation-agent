"""
Run Manager 的测试。全部用假模型，不调真实 API。

主要职责：
1. 守住 run_id 与 thread_id 的分工 —— 混了会导致记忆隔离出错。
2. 守住「单连接单任务」—— 同一 owner 第二次启动要被拒绝，
   而不是静默取消上一轮。
3. 守住取消与注销 —— cancel 要命中、finish 要幂等、跑完要自动注销。
4. 守住「取消最终表现为一条 run.cancelled 事件」——
   客户端据此把界面从「运行中」改到「已取消」。

注意：
Run Manager 依赖 asyncio（它起后台任务），
所以测试都在事件循环里跑。
"""

from __future__ import annotations

# 标准库：asyncio / sys / pathlib。
import asyncio
import sys
from pathlib import Path

# 把项目根目录与 tests/ 插到 sys.path 最前面。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 第三方库：消息类型。
from langchain_core.messages import AIMessage, HumanMessage

# 项目内部：
from agent.context import AgentContext
from agent.graph import build_agent
from fakes import FakeModel
from langgraph.checkpoint.memory import InMemorySaver
from runtime.runs import RunBusyError, RunManager


# ---------- 测试脚手架 ----------


def _agent(*responses):
    """造一个会依次吐出给定响应的 agent。"""
    return build_agent(
        model=FakeModel(messages=iter(responses)),
        checkpointer=InMemorySaver(),
    )


def _ctx() -> AgentContext:
    return AgentContext(task_name="run 测试", trigger="cli", allow_write=False)


async def _drain(queue, stop_types=("run.end", "run.cancelled", "run.error"), timeout=15):
    """从队列里取事件直到收到结束事件。"""
    events = []
    while True:
        event = await asyncio.wait_for(queue.get(), timeout=timeout)
        events.append(event)
        if event.type in stop_types:
            return events


async def _start(mgr, *, owner="conn-1", instruction="列文件", thread_id=None, agent=None):
    """起一个 run，返回 (handle, queue, loop)。"""
    queue = asyncio.Queue(maxsize=64)
    loop = asyncio.get_running_loop()
    handle = mgr.start(
        owner=owner,
        instruction=instruction,
        task_name="t",
        thread_id=thread_id,
        agent=agent if agent is not None else _agent(AIMessage(content="好了")),
        queue=queue,
        loop=loop,
        ctx=_ctx(),
    )
    return handle, queue, loop


# ---------- run_id 与 thread_id ----------


def test_start_allocates_run_id_and_registers():
    """start 要分配 run_id 并登记。"""
    async def scenario():
        mgr = RunManager()
        handle, _, _ = await _start(mgr)
        assert handle.run_id.startswith("run-")
        assert len(handle.run_id) == len("run-") + 8
        assert mgr.get(handle.run_id) is handle
        assert len(mgr.active()) == 1

    asyncio.run(scenario())


def test_thread_id_defaults_to_one_shot():
    """不指定 thread_id 时用一次性的 —— 不留痕。"""
    async def scenario():
        mgr = RunManager()
        handle, _, _ = await _start(mgr, thread_id=None)
        assert handle.thread_id.startswith("task-")

    asyncio.run(scenario())


def test_explicit_thread_id_is_kept():
    """指定了 thread_id 就用它 —— 跨多次执行复用同一份记忆。"""
    async def scenario():
        mgr = RunManager()
        handle, _, _ = await _start(mgr, thread_id="my-chat")
        assert handle.thread_id == "my-chat"

    asyncio.run(scenario())


def test_two_runs_can_share_a_thread():
    """同一个会话里可以连续跑多个 run —— 记忆共享，run_id 不同。"""
    async def scenario():
        mgr = RunManager()
        agent = _agent(AIMessage(content="一"), AIMessage(content="二"))
        h1, _, _ = await _start(mgr, thread_id="chat", agent=agent)
        h2, _, _ = await _start(mgr, owner="conn-2", thread_id="chat", agent=agent)
        assert h1.run_id != h2.run_id
        assert h1.thread_id == h2.thread_id == "chat"

    asyncio.run(scenario())


# ---------- 单连接单任务 ----------


def test_same_owner_second_run_is_rejected():
    """同一连接已有活跃 run 时要拒绝，而不是静默取消旧的。"""
    async def scenario():
        mgr = RunManager()
        await _start(mgr, owner="conn-1")
        try:
            await _start(mgr, owner="conn-1")
            raise AssertionError("应该抛 RunBusyError")
        except RunBusyError:
            pass
        # 旧的 run 还在，没有被静默取消。
        assert len(mgr.active_for("conn-1")) == 1

    asyncio.run(scenario())


def test_different_owners_can_run_concurrently():
    """不同连接之间可以并发。"""
    async def scenario():
        mgr = RunManager()
        await _start(mgr, owner="conn-1")
        await _start(mgr, owner="conn-2")
        assert len(mgr.active()) == 2
        assert len(mgr.active_for("conn-1")) == 1
        assert len(mgr.active_for("conn-2")) == 1

    asyncio.run(scenario())


# ---------- 取消 ----------


def test_cancel_hits_active_run():
    """取消活跃的 run 返回 True。"""
    async def scenario():
        mgr = RunManager()
        handle, _, _ = await _start(mgr)
        assert mgr.cancel(handle.run_id) is True
        assert handle.cancel_event.is_set()

    asyncio.run(scenario())


def test_cancel_unknown_run_returns_false():
    """取消不存在的 run 返回 False。"""
    async def scenario():
        mgr = RunManager()
        assert mgr.cancel("run-nonexistent") is False

    asyncio.run(scenario())


def test_cancel_results_in_cancelled_event():
    """取消最终表现为一条 run.cancelled 事件（不是 run.end）。"""
    async def scenario():
        mgr = RunManager()
        handle, queue, _ = await _start(mgr)
        assert mgr.cancel(handle.run_id) is True
        events = await _drain(queue)
        types = [e.type for e in events]
        assert types[-1] == "run.cancelled"
        assert "run.end" not in types

    asyncio.run(scenario())


# ---------- 注销 ----------


def test_finish_is_idempotent():
    """finish 重复调用无害。"""
    async def scenario():
        mgr = RunManager()
        handle, _, _ = await _start(mgr)
        mgr.finish(handle.run_id)
        mgr.finish(handle.run_id)
        assert mgr.get(handle.run_id) is None
        assert mgr.active() == []

    asyncio.run(scenario())


def test_run_is_unregistered_after_it_finishes():
    """任务跑完后自动注销 —— 不用手动调 finish。"""
    async def scenario():
        mgr = RunManager()
        handle, queue, _ = await _start(mgr)
        await _drain(queue)
        # 等 done callback 跑完。
        await asyncio.sleep(0.05)
        assert mgr.get(handle.run_id) is None
        assert mgr.active() == []

    asyncio.run(scenario())


def test_run_end_event_carries_output():
    """正常跑完发的是 run.end，带最终答案。"""
    async def scenario():
        mgr = RunManager()
        agent = _agent(AIMessage(content="一共 3 个文件"))
        _, queue, _ = await _start(mgr, agent=agent)
        events = await _drain(queue)
        assert events[-1].type == "run.end"
        assert "3 个文件" in events[-1].output

    asyncio.run(scenario())
