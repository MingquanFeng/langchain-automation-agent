"""
同步流消费器的测试。全部用假模型，不调真实 API。

主要职责：
1. 守住「事件顺序与内容」—— 客户端按顺序渲染，顺序错了界面就乱。
2. 守住「异常路径必须闭合」——
   正常结束发 run.end、取消发 run.cancelled、失败发 run.error，
   **任何一种退出都要有一条结束事件**，否则客户端会一直转圈。
3. 守住「背压」—— 队列满时生产者要停下来，不能把内存吃光。
4. 守住「跨线程」—— 消费器在工作线程跑，事件要能到达事件循环的队列。

注意：
消费器是阻塞的，所以测试里它跑在**工作线程**，
asyncio 事件循环在主线程 —— 这正好也是桌面版的真实拓扑。
"""

from __future__ import annotations

# 标准库：asyncio / threading / time。
import asyncio
import threading
import time

# sys / pathlib 处理导入路径。
import sys
from pathlib import Path

# 把项目根目录与 tests/ 插到 sys.path 最前面。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 第三方库：消息类型与假模型。
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

# 项目内部：
from agent.context import AgentContext
from agent.graph import build_agent
from fakes import FakeModel
from langgraph.checkpoint.memory import InMemorySaver
from runtime.consumer import StreamConsumer
from runtime.events import RunCancelled, RunEnd, RunError


# ---------- 测试脚手架 ----------


def _ctx() -> AgentContext:
    """一份普通的运行时上下文。"""
    return AgentContext(task_name="消费器测试", trigger="cli", allow_write=False)


async def _collect(queue, stop_types=("run.end", "run.cancelled", "run.error"), timeout=15):
    """从队列里取事件，直到收到一条结束事件。

    返回事件列表。
    """
    events = []
    while True:
        event = await asyncio.wait_for(queue.get(), timeout=timeout)
        events.append(event)
        if event.type in stop_types:
            return events


def _run_consumer(agent, queue, loop, *, instruction="列出数据目录下的文件",
                  cancel_event=None, thread_id="consumer-1", max_pending=None):
    """在工作线程里跑一次消费器。

    返回工作线程（测试可以 join 它）。
    """
    kwargs = {}
    if max_pending is not None:
        kwargs["max_pending"] = max_pending
    consumer = StreamConsumer(
        agent=agent,
        run_id="run-consumer",
        queue=queue,
        loop=loop,
        cancel_event=cancel_event,
        **kwargs,
    )
    thread = threading.Thread(
        target=consumer.run,
        args=(instruction, _ctx(), thread_id),
        daemon=True,
    )
    thread.start()
    return thread


def _make_agent(*responses):
    """造一个会依次吐出给定响应的 agent。

    第一个响应固定是「要求调工具」，后面的是模型的最终回答。
    """
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "list_files", "args": {}, "id": "c1", "type": "tool_call"}],
    )
    return build_agent(
        model=FakeModel(messages=iter([tool_call, *responses])),
        checkpointer=InMemorySaver(),
    )


# ---------- 正常流程 ----------


def test_normal_run_emits_events_in_order():
    """正常跑完：工具事件 → 模型文本 → run.end。"""
    async def scenario():
        queue = asyncio.Queue(maxsize=64)
        agent = _make_agent(AIMessage(content="有 3 个文件"))
        thread = _run_consumer(agent, queue, asyncio.get_running_loop())
        events = await _collect(queue)
        thread.join(timeout=5)
        return events

    events = asyncio.run(scenario())
    types = [e.type for e in events]

    assert "tool.start" in types
    assert "tool.end" in types
    assert "model.text" in types
    assert types[-1] == "run.end"


def test_run_end_carries_final_answer_and_counts():
    """run.end 要带最终答案与消息条数。"""
    async def scenario():
        queue = asyncio.Queue(maxsize=64)
        agent = _make_agent(AIMessage(content="有 3 个文件"))
        thread = _run_consumer(agent, queue, asyncio.get_running_loop())
        events = await _collect(queue)
        thread.join(timeout=5)
        return events

    events = asyncio.run(scenario())
    end = events[-1]
    assert isinstance(end, RunEnd)
    assert "有 3 个文件" in end.output
    assert end.messages >= 1
    assert end.duration_ms is not None


def test_all_events_share_the_same_run_id():
    """同一批事件共享一个 run_id —— 客户端靠它把事件归到一次执行。"""
    async def scenario():
        queue = asyncio.Queue(maxsize=64)
        agent = _make_agent(AIMessage(content="好了"))
        thread = _run_consumer(agent, queue, asyncio.get_running_loop())
        events = await _collect(queue)
        thread.join(timeout=5)
        return events

    events = asyncio.run(scenario())
    assert {e.run_id for e in events} == {"run-consumer"}


# ---------- 取消 ----------


def test_cancel_emits_cancelled_not_end():
    """取消发的是 run.cancelled，不是 run.end。"""
    async def scenario():
        queue = asyncio.Queue(maxsize=64)
        agent = _make_agent(AIMessage(content="好了"))
        cancel_event = threading.Event()
        cancel_event.set()          # 预先取消
        thread = _run_consumer(agent, queue, asyncio.get_running_loop(),
                               cancel_event=cancel_event)
        events = await _collect(queue)
        thread.join(timeout=5)
        return events

    events = asyncio.run(scenario())
    types = [e.type for e in events]
    assert types[-1] == "run.cancelled"
    assert "run.end" not in types
    assert isinstance(events[-1], RunCancelled)


def test_cancel_midway_stops_the_run():
    """跑到一半取消：后续事件不再产生。"""
    async def scenario():
        queue = asyncio.Queue(maxsize=64)

        class SlowModel(FakeModel):
            """第一次调用慢一点，给取消留出时间窗。"""

            def _stream(self, messages, stop=None, run_manager=None, **kwargs):
                time.sleep(0.5)
                yield from super()._stream(messages, stop=stop,
                                          run_manager=run_manager, **kwargs)

        tool_call = AIMessage(
            content="",
            tool_calls=[{"name": "list_files", "args": {}, "id": "c1", "type": "tool_call"}],
        )
        agent = build_agent(
            model=SlowModel(messages=iter([tool_call, AIMessage(content="x")])),
            checkpointer=InMemorySaver(),
        )

        cancel_event = threading.Event()
        thread = _run_consumer(agent, queue, asyncio.get_running_loop(),
                               cancel_event=cancel_event)
        # 等第一次模型调用过去，取消标志置位。
        await asyncio.sleep(0.8)
        cancel_event.set()
        events = await _collect(queue)
        thread.join(timeout=5)
        return events

    events = asyncio.run(scenario())
    types = [e.type for e in events]
    assert types[-1] == "run.cancelled"
    assert "run.end" not in types


# ---------- 异常 ----------


def test_model_failure_is_wrapped_into_the_answer():
    """模型调用失败不会逃出消费循环。

    为什么：
    `ModelRetryMiddleware` 的默认 `on_failure="continue"` 会把重试耗尽后的失败
    包成一条 AIMessage 继续跑 —— 图本身是「成功」跑完的。
    所以这里发的是 run.end，而错误信息出现在最终答案里。

    这一点与改造计划里写的「模型调用失败 → 异常逃出消费循环 → run.error」不同，
    是项目的 `on_failure="continue"` 造成的。要让模型失败触发 run.error，
    得把 `agent/graph.py` 里 `ModelRetryMiddleware` 的 `on_failure` 改成 `"error"`。

    另外一条路径仍会发 run.error —— 异常不被任何中间件兜住时
    （见下面 test_unhandled_exception_emits_run_error）。
    """

    async def scenario():
        queue = asyncio.Queue(maxsize=64)

        class BoomModel(GenericFakeChatModel):
            """一调用就炸的模型。"""

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, *args, **kwargs):
                raise RuntimeError("模型挂了")

        agent = build_agent(
            model=BoomModel(messages=iter([])),
            checkpointer=InMemorySaver(),
        )
        thread = _run_consumer(agent, queue, asyncio.get_running_loop())
        events = await _collect(queue)
        thread.join(timeout=5)
        return events

    events = asyncio.run(scenario())
    types = [e.type for e in events]
    assert types[-1] == "run.end"
    assert isinstance(events[-1], RunEnd)
    # 错误信息出现在最终答案里 —— 界面能看出来这次其实失败了。
    assert "模型挂了" in events[-1].output


def test_unhandled_exception_emits_run_error():
    """没有被中间件兜住的异常 → run.error，没有 run.end。

    模拟图执行本身出错（比如 checkpointer 挂了）——
    这种异常不经过重试中间件，会直接逃出消费循环。
    """

    async def scenario():
        queue = asyncio.Queue(maxsize=64)

        class ExplodingAgent:
            """一开流就炸的「图」。"""

            def stream(self, *args, **kwargs):
                raise RuntimeError("图崩了")
                yield  # pragma: no cover —— 让它是个生成器

        consumer = StreamConsumer(
            agent=ExplodingAgent(),
            run_id="run-consumer",
            queue=queue,
            loop=asyncio.get_running_loop(),
        )
        thread = threading.Thread(
            target=consumer.run,
            args=("跑一下", _ctx(), "consumer-x"),
            daemon=True,
        )
        thread.start()
        events = await _collect(queue)
        thread.join(timeout=5)
        return events

    events = asyncio.run(scenario())
    types = [e.type for e in events]
    assert types[-1] == "run.error"
    assert "run.end" not in types
    assert isinstance(events[-1], RunError)
    assert "图崩了" in events[-1].message


# ---------- 背压 ----------


def test_pending_limit_blocks_the_producer():
    """在途事件达到上限时生产者停下来 —— 背压。

    两层上限：
        队列容量（maxsize）管「已入队」
        在途上限（max_pending）管「已投递未入队」

    这里把两层都设成 1：不消费队列，
    生产者投递第二个事件时就会卡在计数上。
    """
    async def scenario():
        queue = asyncio.Queue(maxsize=1)
        agent = _make_agent(AIMessage(content="好了"))
        thread = _run_consumer(agent, queue, asyncio.get_running_loop(), max_pending=1)
        # 给它一点时间去产出事件并卡住。
        await asyncio.sleep(1.0)
        alive = thread.is_alive()
        size = queue.qsize()
        return alive, size

    alive, size = asyncio.run(scenario())
    # 队列满（1），生产者还活着（卡在计数上）—— 这就是背压。
    assert size == 1
    assert alive, "在途事件达到上限时生产者应该被阻塞，而不是把内存吃光"


def test_events_are_not_dropped_when_queue_is_full():
    """队列满着的时候，事件一个都不能丢。"""
    async def scenario():
        queue = asyncio.Queue(maxsize=1)
        agent = _make_agent(AIMessage(content="好了"))
        thread = _run_consumer(agent, queue, asyncio.get_running_loop())
        # 先让它卡住，再开始消费。
        await asyncio.sleep(0.5)
        events = await _collect(queue)
        thread.join(timeout=5)
        return events

    events = asyncio.run(scenario())
    # 消费开始后，队列里积压的事件会依次放出来，最后收到结束事件。
    assert events[-1].type in ("run.end", "run.cancelled", "run.error")
    # 工具事件没丢。
    assert any(e.type == "tool.start" for e in events)
