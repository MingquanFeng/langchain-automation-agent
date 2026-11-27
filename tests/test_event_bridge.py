"""
事件桥接中间件的测试。全部用假模型，不调真实 API。

主要职责：
1. 守住「工具执行时发出 tool.start / tool.end」—— 桌面版的工具卡片靠它。
2. 守住「工具失败时发 tool.error，且异常继续往外抛」——
   中间件不吞异常，重试策略是别人的职责。
3. 守住「发不出去事件不影响工具执行」—— 拿不到 stream writer 时要静默跳过。
4. 守住「发出的事件能被归一化层解析成事件对象」——
   这是阶段 1 与阶段 2 的接缝，字段对不上就白做。

注意：
这些测试要真的调工具（用项目的真实工具），
所以断言的是「客户端实际收到了什么」，不是中间件的返回值。
"""

from __future__ import annotations

# 标准库：sys / pathlib 处理导入路径。
import sys
import threading
from pathlib import Path

# 把项目根目录与 tests/ 插到 sys.path 最前面。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# pytest：测试框架，用 pytest.raises 断言异常往外抛。
import pytest

# 第三方库：工具与消息类型。
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

# 项目内部：
from agent.context import AgentContext
from agent.graph import build_agent
from agent.middleware.events import EventBridgeMiddleware
from fakes import FakeModel
from runtime.events import StreamNormalizer, ToolEnd, ToolError, ToolStart


# ---------- 端到端：事件真的进了流 ----------


def _run_agent(model, tools=None, thread_id="evt-1"):
    """跑一次 agent，把所有流归一化成事件。

    返回归一化后的事件列表。
    """
    from langgraph.checkpoint.memory import InMemorySaver

    agent = build_agent(model=model, tools=tools, checkpointer=InMemorySaver())
    normalizer = StreamNormalizer(run_id="run-evt")
    events = []
    for mode, payload in agent.stream(
        {"messages": [HumanMessage(content="列出数据目录下的文件")]},
        {"configurable": {"thread_id": thread_id}},
        context=AgentContext(task_name="t", trigger="cli", allow_write=False),
        stream_mode=["messages", "custom"],
    ):
        events.extend(normalizer.normalize(mode, payload))
    return events


def test_tool_start_and_end_are_emitted():
    """工具执行时前后各发一条事件。"""
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "list_files", "args": {}, "id": "c1", "type": "tool_call"}],
    )
    model = FakeModel(messages=iter([tool_call, AIMessage(content="有 3 个文件")]))
    events = _run_agent(model)

    starts = [e for e in events if isinstance(e, ToolStart)]
    ends = [e for e in events if isinstance(e, ToolEnd)]

    assert len(starts) == 1
    assert starts[0].name == "list_files"
    assert starts[0].call_id == "c1"
    assert starts[0].run_id == "run-evt"

    assert len(ends) == 1
    assert ends[0].ok is True
    assert ends[0].duration_ms is not None
    assert ends[0].duration_ms >= 0


def test_events_appear_in_order():
    """tool.start 必须在 tool.end 之前。"""
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "list_files", "args": {}, "id": "c1", "type": "tool_call"}],
    )
    model = FakeModel(messages=iter([tool_call, AIMessage(content="好了")]))
    events = _run_agent(model)

    types = [e.type for e in events if e.type in ("tool.start", "tool.end")]
    assert types == ["tool.start", "tool.end"]


def test_tool_error_is_emitted_once():
    """工具失败时发 tool.error。

    注意两条：
    1. **只发一条**，不是重试一次发一条。
       事件桥接在中间件栈的最外层，包住的是「工具调用 + 重试」整体，
       所以客户端看到的是「一次工具调用失败了」，而不是「失败了三次」。
    2. **异常不往外抛**。
       ToolRetryMiddleware 的 `on_failure="continue"` 会把失败包成 ToolMessage 继续跑。
       这是项目的既有行为（见 CLAUDE.md 的陷阱 6.10）。
    """

    @tool
    def boom(x: str) -> str:
        """一个会失败的工具"""
        raise RuntimeError("工具炸了")

    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "boom", "args": {"x": "1"}, "id": "c9", "type": "tool_call"}],
    )
    # 工具错误交给模型解释；必须提供后续回答，避免以模型耗尽响应冒充正常结束。
    model = FakeModel(messages=iter([tool_call, AIMessage(content="工具失败，未完成操作")]))

    from langgraph.checkpoint.memory import InMemorySaver

    agent = build_agent(model=model, tools=[boom], checkpointer=InMemorySaver())
    normalizer = StreamNormalizer(run_id="run-evt")
    events = []

    # 不抛异常 —— 失败被重试中间件吃掉了。
    for mode, payload in agent.stream(
        {"messages": [HumanMessage(content="跑一下")]},
        {"configurable": {"thread_id": "evt-err"}},
        # boom 是本测试临时注入的工具，不在生产工具风险登记表中。
        # 显式允许执行，确保测试覆盖 EventBridge 的异常事件，而不是被只读闸门提前拦截。
        context=AgentContext(task_name="t", trigger="cli", allow_write=True),
        stream_mode=["messages", "custom"],
    ):
        events.extend(normalizer.normalize(mode, payload))

    err_events = [e for e in events if isinstance(e, ToolError)]
    assert len(err_events) == 1
    assert err_events[0].name == "boom"
    assert err_events[0].call_id == "c9"
    assert "工具炸了" in err_events[0].message


def test_failed_command_result_is_preserved_on_tool_error(monkeypatch):
    """失败命令走 tool.error 时仍要保留状态，工作台才能显示修复入口。"""
    from langchain_core.messages import ToolMessage

    context = AgentContext(allow_write=True, single_command=False)
    monkeypatch.setattr("agent.middleware.events.decide_tool_permission", lambda *args, **kwargs: "allow")
    monkeypatch.setattr(
        "agent.middleware.events.prepare_approval_preview",
        lambda *args, **kwargs: type("_Preview", (), {"args": {}, "error": None})(),
    )
    context.command_results["failed-command"] = {"status": "failed", "exit_code": 1}

    class _Req:
        tool_call = {"name": "run_command", "args": {"argv": ["npm", "test"]}, "id": "failed-command"}
        tool = None
        state = {}
        runtime = type("_Runtime", (), {"context": context})()

    events = []
    middleware = EventBridgeMiddleware()
    middleware._emit = events.append

    def failed_command(request):
        return ToolMessage(
            content="命令失败（退出码 1）",
            name="run_command",
            tool_call_id="failed-command",
            status="error",
        )

    middleware.wrap_tool_call(_Req(), failed_command)

    normalizer = StreamNormalizer(run_id="run-failed")
    normalized = [
        event
        for payload in events
        for event in normalizer.normalize("custom", payload)
    ]
    error = next(event for event in normalized if isinstance(event, ToolError))
    assert error.command_result == {"status": "failed", "exit_code": 1}


# ---------- 边界：发不出去也不该崩 ----------


def test_missing_stream_writer_is_silent():
    """不在图的执行上下文里发事件，不该抛异常。"""
    from langchain_core.messages import ToolMessage

    mw = EventBridgeMiddleware()

    class _Req:
        tool_call = {"name": "ping", "args": {}, "id": "c1"}
        tool = None
        state = {}
        class runtime:
            from agent.context import AgentContext
            context = AgentContext(allow_write=True)

    def handler(request):
        return ToolMessage(content="pong", tool_call_id="c1")

    # 图外调 get_stream_writer() 会抛 RuntimeError，中间件必须兜住。
    result = mw.wrap_tool_call(_Req(), handler)
    assert result.content == "pong"


def test_cancelled_approval_returns_tool_message_for_checkpoint():
    """审批期间取消也必须闭合 tool_call，避免同会话下一轮请求触发协议 400。"""
    from langchain_core.messages import ToolMessage

    cancel_event = threading.Event()

    def cancel_while_waiting(name, call_id, args):
        """模拟用户在审批卡片出现后点击取消。"""
        cancel_event.set()
        return False

    context = AgentContext(
        allow_write=False,
        approval_handler=cancel_while_waiting,
        cancel_event=cancel_event,
    )

    class _Req:
        tool_call = {
            "name": "write_file",
            "args": {"path": "out/cancelled.md", "content": "x"},
            "id": "cancel-call",
        }
        tool = None
        state = {}

    _Req.runtime = type("_Runtime", (), {"context": context})()

    def must_not_run(request):
        raise AssertionError("取消后的工具不应执行")

    result = EventBridgeMiddleware().wrap_tool_call(_Req(), must_not_run)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "cancel-call"
    assert result.status == "error"
    assert result.content == "任务已取消，工具未执行。"


def test_result_preview_is_truncated():
    """工具结果太长时只带一段预览。"""
    from langchain_core.messages import ToolMessage

    mw = EventBridgeMiddleware()
    long_text = "x" * 5000

    class _Req:
        tool_call = {"name": "ping", "args": {}, "id": "c1"}
        tool = None
        state = {}
        runtime = None

    def handler(request):
        return ToolMessage(content=long_text, tool_call_id="c1")

    mw.wrap_tool_call(_Req(), handler)
    # 图外发不出去，事件进不了流 —— 这里只能验证不崩。
    # 截断逻辑本身由 _preview 负责，另有一项直接测它。
    from agent.middleware.events import _preview

    assert len(_preview(ToolMessage(content=long_text, tool_call_id="c1"))) == 500
