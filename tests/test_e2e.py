"""端到端测试。全部用假模型，不调真实 API。

注意：FakeListChatModel 不支持 tool_calls（bind_tools 未实现），
要测工具调用链必须用 GenericFakeChatModel，它接受一个 AIMessage 迭代器。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agent.context import AgentContext
from agent.graph import build_agent
from agent.tools import ALL_TOOLS
from agent.tools.files import SANDBOX


def make_model(*responses: AIMessage) -> "ToolCapableFakeModel":
    """把固定的 AIMessage 序列喂给假模型。

    必须实现 bind_tools —— BaseChatModel.bind_tools 默认抛 NotImplementedError，
    而任何带工具的 agent 都会走到这里。
    """
    return ToolCapableFakeModel(messages=iter(responses))


class ToolCapableFakeModel(GenericFakeChatModel):
    """支持 bind_tools 的假模型。"""

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


# ---------- 1. 工具可调用性 ----------


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
def test_tool_is_callable(tool):
    args = {
        "path": "probe.txt",
        "content": "hello",
        "data": {"a": 1},
        "pattern": "*",
        "text": "t",
        "title": "x",
        "items": "y",
    }
    result = tool.invoke(args)
    assert isinstance(result, str)


# ---------- 2. Agent 装配 ----------


def test_agent_builds_with_full_middleware_stack():
    import agent.graph as graph_mod

    agent = build_agent(model=make_model(AIMessage(content="ok")), checkpointer=None)
    assert agent is not None

    assembled = [
        *graph_mod.LIMITS,
        *graph_mod.RETRY,
        graph_mod.TODO,
        graph_mod.CONTEXT_EDITING,
        *graph_mod.CUSTOM_MIDDLEWARE,
    ]
    names = {type(m).__name__ for m in assembled}
    for expected in [
        "ModelCallLimitMiddleware",
        "ToolCallLimitMiddleware",
        "ModelRetryMiddleware",
        "ToolRetryMiddleware",
        "TodoListMiddleware",
        "ContextEditingMiddleware",
        "TaskGateMiddleware",
        "ContextInjectionMiddleware",
        "TimingMiddleware",
    ]:
        assert expected in names, f"缺少中间件 {expected}"


# ---------- 3. 工具调用链真的执行 ----------


def test_write_tool_actually_runs(tmp_path):
    model = make_model(
        tool_call("write_file", {"path": "out/report.md", "content": "今日要点三条。"}),
        AIMessage(content="已写入。"),
    )
    agent = build_agent(model=model, checkpointer=None)
    result = agent.invoke(
        {"messages": [HumanMessage(content="写一份报告")]},
        {"configurable": {"thread_id": "t-write"}},
        context=AgentContext(task_name="t", allow_write=True),
    )
    assert "messages" in result
    assert (SANDBOX / "out" / "report.md").exists()
    assert "今日要点三条" in (SANDBOX / "out" / "report.md").read_text(encoding="utf-8")


# ---------- 4. 只读闸门 ----------


def test_readonly_mode_blocks_write():
    model = make_model(
        tool_call("write_file", {"path": "out/should_not_exist.md", "content": "x"}),
        AIMessage(content="已写入。"),
    )
    agent = build_agent(model=model, checkpointer=None)
    agent.invoke(
        {"messages": [HumanMessage(content="写文件")]},
        {"configurable": {"thread_id": "t-ro"}},
        context=AgentContext(task_name="ro", allow_write=False),
    )
    assert not (SANDBOX / "out" / "should_not_exist.md").exists()


# ---------- 5. 动态提示词注入 ----------


def test_context_injected_into_system_prompt():
    class Capture(ToolCapableFakeModel):
        """记录收到的消息。_generate 必须返回 ChatResult，不能返回 AIMessage。"""

        seen: list = []

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.outputs import ChatGeneration, ChatResult

            self.seen = [str(m.content) for m in messages]
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    model = Capture(messages=iter([AIMessage(content="ok")]))
    agent = build_agent(model=model, checkpointer=None)
    agent.invoke(
        {"messages": [HumanMessage(content="hi")]},
        {"configurable": {"thread_id": "t-prompt"}},
        context=AgentContext(task_name="prompt-test", allow_write=True),
    )
    assert model.seen, "模型没收到任何消息"
    joined = "\n".join(model.seen)
    assert "运行时上下文" in joined
    assert "prompt-test" in joined
    # override 是替换语义，这段断言守住「原系统提示词不能被冲掉」
    assert "自动化日常事务助手" in joined


# ---------- 6. 记忆 ----------


def test_same_thread_shares_history():
    from langgraph.checkpoint.memory import InMemorySaver

    agent = build_agent(
        model=make_model(AIMessage(content="a"), AIMessage(content="b")),
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "shared"}}
    ctx = AgentContext(task_name="mem")
    r1 = agent.invoke({"messages": [HumanMessage(content="我叫张三")]}, cfg, context=ctx)
    r2 = agent.invoke({"messages": [HumanMessage(content="我叫什么")]}, cfg, context=ctx)
    assert len(r2["messages"]) > len(r1["messages"])


# ---------- 7. 中间件不因缺省 context 而崩溃 ----------


def test_middleware_tolerates_missing_context():
    """不传 context 时 runtime.context 是 None，中间件不能因此抛错。"""
    agent = build_agent(model=make_model(AIMessage(content="ok")), checkpointer=None)
    result = agent.invoke(
        {"messages": [HumanMessage(content="hi")]},
        {"configurable": {"thread_id": "t-noctx"}},
    )
    assert "messages" in result


# ---------- 8. 沙箱边界 ----------


def test_file_tools_reject_path_outside_sandbox():
    from agent.tools.files import write_file

    with pytest.raises(ValueError, match="超出允许范围"):
        write_file.invoke({"path": "../../etc/passwd", "content": "x"})


# ---------- 9. 流式接口的会话隔离 ----------


def test_stream_endpoint_isolates_threads(monkeypatch):
    """流式接口默认每次新会话；显式传 thread_id 才复用。

    回归防护：曾经硬编码成 "stream-session"，导致所有流式请求共享同一份记忆。
    """
    from runtime import server
    from runtime.server import TaskRequest

    seen: list[str] = []

    class FakeAgent:
        def stream(self, messages, config, context=None, stream_mode=None):
            seen.append(config["configurable"]["thread_id"])
            yield {"messages": [AIMessage(content="ok")]}

    monkeypatch.setattr(server, "get_agent", lambda: FakeAgent())

    async def drain(resp):
        return [chunk async for chunk in resp.body_iterator]

    asyncio.run(drain(server.run_stream(TaskRequest(instruction="第一次"))))
    asyncio.run(drain(server.run_stream(TaskRequest(instruction="第二次"))))
    asyncio.run(drain(server.run_stream(TaskRequest(instruction="第三次", thread_id="fixed"))))

    assert seen[0] != seen[1], "默认不该共享会话"
    assert seen[0].startswith("stream-")
    assert seen[1].startswith("stream-")
    assert seen[2] == "fixed", "显式传的 thread_id 要被尊重"
