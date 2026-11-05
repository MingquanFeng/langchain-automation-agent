"""
事件归一化层的测试。全部用假模型，不调真实 API。

主要职责：
1. 守住事件模型的公共字段与判别字段 —— 这是客户端的协议。
2. 守住思考内容过滤 —— 滤漏了会被大段自说自话刷屏，误伤了会丢正文。
3. 守住中间件噪音判定 —— 滤多了会丢掉「写操作被拦截」这类必须显示的消息。
4. 守住归一化层的分派规则 —— 认错模式就会把工具结果当成模型输出发出去。

为什么归一化层要单独测：
它横在「LangGraph 的流」和「客户端的事件」之间，
两边任一边改了形状，最先坏的就是这里。
"""

from __future__ import annotations

# 标准库：sys / pathlib 处理导入路径。
import sys
from pathlib import Path

# 把项目根目录插到 sys.path 最前面。
# 这样不管从哪个目录跑 pytest，都能 import 到 agent / runtime 包。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 把 tests/ 目录也插进去，才能 import 到同目录的 fakes.py。
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 第三方库：消息类型，构造各种形态的 chunk。
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

# 项目内部：
# 事件模型与归一化层。
from runtime.events import (
    EVENT_CLASSES,
    BaseEvent,
    ModelText,
    RunEnd,
    RunStart,
    StreamNormalizer,
    ThinkFilter,
    ToolEnd,
    ToolStart,
    is_hint_node,
)


# ---------- 事件模型 ----------


def test_base_event_has_common_fields():
    """每个事件都要带 version / run_id / timestamp / type。"""
    e = ModelText(run_id="run-1", text="你好")
    assert e.version == 1
    assert e.run_id == "run-1"
    assert e.timestamp > 0
    assert e.type == "model.text"


def test_all_nine_event_types_registered():
    """九种事件都要在注册表里，一个都不能少。"""
    assert set(EVENT_CLASSES) == {
        "run.start",
        "model.text",
        "tool.start",
        "tool.end",
        "tool.error",
        "run.progress",
        "run.end",
        "run.cancelled",
        "run.error",
    }
    for name, cls in EVENT_CLASSES.items():
        # 每个类的 type 字段默认值必须与注册表的键一致 ——
        # 客户端靠 type 分发，键和字段对不上就会发错事件。
        assert cls.model_fields["type"].default == name, name
        assert issubclass(cls, BaseEvent)


def test_event_json_shape_matches_protocol():
    """序列化后的字段名就是客户端看到的协议，不能随便改。"""
    import json

    payload = json.loads(RunStart(
        run_id="run-a1b2", task_name="列文件", thread_id="t1", instruction="列文件"
    ).model_dump_json())

    assert payload["version"] == 1
    assert payload["run_id"] == "run-a1b2"
    assert payload["type"] == "run.start"
    assert payload["task_name"] == "列文件"
    assert payload["thread_id"] == "t1"
    assert payload["instruction"] == "列文件"
    assert isinstance(payload["timestamp"], float)


def test_extra_fields_are_ignored():
    """中间件发来的 dict 带额外字段时要忽略，不能报错。"""
    e = ModelText(run_id="r", text="x", something_new=123)
    assert e.text == "x"


def test_run_end_carries_result_fields():
    """run.end 要带最终答案与消息条数。"""
    e = RunEnd(run_id="r", output="完成", messages=4, duration_ms=812)
    assert e.output == "完成"
    assert e.messages == 4
    assert e.duration_ms == 812


# ---------- 思考内容过滤 ----------

# MiniMax 这类推理模型会把思考过程混在 content 里返回，
# 默认要滤掉，否则界面会被大段自说自话刷屏。


def test_think_filter_removes_think_block():
    """完整的 <think> 块要被整段拿掉，只留下正文。"""
    f = ThinkFilter()
    assert f.feed("<think>思考中</think>答案") == "答案"


def test_think_filter_handles_split_chunks():
    """闭标签还没到时也不能漏出思考内容。"""
    f = ThinkFilter()
    assert f.feed("<think>我在") == ""
    assert f.feed("思考</think>正文") == "正文"


def test_think_filter_keeps_plain_text():
    """普通文本要原样透传，过滤器不能误伤。"""
    f = ThinkFilter()
    assert f.feed("普通") == "普通"
    assert f.feed("回答") == "回答"


def test_think_filter_reset_clears_buffer():
    """reset 要把缓冲清空，否则上一轮的思考片段会混进这一轮。"""
    f = ThinkFilter()
    assert f.feed("<think>我在思考") == ""
    f.reset()
    assert f.feed("新的一轮") == "新的一轮"


# ---------- 中间件噪音判定 ----------


def test_is_hint_node_filters_context_injection():
    """中间件注入的例行提示要能被认出来。

    注意 node 名带后缀（实测是 "ContextInjectionMiddleware.before_model"），
    所以判定用的是前缀匹配。
    """
    assert is_hint_node("ContextInjectionMiddleware") is True
    assert is_hint_node("ContextInjectionMiddleware.before_model") is True


def test_is_hint_node_keeps_other_nodes():
    """模型节点和工具节点的消息必须留着 —— 写操作被拦的说明在那儿。"""
    assert is_hint_node("model") is False
    assert is_hint_node("tools") is False
    assert is_hint_node("") is False


# ---------- 归一化层 ----------


def test_normalizer_emits_model_text_from_chunk():
    """messages 模式里的增量文本要变成 model.text 事件。"""
    n = StreamNormalizer(run_id="run-1")
    events = n.normalize("messages", (AIMessageChunk(content="你好"), {"langgraph_node": "model"}))

    assert len(events) == 1
    assert isinstance(events[0], ModelText)
    assert events[0].text == "你好"
    assert events[0].run_id == "run-1"


def test_normalizer_skips_empty_content():
    """空内容的 chunk 不发事件 —— 工具调用的 chunk 就是空内容。"""
    n = StreamNormalizer(run_id="run-1")
    assert n.normalize("messages", (AIMessageChunk(content=""), {})) == []


def test_normalizer_skips_non_string_content():
    """content 可能是内容块列表（多模态），非字符串形态直接跳过。"""
    n = StreamNormalizer(run_id="run-1")
    chunk = AIMessageChunk(content=[{"type": "text", "text": "hi"}])
    assert n.normalize("messages", (chunk, {})) == []


def test_normalizer_drops_tool_message():
    """ToolMessage 是工具结果不是模型输出，必须丢掉。

    漏掉这条，模型的一句话会被当成两段发出去。
    """
    n = StreamNormalizer(run_id="run-1")
    msg = ToolMessage(content="文件列表", tool_call_id="c1")
    assert n.normalize("messages", (msg, {"langgraph_node": "tools"})) == []


def test_normalizer_drops_hint_message():
    """中间件插入的例行提示要滤掉。"""
    n = StreamNormalizer(run_id="run-1")
    msg = AIMessage(content="（提示：本轮为只读模式。）")
    events = n.normalize("messages", (msg, {"langgraph_node": "ContextInjectionMiddleware.before_model"}))
    assert events == []


def test_normalizer_keeps_non_hint_message():
    """非例行提示的整条消息要显示出来。"""
    n = StreamNormalizer(run_id="run-1")
    msg = AIMessage(content="写操作已被阻止")
    events = n.normalize("messages", (msg, {"langgraph_node": "model"}))
    assert len(events) == 1
    assert events[0].text == "写操作已被阻止"


def test_normalizer_filters_think_blocks():
    """混在正文里的 <think> 块不能发出去。"""
    n = StreamNormalizer(run_id="run-1")
    events = n.normalize("messages", (AIMessageChunk(content="<think>在想</think>答案"), {}))
    assert len(events) == 1
    assert events[0].text == "答案"


def test_normalizer_ignores_unknown_mode():
    """认不出来的 mode 返回空列表，不能抛异常。"""
    n = StreamNormalizer(run_id="run-1")
    assert n.normalize("values", {"messages": []}) == []
    assert n.normalize("debug", {}) == []


def test_normalizer_dispatches_custom_event():
    """custom 模式发来的 dict 要按 type 分发成对应事件。"""
    n = StreamNormalizer(run_id="run-1")
    events = n.normalize("custom", {
        "type": "tool.start", "name": "list_files", "call_id": "c1", "args": {},
    })

    assert len(events) == 1
    assert isinstance(events[0], ToolStart)
    assert events[0].name == "list_files"
    assert events[0].call_id == "c1"
    assert events[0].run_id == "run-1"


def test_normalizer_ignores_unknown_custom_type():
    """未知的事件类型要跳过 —— 中间件可能发了客户端还不认识的东西。"""
    n = StreamNormalizer(run_id="run-1")
    assert n.normalize("custom", {"type": "tool.whatever", "name": "x"}) == []


def test_normalizer_skips_malformed_custom_event():
    """字段对不上的坏事件要跳过，不能把整次执行搞崩。"""
    n = StreamNormalizer(run_id="run-1")
    # tool.end 需要 name 与 call_id，这里都不给。
    assert n.normalize("custom", {"type": "tool.end"}) == []


def test_normalizer_keeps_custom_timestamp():
    """custom 载荷自带 timestamp 时用它，否则用事件自己的默认时间。"""
    n = StreamNormalizer(run_id="run-1")
    events = n.normalize("custom", {
        "type": "tool.end", "name": "x", "call_id": "c1", "ok": True,
        "duration_ms": 3, "result_preview": "…", "timestamp": 1234.5,
    })
    assert isinstance(events[0], ToolEnd)
    assert events[0].timestamp == 1234.5
    assert events[0].duration_ms == 3


def test_normalizer_reset_clears_think_buffer():
    """reset 要把思考过滤器的缓冲清空。"""
    n = StreamNormalizer(run_id="run-1")
    assert n.normalize("messages", (AIMessageChunk(content="<think>我在想"), {})) == []
    n.reset()
    events = n.normalize("messages", (AIMessageChunk(content="正文"), {}))
    assert len(events) == 1
    assert events[0].text == "正文"


# ---------- 端到端：真跑一遍 agent ----------


def test_normalizer_end_to_end_with_real_agent():
    """拿真 agent 跑一遍，检查归一化后的结果符合预期。

    走的是项目真实的中间件栈（含只读闸门与上下文注入），
    验证三件事：
    1. 中间件插入的例行提示没漏进事件；
    2. 工具结果没被当成模型输出；
    3. 最终答案以 model.text 的形式出来了。
    """
    from agent.context import AgentContext
    from agent.graph import build_agent
    from fakes import FakeModel
    from langgraph.checkpoint.memory import InMemorySaver

    # 模型先要工具，再给答案。
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "list_files", "args": {}, "id": "c1", "type": "tool_call"}],
    )
    agent = build_agent(
        model=FakeModel(messages=iter([tool_call_msg, AIMessage(content="数据目录下有 3 个文件")])),
        checkpointer=InMemorySaver(),
    )

    ctx = AgentContext(task_name="t", trigger="cli", allow_write=False)
    normalizer = StreamNormalizer(run_id="run-e2e")

    events = []
    for mode, payload in agent.stream(
        {"messages": [HumanMessage(content="列出数据目录下的文件")]},
        {"configurable": {"thread_id": "test-thread"}},
        context=ctx,
        stream_mode=["messages"],
    ):
        events.extend(normalizer.normalize(mode, payload))

    texts = [e.text for e in events if isinstance(e, ModelText)]

    # 例行提示不能出现。
    assert not any("只读模式" in t for t in texts), texts
    # 工具结果不能出现。
    assert not any(".md" in t for t in texts), texts
    # 最终答案要出现。
    assert "数据目录下有 3 个文件" in "".join(texts), texts
    # 所有事件都带同一个 run_id。
    assert {e.run_id for e in events} == {"run-e2e"}
