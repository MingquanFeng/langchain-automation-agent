"""
PII 脱敏中间件的测试。全部用假模型，不调真实 API。

主要职责：
1. 守住「用户指令里的 PII 不进模型」—— 这是脱敏的第一道关。
2. 守住「工具结果里的 PII 不进模型」—— 文件工具读出来的内容走这条，
   是本项目最主要的泄露通道。
3. 守住「不误伤」—— 订单号、金额、日期这类数字不该被当手机号改掉。
4. 守住「模型输出不被脱敏」—— 用户自己知道自己的邮箱是什么，
   脱敏输出会让「提醒我邮箱是多少」这种场景失效。

注意：
这些测试不碰真实 API，但**会真的调用模型**（用假模型），
所以脱敏效果要从「模型实际收到了什么」来断言，而不是从中间件的返回值断言。
"""

from __future__ import annotations

# 标准库：dataclasses.replace 用来造一份改过开关的配置副本。
import dataclasses
import sys
from pathlib import Path

# 把项目根目录插到 sys.path 最前面。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 第三方库：消息类型，构造带 PII 的输入。
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# 项目内部：
from agent.config import RUNTIME
from agent.middleware import pii as pii_mod
from agent.middleware.pii import build_pii_middleware


# ---------- 单元层：中间件本身 ----------


def _run_middleware(messages, middleware=None):
    """让 PII 中间件处理一组消息，返回处理后的消息列表。

    没检测到 PII 时中间件返回 None（不改 state），此时原样返回。
    """
    mws = middleware if middleware is not None else build_pii_middleware()
    state = {"messages": messages}
    for mw in mws:
        result = mw.before_model(state, None)
        if result:
            state = {**state, **result}
    return state["messages"]


def test_user_message_pii_is_redacted():
    """用户指令里的邮箱、手机号、身份证都要被替换掉。"""
    msgs = [HumanMessage(content="发邮件给 zhangsan@example.com，手机 13812345678，身份证 11010119900307861X")]
    out = _run_middleware(msgs)

    text = out[0].content
    assert "zhangsan@example.com" not in text
    assert "13812345678" not in text
    assert "11010119900307861X" not in text
    assert "[REDACTED_EMAIL]" in text
    assert "[REDACTED_PHONE_CN]" in text
    assert "[REDACTED_ID_CARD_CN]" in text


def test_tool_result_pii_is_redacted():
    """工具结果里的 PII 也要被替换 —— 文件内容走的就是这条路。"""
    msgs = [
        HumanMessage(content="读一下通讯录"),
        AIMessage(content="好的", tool_calls=[{"name": "read_file", "args": {}, "id": "c1", "type": "tool_call"}]),
        ToolMessage(content="张三 zhangsan@example.com 13812345678", tool_call_id="c1"),
    ]
    out = _run_middleware(msgs)

    tool_msg = [m for m in out if isinstance(m, ToolMessage)][0]
    assert "zhangsan@example.com" not in tool_msg.content
    assert "13812345678" not in tool_msg.content
    assert "[REDACTED_EMAIL]" in tool_msg.content


def test_plain_numbers_are_not_touched():
    """订单号、金额这类数字不能被当手机号改掉。"""
    msgs = [HumanMessage(content="订单 20261103 共 1234567890 元")]
    out = _run_middleware(msgs)
    assert out[0].content == "订单 20261103 共 1234567890 元"


def test_message_without_pii_is_unchanged():
    """没有 PII 时内容一个字都不该变。"""
    msgs = [HumanMessage(content="把 data 目录下的报告读出来")]
    out = _run_middleware(msgs)
    assert out[0].content == "把 data 目录下的报告读出来"


def test_masking_disabled_returns_empty_list():
    """脱敏开关关掉时不该挂任何中间件。"""
    disabled = dataclasses.replace(RUNTIME, pii_masking=False)
    original = pii_mod.RUNTIME
    try:
        pii_mod.RUNTIME = disabled
        assert build_pii_middleware() == []
    finally:
        pii_mod.RUNTIME = original


def test_four_pii_types_configured():
    """四种类型都要挂上 —— 少一种就漏一种。"""
    mws = build_pii_middleware()
    assert len(mws) == 4
    # 用「检测器实际能不能命中」来验证，比依赖内部字段名稳。
    for pii_type, sample in [
        ("email", "a@b.com"),
        ("credit_card", "4111111111111111"),
        ("phone_cn", "13812345678"),
        ("id_card_cn", "11010119900307861X"),
    ]:
        hit = [mw for mw in mws if mw._process_content(sample)[1]]
        assert hit, f"{pii_type} 没命中"


# ---------- 端到端层：模型实际收到了什么 ----------


class _Recorder:
    """记录模型实际收到的消息列表。"""

    def __init__(self):
        self.seen = []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, *args, **kwargs):
        self.seen.append(messages)
        return AIMessage(content="好的")


def test_model_receives_redacted_instruction():
    """模型实际收到的指令里不能有原始 PII。"""
    from agent.context import AgentContext
    from agent.graph import build_agent
    from langgraph.checkpoint.memory import InMemorySaver

    rec = _Recorder()
    agent = build_agent(model=rec, checkpointer=InMemorySaver())
    agent.invoke(
        {"messages": [HumanMessage(content="发邮件给 zhangsan@example.com")]},
        {"configurable": {"thread_id": "pii-e2e"}},
        context=AgentContext(task_name="t", trigger="cli", allow_write=False),
    )

    seen = " ".join(str(getattr(m, "content", "")) for m in rec.seen[0])
    assert "zhangsan@example.com" not in seen
    assert "[REDACTED_EMAIL]" in seen


def test_model_output_is_not_redacted():
    """模型输出不脱敏 —— 用户自己知道自己的邮箱是什么。"""
    from agent.context import AgentContext
    from agent.graph import build_agent
    from langgraph.checkpoint.memory import InMemorySaver

    class _EchoModel(_Recorder):
        def invoke(self, messages, *args, **kwargs):
            self.seen.append(messages)
            return AIMessage(content="您的邮箱是 zhangsan@example.com")

    rec = _EchoModel()
    agent = build_agent(model=rec, checkpointer=InMemorySaver())
    result = agent.invoke(
        {"messages": [HumanMessage(content="我的邮箱是 zhangsan@example.com")]},
        {"configurable": {"thread_id": "pii-out"}},
        context=AgentContext(task_name="t", trigger="cli", allow_write=False),
    )

    # 输入被脱敏了
    seen = " ".join(str(getattr(m, "content", "")) for m in rec.seen[0])
    assert "zhangsan@example.com" not in seen
    # 但输出原样保留
    assert "zhangsan@example.com" in str(result["messages"][-1].content)
