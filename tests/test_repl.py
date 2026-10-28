"""
交互式终端（REPL）的测试。全部用假模型，不调真实 API。

主要职责：
1. 守住终端的四块行为：思考内容过滤、输出代理、斜杠命令、一轮对话。
2. 让终端的每一次输出都可预期 —— REPL 是人机界面，
   多一个换行、少一句提示，在屏幕上都很显眼。

为什么用假模型：
REPL 走的是 agent.stream() 流式路径，
真实模型每次吐的 token 都不一样，断言没法写；
假模型可以精确控制每一轮吐什么。

注意：
agent 走的是流式调用，所以假模型必须重写 _stream ——
GenericFakeChatModel 自带的 _stream 只吐 content，会把 tool_calls 整个丢掉。
"""

from __future__ import annotations

# 标准库：
# io：内存流，用来断言终端到底输出了什么。
# json：把工具参数序列化成字符串（流式协议里 args 是 JSON 文本）。
# re：按空白切分正文，模拟真实模型一个 token 一个 token 地吐。
# sys / pathlib：处理导入路径。
import io
import json
import re
import sys
from pathlib import Path

# pytest：测试框架，本文件用它做 capsys 捕获终端输出。
import pytest

# 把项目根目录插到 sys.path 最前面。
# 这样不管从哪个目录跑 pytest，都能 import 到 agent / runtime 包。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 第三方库：
# GenericFakeChatModel：可编排输出的假模型。
# AIMessage / AIMessageChunk：整条消息与流式增量两种形态。
# ChatGenerationChunk：流式协议里的最小单位。
# InMemorySaver：进程内存版 Checkpointer，不往 data/memory.sqlite 落东西。
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver

# 项目内部：
# build_agent：装配 Agent 的唯一入口。
# ReplSession：一次终端会话。
# _ThinkFilter：滤掉 <think> 思考块。
# _LineGuard：输出代理，保证第三方 print 独占一行。
from agent.graph import build_agent
from runtime.repl import ReplSession, _LineGuard, _ThinkFilter


class FakeModel(GenericFakeChatModel):
    """支持 bind_tools 与流式工具调用的假模型。

    GenericFakeChatModel 自带的 _stream 只吐 content，会把 tool_calls 整个丢掉，
    所以这里自己实现 _stream —— agent 走的就是流式路径。
    """

    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # 必须自己实现 _stream：自带的版本只 yield content，
        # tool_calls 会被整个丢掉 —— 而 REPL 正是靠流里的工具调用
        # 才知道「模型这一步调了哪个工具」。
        msg = next(self.messages)

        # 先把工具调用拆成 chunk 吐出去。
        # index 是同一轮里多个工具调用的序号，不能省，
        # 否则 LangChain 拼不回来。
        for i, tc in enumerate(getattr(msg, "tool_calls", None) or []):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    id=msg.id,
                    tool_call_chunks=[
                        {
                            "name": tc["name"],
                            # 流式协议里 args 是 JSON 文本，不是 dict。
                            "args": json.dumps(tc.get("args", {}), ensure_ascii=False),
                            "id": tc.get("id", "c1"),
                            "index": i,
                            "type": "tool_call_chunk",
                        }
                    ],
                )
            )
        # 再按空白切分正文，模拟真实模型一个 token 一个 token 地吐。
        # re.split 带捕获组，空白本身也会被保留，
        # 拼回去才是完整的原文。
        if msg.content:
            for token in re.split(r"(\s)", msg.content):
                yield ChatGenerationChunk(message=AIMessageChunk(content=token, id=msg.id))


def make_session(*responses: AIMessage) -> ReplSession:
    """造一个不连真模型的 REPL 会话。

    用 __new__ 绕开 __init__，
    因为 __init__ 会调 get_agent() 拿到真模型单例。
    """
    session = ReplSession.__new__(ReplSession)  # 绕开 get_agent，避免真连模型
    session.agent = build_agent(model=FakeModel(messages=iter(responses)), checkpointer=InMemorySaver())
    session.thread_id = "test-thread"
    session.allow_write = False
    session.show_thinking = False
    session._think = _ThinkFilter()
    return session


# ---------- 思考内容过滤 ----------

# MiniMax 这类推理模型会把思考过程混在 content 里返回，
# 终端默认要滤掉，否则会被大段自说自话刷屏。


def test_think_filter_removes_think_block():
    """完整的 <think> 块要被整段拿掉，只留下正文。"""
    f = _ThinkFilter()
    assert f.feed("<think>思考中</think>答案") == "答案"


def test_think_filter_handles_split_chunks():
    """闭标签还没到时也不能漏出思考内容。"""
    f = _ThinkFilter()
    assert f.feed("<think>我在") == ""
    assert f.feed("思考</think>正文") == "正文"


def test_think_filter_keeps_plain_text():
    """普通文本要原样透传，过滤器不能误伤。"""
    f = _ThinkFilter()
    assert f.feed("普通") == "普通"
    assert f.feed("回答") == "回答"


# ---------- 输出代理 ----------

# 中间件的统计、notify 的输出都是直接 print 的，
# 流式回答过程中混进来会粘在回答末尾。
# _LineGuard 保证它们另起一行。


def test_line_guard_keeps_third_party_print_on_own_line():
    """第三方 print 要另起一行，不能粘在流式回答的末尾。"""
    buf = io.StringIO()
    guard = _LineGuard(buf)
    guard.raw("回答正文")  # 模拟流式输出，末尾没有换行
    guard.write("中间件统计\n")  # 第三方 print
    assert buf.getvalue() == "回答正文\n中间件统计\n"


def test_line_guard_does_not_add_extra_blank_line():
    """自己用 line() 写的整行后面跟第三方 print，不该多出一个空行。"""
    buf = io.StringIO()
    guard = _LineGuard(buf)
    guard.line("工具调用\n")
    guard.write("统计\n")
    assert buf.getvalue() == "工具调用\n统计\n"


# ---------- 斜杠命令 ----------

# 斜杠命令是终端的控制面：换会话、开关写权限、开关思考显示、退出。
# 约定：command() 返回 False 表示要退出主循环。


def test_write_command_toggles_permission(capsys):
    """/write on|off 能来回切换写权限。"""
    session = make_session()
    assert session.command("/write on") is True
    assert session.allow_write is True
    session.command("/write off")
    assert session.allow_write is False


def test_new_command_switches_thread():
    """/new 要换一个 thread_id，也就是开一个上下文为空的会话。"""
    session = make_session()
    old = session.thread_id
    session.command("/new")
    assert session.thread_id != old


def test_exit_command_returns_false():
    """/exit 返回 False，主循环据此退出。"""
    session = make_session()
    assert session.command("/exit") is False


def test_unknown_command_does_not_crash(capsys):
    """未知命令只提示一句，终端不能因此退出或报错。"""
    session = make_session()
    assert session.command("/nope") is True
    assert "未知命令" in capsys.readouterr().out


# ---------- 一轮对话 ----------

# send() 是 REPL 的核心：把一条指令发给 Agent，再把回答打出来。
# 守住四件事：回答要显示、工具调用要显示、报错要兜住、思考内容按开关处理。


def test_send_prints_answer(capsys):
    """模型的回答要出现在终端输出里。"""
    session = make_session(AIMessage(content="你好，我是助手"))
    session.send("你是谁")
    out = capsys.readouterr().out
    assert "你好，我是助手" in out


def test_send_reports_tool_call(capsys):
    """模型调了工具，终端要把工具名显示出来，再显示最终回答。"""
    session = make_session(
        AIMessage(
            content="",
            tool_calls=[{"name": "list_files", "args": {}, "id": "c1", "type": "tool_call"}],
        ),
        AIMessage(content="列完了"),
    )
    session.send("列出文件")
    out = capsys.readouterr().out
    assert "list_files" in out
    assert "列完了" in out


def test_send_survives_model_error(capsys):
    """模型一直报错时，REPL 要把错误显示出来而不是崩掉。

    注意：项目的 RETRY 中间件会把重试耗尽的错误转成一条 AIMessage 返回，
    所以这里断言的是错误信息出现在输出里。
    """

    class BoomModel(FakeModel):
        # 走的是 _stream（agent 用流式调用），所以要在这里抛
        def _stream(self, *args, **kwargs):
            raise RuntimeError("模型炸了")
            yield  # pragma: no cover - 让它成为生成器

    session = make_session()
    session.agent = build_agent(model=BoomModel(messages=iter([])), checkpointer=InMemorySaver())
    session.send("随便说点什么")
    out = capsys.readouterr().out
    assert "模型炸了" in out


def test_send_survives_unexpected_error(capsys):
    """流本身炸了（比如代理层异常）时，REPL 自己兜住。"""

    class BrokenAgent:
        def stream(self, *args, **kwargs):
            raise RuntimeError("流崩了")

    session = make_session()
    session.agent = BrokenAgent()
    session.send("随便说说")
    out = capsys.readouterr().out
    assert "调用失败" in out


def test_send_hides_thinking_by_default(capsys):
    """默认不显示 <think> 思考过程。"""
    session = make_session(AIMessage(content="<think>内部思考</think>最终答案"))
    session.send("思考一下")
    out = capsys.readouterr().out
    assert "内部思考" not in out
    assert "最终答案" in out


def test_send_shows_thinking_when_enabled(capsys):
    """开了 /think on 之后，思考过程要原样显示。"""
    session = make_session(AIMessage(content="<think>内部思考</think>最终答案"))
    session.show_thinking = True
    session.send("思考一下")
    out = capsys.readouterr().out
    assert "内部思考" in out
