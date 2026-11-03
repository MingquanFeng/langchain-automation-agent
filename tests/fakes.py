"""
测试用的假模型。

为什么单独放一个文件：
流式路径（`agent.stream()`）只有这一个假模型能用 ——
`GenericFakeChatModel` 自带的 `_stream` 只吐 content，
会把 `tool_calls` 整个丢掉，
而流式路径恰恰要靠 tool_calls 才知道「模型这一步调了哪个工具」。

原先它住在 `tests/test_repl.py` 里，
那个文件是交互终端的测试，随终端一起删了。
但桌面版的事件流测试马上要用它，
所以搬到这里 —— 与被测对象无关，纯粹是测试替身。

注意：
`tests/test_e2e.py` 里的 `ToolCapableFakeModel` 是另一个东西，
它服务的是非流式的工具调用链，两者别混。
"""

from __future__ import annotations

# json：把工具参数序列化成字符串（流式协议里 args 是 JSON 文本）。
# re：按空白切分正文，模拟真实模型一个 token 一个 token 地吐。
import json
import re

# 第三方库：
# GenericFakeChatModel：可编排输出的假模型。
# AIMessageChunk：流式增量形态。
# ChatGenerationChunk：流式协议里的最小单位。
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk


class FakeModel(GenericFakeChatModel):
    """支持 bind_tools 与流式工具调用的假模型。

    GenericFakeChatModel 自带的 _stream 只吐 content，会把 tool_calls 整个丢掉，
    所以这里自己实现 _stream —— agent 走的就是流式路径。
    """

    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # 必须自己实现 _stream：自带的版本只 yield content，
        # tool_calls 会被整个丢掉 —— 而流式消费者正是靠流里的工具调用
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
