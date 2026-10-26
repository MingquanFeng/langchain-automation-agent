"""实验 3：短期记忆与 thread_id 隔离。

同一个模型脚本、同一句话，问两次。
差别只在 thread_id —— 看第二次调用时模型能看到多少历史。

对照课件：09 章（上下文与记忆）

用法：
    $PY lab/03_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.context import AgentContext
from agent.graph import build_agent


class FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def ask(agent, thread_id: str, question: str) -> int:
    """问一句话，返回这次调用结束后 messages 的条数。"""
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": thread_id}},
        context=AgentContext(task_name="lab-03", trigger="lab"),
    )
    return len(result["messages"])


def show(title: str, first: int, same: int, other: int) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print("=" * 72)
    print(f"第一次调用结束后，       messages 共 {first} 条")
    print(f"同一 thread_id 第二次后，  messages 共 {same} 条")
    print(f"换一个 thread_id 调用后，  messages 共 {other} 条")
    print(
        f"\n结论：同 thread 的第二次比第一次多 {same - first} 条 —— "
        "多出来的就是上一轮的历史被 checkpointer 恢复出来了。"
    )
    print(
        f"换 thread 后只有 {other} 条 —— 新会话看不到旧会话的任何消息。"
        if other < same
        else f"换 thread 后仍有 {other} 条 —— 说明 thread_id 没起到隔离作用。"
    )


def main() -> None:
    # 关键：checkpointer 必须显式传。传 None 不是「不要记忆」，
    # 而是走 graph.py 里的默认分支 —— 那会用 sqlite 落盘到 data/memory.sqlite。
    saver = InMemorySaver()
    agent = build_agent(
        model=FakeModel(messages=iter([AIMessage(content="收到。")] * 10)),
        checkpointer=saver,
    )

    first = ask(agent, "会话A", "我叫张三")
    same = ask(agent, "会话A", "我叫什么")
    other = ask(agent, "会话B", "我叫什么")

    show("同一个 checkpointer，两个 thread", first, same, other)

    print(
        "\n延伸：把 checkpointer 换成 build_agent() 默认的 sqlite，"
        "重启进程后同一个 thread_id 仍能恢复历史；"
        "换成 InMemorySaver（当前脚本用的），重启即丢。\n"
        "验证方式见 LEARNING.md 主线 4 的练习 3。"
    )
    print(f"\n（第一次调用结束时 messages 是 {first} 条）")


if __name__ == "__main__":
    main()
