"""
实验 3：短期记忆与 thread_id 隔离。

这个实验想让人看清什么：
记忆不是存在 Agent 对象里，而是存在 Checkpointer 里、按 thread_id 分桶。
同一个 thread_id 继续往下聊，历史会被恢复出来；
换一个 thread_id 就是全新会话，一条旧消息都看不到。

怎么做到不花钱：
用假模型问同样的问题三遍，只改 thread_id，
对比每次结束后 messages 的条数。

对照课件：
- 09 章（上下文与记忆）：Checkpointer 与 thread_id 的关系。

用法：
    $PY lab/03_memory.py
"""

from __future__ import annotations

# 标准库：sys 处理导入路径，Path 定位项目根目录。
import sys
from pathlib import Path

# 把项目根目录插到 sys.path 最前面。
# 这样不管从哪个目录跑本脚本，都能 import 到 agent 包。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 第三方库：
# GenericFakeChatModel：可编排输出的假模型。
# AIMessage / HumanMessage：agent 循环里的两种消息。
# InMemorySaver：进程内存版 Checkpointer，重启即丢。
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# 项目内部：
# AgentContext：单次调用的运行时上下文。
# build_agent：装配 Agent 的唯一入口。
from agent.context import AgentContext
from agent.graph import build_agent


# ============================================================
# 1. 假模型：每次都回同一句话
# ============================================================


class FakeModel(GenericFakeChatModel):
    """只会重复同一句话的模型。

    模型回答什么跟实验无关 —— 这个实验只看消息条数，
    所以模型说得再单调也不影响观察。
    """

    def bind_tools(self, tools, **kwargs):
        return self


# ============================================================
# 2. 问一句话，数一数消息条数
# ============================================================


def ask(agent, thread_id: str, question: str) -> int:
    """问一句话，返回这次调用结束后 messages 的条数。

    同一个 agent 实例、同一个 checkpointer，
    只靠 thread_id 区分会话 —— 这就是记忆隔离的全部机制。
    """
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": thread_id}},
        context=AgentContext(task_name="lab-03", trigger="lab"),
    )
    return len(result["messages"])


# ============================================================
# 3. 对比输出
# ============================================================


def show(title: str, first: int, same: int, other: int) -> None:
    """把三次调用的消息条数摆在一起对比。

    要看的是差值：
    同一 thread 的第二条比第一条多出来的部分，
    就是上一轮历史被 checkpointer 恢复出来的证据。
    """
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


# ============================================================
# 4. 主流程
# ============================================================


def main() -> None:
    # 关键：checkpointer 必须显式传。传 None 不是「不要记忆」，
    # 而是走 graph.py 里的默认分支 —— 那会用 sqlite 落盘到 data/memory.sqlite。
    saver = InMemorySaver()
    agent = build_agent(
        model=FakeModel(messages=iter([AIMessage(content="收到。")] * 10)),
        checkpointer=saver,
    )

    # 三次调用，只有 thread_id 和问题在变，
    # 模型、checkpointer、agent 实例全都是同一个。
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
