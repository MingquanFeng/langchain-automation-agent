"""实验 2：中间件怎么改变 Agent 的行为。

同一份模型脚本，跑两次，唯一区别是 AgentContext.allow_write：
一次放行写操作，一次被 TaskGateMiddleware 拦住。

对照课件：08 章（中间件）、09 章 4.1 节（context）

用法：
    $PY lab/02_middleware.py
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
    """只会说「写文件」这一句话的模型。

    bind_tools 必须实现：BaseChatModel.bind_tools 默认抛 NotImplementedError。
    """

    def bind_tools(self, tools, **kwargs):
        return self


def write_call(path: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"path": path, "content": "闸门测试\n"},
                "id": "c1",
                "type": "tool_call",
            }
        ],
    )


def run_once(allow_write: bool, path: str) -> tuple[bool, list]:
    model = FakeModel(messages=iter([write_call(path), AIMessage(content="写好了。")]))
    agent = build_agent(model=model, checkpointer=InMemorySaver())
    ctx = AgentContext(task_name="lab-02", trigger="lab", allow_write=allow_write)

    result = agent.invoke(
        {"messages": [HumanMessage(content="写一个文件")]},
        {"configurable": {"thread_id": f"lab-02-{allow_write}"}},
        context=ctx,
    )
    target = Path(__file__).resolve().parent.parent / "data" / path
    return target.exists(), result["messages"]


def show(label: str, exists: bool, messages: list) -> None:
    print(f"\n{'=' * 72}")
    print(f"{label}")
    print("=" * 72)
    for i, m in enumerate(messages, 1):
        kind = type(m).__name__
        if isinstance(m, AIMessage) and m.tool_calls:
            print(f"{i}. [{kind}] 请求调用 → {[c['name'] for c in m.tool_calls]}")
        elif kind == "ToolMessage":
            print(f"{i}. [{kind}] 工具执行 → {str(m.content)[:60]}")
        else:
            print(f"{i}. [{kind}] {str(m.content)[:100]}")
    print(f"→ 文件最终存在吗？ {exists}")


def main() -> None:
    print("同一个模型脚本，只改 AgentContext.allow_write。\n")

    allowed_path = "lab_demo/放行.txt"
    blocked_path = "lab_demo/拦截.txt"

    exists, messages = run_once(True, allowed_path)
    show("场景 A：allow_write=True（放行）", exists, messages)

    exists, messages = run_once(False, blocked_path)
    show("场景 B：allow_write=False（TaskGateMiddleware 拦截）", exists, messages)

    print(
        "\n观察重点：\n"
        "1. 场景 B 里模型照样请求了 write_file，但没有任何 ToolMessage —— "
        "说明工具根本没被执行，中间件在 after_model 阶段就拦下了。\n"
        "2. 场景 B 末尾多了一条「已阻止」的说明消息，"
        "替代了模型原本的最终回答 —— 因为最后一条消息没有 tool_calls，agent 循环就结束了。\n"
        "3. 想知道中间件卡在哪一层，去看 agent/middleware/custom.py 的 TaskGateMiddleware.after_model。"
    )


if __name__ == "__main__":
    main()
