"""实验 1：看清一次 Agent 调用的完整循环。

不花 API 钱：用脚本化的假模型编排一次
「调工具 → 观察结果 → 再调工具 → 总结」的完整链路，
把每一步的中间状态打印出来。

对照课件：04 章（Message）、05 章（Tools）、07 章（create_agent）

用法：
    $PY lab/01_agent_loop.py          # 假模型，不需要 API key
    $PY lab/01_agent_loop.py --live   # 真实模型，需要 .env 里的 LLM_API_KEY
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import PrivateAttr

from agent.context import AgentContext
from agent.graph import build_agent


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    """构造一条带工具调用的 AI 消息（注意 type 字段不能省）。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


class TeachingFakeModel(GenericFakeChatModel):
    """每被调用一次，就打印「模型此刻看到的完整对话」。

    这是理解 agent 循环最有效的观察点：
    模型输出的 tool_calls 会变成 ToolMessage 回灌进 messages，
    模型下一轮就能看到工具结果 —— 这个来回就是 agent 循环的全部。
    """

    # BaseChatModel 是 pydantic 模型，普通属性加不进去，要用 PrivateAttr
    _round: int = PrivateAttr(default=0)

    def __init__(self, responses, **kw):
        super().__init__(messages=iter(responses), **kw)

    def bind_tools(self, tools, **kwargs):
        # BaseChatModel.bind_tools 默认抛 NotImplementedError，
        # 任何带工具的 agent 都会走到这里，所以必须实现。
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._round += 1
        print(f"\n{'=' * 72}")
        print(f"第 {self._round} 轮 —— 模型被调用，此刻它看到的对话：")
        print("=" * 72)
        for m in messages:
            kind = type(m).__name__
            if isinstance(m, AIMessage) and m.tool_calls:
                calls = ", ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
                print(f"  [{kind}] 决定调用工具 → {calls}")
            elif kind == "SystemMessage":
                # 完整打印：中间件注入的「运行时上下文」就挂在系统提示词末尾
                print(f"  [{kind}] {m.content}")
            else:
                text = str(m.content).replace("\n", " ")[:150]
                print(f"  [{kind}] {text}")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def show_final(result: dict) -> None:
    print(f"\n{'=' * 72}")
    print("循环结束，最终消息序列（这就是 agent 的完整思考轨迹）：")
    print("=" * 72)
    for i, m in enumerate(result["messages"], 1):
        kind = type(m).__name__
        if isinstance(m, AIMessage) and m.tool_calls:
            calls = ", ".join(c["name"] for c in m.tool_calls)
            print(f"{i}. [{kind}] 工具调用 → {calls}")
        elif kind == "ToolMessage":
            print(f"{i}. [{kind}] 工具返回 → {str(m.content)[:80]}")
        else:
            print(f"{i}. [{kind}] {str(m.content)[:80]}")


def run_fake() -> None:
    # 预先编排模型每一轮要输出什么。真实模型是「自己决定」，这里是「按剧本演」。
    responses = [
        tool_call("list_files", {"pattern": "*.md"}, "c1"),
        tool_call(
            "write_file",
            {"path": "lab_demo/笔记.md", "content": "# 实验一\n\n- agent 循环看得见\n"},
            "c2",
        ),
        AIMessage(content="已在 lab_demo/笔记.md 写入两条要点，文件写好了。"),
    ]

    model = TeachingFakeModel(responses)
    agent = build_agent(model=model, checkpointer=InMemorySaver())
    ctx = AgentContext(task_name="lab-01", trigger="lab", allow_write=True)

    result = agent.invoke(
        {"messages": [HumanMessage(content="在数据目录下写一份实验笔记")]},
        {"configurable": {"thread_id": "lab-01"}},
        context=ctx,
    )
    show_final(result)

    written = Path(__file__).resolve().parent.parent / "data" / "lab_demo" / "笔记.md"
    print(f"\n文件真的落盘了吗？ {written.exists()} → {written}")
    print(
        "\n注意上面那行「模型调用 0 次，累计 token 0」："
        "TimingMiddleware 统计的是 AIMessage 上的 usage_metadata，"
        "只有真实模型 API 才会返回，假模型没有 —— 换成 --live 就能看到真实数字。"
    )


def run_live() -> None:
    """走真实模型。需要在 .env 里配好 LLM_API_KEY。"""
    from agent.config import MODEL

    if not MODEL.api_key:
        sys.exit("没有配置 LLM_API_KEY。先 `cp .env.example .env` 并填上 key。")

    agent = build_agent(checkpointer=InMemorySaver())
    ctx = AgentContext(task_name="lab-01-live", trigger="lab", allow_write=True)
    result = agent.invoke(
        {"messages": [HumanMessage(content="在数据目录下写一份实验笔记，说明 agent 循环是怎么回事")]},
        {"configurable": {"thread_id": "lab-01-live"}},
        context=ctx,
    )
    show_final(result)


if __name__ == "__main__":
    print("模式：真实模型" if "--live" in sys.argv else "模式：假模型（不调 API）")
    run_live() if "--live" in sys.argv else run_fake()
