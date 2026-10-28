"""
实验 1：看清一次 Agent 调用的完整循环。

这个实验想让人看清什么：
模型说「我要调 write_file」之后，工具到底跑没跑？
跑完的结果又是怎么回到模型手里的？
看清了就明白 —— 模型输出的 tool_calls 会被转成 ToolMessage 回灌进 messages，
模型下一轮就能看到工具结果。这个来回，就是 agent 循环的全部。

怎么做到不花钱：
用脚本化的假模型编排一次
「调工具 → 观察结果 → 再调工具 → 总结」的完整链路，
把每一步的中间状态都打印出来。

对照课件：
- 04 章（Message）：四种消息类型在循环里各占什么位置。
- 05 章（Tools）：tool_calls 怎么变成一次真实的工具执行。
- 07 章（create_agent）：create_agent 怎么把这一切串成一张图。

用法：
    $PY lab/01_agent_loop.py          # 假模型，不需要 API key
    $PY lab/01_agent_loop.py --live   # 真实模型，需要 .env 里的 LLM_API_KEY
"""

from __future__ import annotations

# 标准库：sys 处理命令行参数与导入路径，Path 定位文件。
import sys
from pathlib import Path

# 把项目根目录插到 sys.path 最前面。
# 这样不管从哪个目录跑本脚本，都能 import 到 agent 包。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 第三方库：
# GenericFakeChatModel：可编排输出的假模型，支持 tool_calls。
# AIMessage / HumanMessage：agent 循环里的两种消息。
# InMemorySaver：进程内存版 Checkpointer，实验不留痕。
# PrivateAttr：pydantic 私有属性装饰器，
#   用来给模型挂不属于 schema 的辅助状态（比如轮次计数）。
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import PrivateAttr

# 项目内部：
# AgentContext：单次调用的运行时上下文，写权限在里面。
# build_agent：装配 Agent 的唯一入口。
from agent.context import AgentContext
from agent.graph import build_agent


# ============================================================
# 1. 编排模型每一轮要说什么
# ============================================================

# 真实模型是「自己决定」调不调工具；实验里我们把它换成一个按剧本演的演员。


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    """构造一条带工具调用的 AI 消息（注意 type 字段不能省）。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


# ============================================================
# 2. 会打印「模型此刻看到的对话」的假模型
# ============================================================

# 这是整个实验的观察点所在：
# 每调一次模型，就把模型此刻看到的完整对话打出来。


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
        # 打印发生在真正生成之前，
        # 所以这里看到的是「模型这一轮收到的输入」，
        # 也就是上一轮工具执行完、结果回灌之后的状态。
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


# ============================================================
# 3. 打印最终的完整消息序列
# ============================================================

# 循环结束后再看一眼「留下了什么」：
# 完整的思考轨迹 = 模型决定的 + 工具返回的 + 模型总结的。


def show_final(result: dict) -> None:
    """把跑完之后的完整消息序列逐条打出来 —— 这就是 agent 的思考轨迹。

    要重点看的是中间那一段：
    模型的 tool_calls 后面紧跟着 ToolMessage（工具真的返回了），
    再往后才是模型的最终回答。
    """
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


# ============================================================
# 4. 假模型模式（不花钱）
# ============================================================


def run_fake() -> None:
    # 预先编排模型每一轮要输出什么。真实模型是「自己决定」，这里是「按剧本演」。
    #
    # 三轮分别是：列目录 → 写文件 → 总结。
    #
    # 为什么要绕这一圈：
    # 真实模型可能一句话就写完文件，看不到「工具结果回灌」这一步；
    # 这里故意让它先列再写，才能看清中间那次模型调用
    # 是带着 ToolMessage 进来的。
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

    # allow_write=True：这一轮放开写，闸门才不会拦掉 write_file。
    result = agent.invoke(
        {"messages": [HumanMessage(content="在数据目录下写一份实验笔记")]},
        {"configurable": {"thread_id": "lab-01"}},
        context=ctx,
    )
    show_final(result)

    # 工具返回的是「已写入」，但真正证明落盘的是文件本身存在。
    written = Path(__file__).resolve().parent.parent / "data" / "lab_demo" / "笔记.md"
    print(f"\n文件真的落盘了吗？ {written.exists()} → {written}")
    print(
        "\n注意上面那行「模型调用 0 次，累计 token 0」："
        "TimingMiddleware 统计的是 AIMessage 上的 usage_metadata，"
        "只有真实模型 API 才会返回，假模型没有 —— 换成 --live 就能看到真实数字。"
    )


# ============================================================
# 5. 真实模型模式（需要 API key）
# ============================================================


def run_live() -> None:
    """走真实模型。需要在 .env 里配好 LLM_API_KEY。

    跟假模型模式的区别：
    真实模型自己决定调不调工具、调几次，
    所以消息序列不会跟上面那份剧本一一对应，
    轮次也可能不一样。
    """
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
