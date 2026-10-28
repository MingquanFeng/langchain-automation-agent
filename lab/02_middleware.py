"""
实验 2：中间件怎么改变 Agent 的行为。

这个实验想让人看清什么：
中间件不是「装饰」，它真的能改变 Agent 的行为结果。
同一份模型脚本、同一句话，只改一个字段（AgentContext.allow_write），
一次写操作真的落盘了，另一次被完整拦下 ——
而且模型拿到的仍是一条合法的 ToolMessage，agent 循环没有崩。

怎么做到不花钱：
用只会说「写文件」的假模型跑两遍，
一遍 allow_write=True，一遍 allow_write=False，对比两次的结果。

对照课件：
- 08 章（中间件）：中间件在 agent 循环里的位置与钩子。
- 09 章 4.1 节（context）：运行时上下文怎么传到中间件。

用法：
    $PY lab/02_middleware.py
"""

from __future__ import annotations

# 标准库：sys 处理导入路径，Path 定位文件。
import sys
from pathlib import Path

# 把项目根目录插到 sys.path 最前面。
# 这样不管从哪个目录跑本脚本，都能 import 到 agent 包。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 第三方库：
# GenericFakeChatModel：可编排输出的假模型。
# AIMessage / HumanMessage：agent 循环里的两种消息。
# InMemorySaver：进程内存版 Checkpointer，实验不留痕。
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# 项目内部：
# AgentContext：单次调用的运行时上下文，allow_write 就在里面。
# build_agent：装配 Agent 的唯一入口。
from agent.context import AgentContext
from agent.graph import build_agent


# ============================================================
# 1. 一个只会说「写文件」的假模型
# ============================================================


class FakeModel(GenericFakeChatModel):
    """只会说「写文件」这一句话的模型。

    bind_tools 必须实现：BaseChatModel.bind_tools 默认抛 NotImplementedError。
    """

    def bind_tools(self, tools, **kwargs):
        return self


def write_call(path: str) -> AIMessage:
    """构造「模型决定调用 write_file」的那一条消息。

    两次实验都用同一个模型脚本，只是 path 不同，
    这样对比出来的差异才能全部归因到 allow_write 上。
    """
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


# ============================================================
# 2. 跑一次 agent
# ============================================================


def run_once(allow_write: bool, path: str) -> tuple[bool, list]:
    """跑一轮 agent，返回（文件有没有真的产生, 最终消息列表）。

    注意：
    两次调用用了不同的 thread_id（f"lab-02-{allow_write}"），
    否则第二次会继承第一次的历史，两边消息条数就没法对比了。
    """
    model = FakeModel(messages=iter([write_call(path), AIMessage(content="写好了。")]))
    agent = build_agent(model=model, checkpointer=InMemorySaver())
    ctx = AgentContext(task_name="lab-02", trigger="lab", allow_write=allow_write)

    result = agent.invoke(
        {"messages": [HumanMessage(content="写一个文件")]},
        {"configurable": {"thread_id": f"lab-02-{allow_write}"}},
        context=ctx,
    )
    # 除了消息序列，还要看文件到底有没有落盘 ——
    # 这是判断「写操作有没有真的发生」的硬证据。
    target = Path(__file__).resolve().parent.parent / "data" / path
    return target.exists(), result["messages"]


# ============================================================
# 3. 把结果逐条打出来
# ============================================================


def show(label: str, exists: bool, messages: list) -> None:
    """把一轮跑完的消息序列逐条打出来。

    要重点看的是场景 B 里的那条 ToolMessage：
    它是闸门给的「拒绝结果」，工具压根没跑。
    """
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


# ============================================================
# 4. 两个场景对比
# ============================================================


def main() -> None:
    print("同一个模型脚本，只改 AgentContext.allow_write。\n")

    allowed_path = "lab_demo/放行.txt"
    blocked_path = "lab_demo/拦截.txt"

    # 场景 A：放行写操作，文件应该真的出现。
    exists, messages = run_once(True, allowed_path)
    show("场景 A：allow_write=True（放行）", exists, messages)

    # 场景 B：只读模式，闸门在 after_model 阶段就拦下，
    # 文件不该出现，但历史里仍要留下一条 ToolMessage。
    exists, messages = run_once(False, blocked_path)
    show("场景 B：allow_write=False（TaskGateMiddleware 拦截）", exists, messages)

    print(
        "\n观察重点：\n"
        "1. 场景 B 里模型照样请求了 write_file，但文件最终不存在 —— 写操作确实没发生，\n"
        "   中间件在 after_model 阶段就拦下了。\n"
        "2. 场景 B 第 4 条 ToolMessage 是闸门给的「拒绝结果」，不是工具真的执行了\n"
        "   —— 工具压根没跑，内容是「已阻止写操作」。\n"
        "3. 这条 ToolMessage 是必须的：每个 tool_call 都要有对应的工具结果。少了它，\n"
        "   历史里就留着悬空的 tool_call，下一轮请求会被 OpenAI / MiniMax 兼容接口\n"
        "   直接 400（tool call result does not follow tool call）。\n"
        "4. 因为那个 tool_call 已经被「回答」了，agent 图的路由判定为人工注入的工具消息，\n"
        "   会跳回模型节点重新回答 —— 于是最后一条是模型改用文字给出的结论。\n"
        "5. 想知道中间件卡在哪一层，去看 agent/middleware/custom.py 的 TaskGateMiddleware.after_model。"
    )


if __name__ == "__main__":
    main()
