
"""
Agent 调用封装。调度器和 HTTP 服务都走这里，保证行为一致。

对应课件第 09 章（courseware/尚硅谷-09-上下文与记忆.pdf）：记忆与状态持久化。

主要职责：
1. 维护进程内唯一的 Agent 单例。
2. 把「指令 + 上下文 + 会话 ID」组装成一次 invoke。
3. 把执行结果整理成统一的字典返回。

设计要点：
只有这一个入口能调用 Agent。
调度器、HTTP 服务、REPL 都从这里走，
这样三者触发出来的行为（上下文构造、thread_id 规则、
写权限默认值）完全一致，不会出现「某个入口忘了传 context」的情况。
"""

from __future__ import annotations

# uuid：标准库，用来生成随机的会话 ID。
import uuid

# Any：typing 里的任意类型标注。
from typing import Any

# HumanMessage：LangChain 消息类型，代表用户输入。
from langchain_core.messages import HumanMessage

# RUNTIME：应用运行配置，这里主要取通知渠道。
from agent.config import RUNTIME

# AgentContext：单次调用的运行时上下文结构。
# Trigger：触发来源的字面量类型（schedule/api/cli/lab）。
from agent.context import AgentContext, Trigger

# build_agent：Agent 装配入口，见 agent/graph.py。
from agent.graph import build_agent

# 复用同一个 agent 实例，保留 checkpointer 里的记忆
#
# 为什么必须复用：
# build_agent() 内部会创建 Checkpointer，
# SQLite 后端下它是连着数据库的。
# 每建一个 Agent 就多开一条连接、多建一套中间件，
# 而且不同 Agent 的 thread_id 记忆是隔离的，
# 换 Agent 等于把之前的会话记忆丢掉。
#
# 所以这里用模块级变量做惰性单例：
# 第一次调用 get_agent() 时才真正创建。
_AGENT = None


# ============================================================
# 1. 获取 Agent 单例
# ============================================================

def get_agent():
    """
    获取进程内唯一的 Agent 实例。

    返回：
        build_agent() 创建的 Agent。

    注意：
    惰性创建：模块导入时不建 Agent，
    避免仅仅 import runtime 就连上数据库、初始化模型客户端。

    线程安全：
    这里没有加锁。
    并发首次调用时理论上可能创建出两个实例，
    但因为赋值是原子的，最终持有的仍是同一个对象，
    丢掉的只是一个多余的实例，不影响正确性。
    """
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT


# ============================================================
# 2. 执行一次任务
# ============================================================

def run_task(
    instruction: str,
    task_name: str | None = None,
    trigger: Trigger = "cli",
    allow_write: bool = False,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """执行一次任务。

    参数：
        instruction：
            给 Agent 的自然语言指令。

        task_name：
            任务名，用于日志和通知标题。
            不传时截取指令前 30 个字符兜底。

        trigger：
            触发来源，见 agent/context.py 的 Trigger 定义。
            调度器传 schedule，HTTP 传 api，命令行传 cli。

        allow_write：
            是否允许写操作。
            默认 False，即默认只读。
            写权限最终由 TaskGateMiddleware 强制检查。

        thread_id：
            会话 ID。
            相同则共享记忆（对应课件 09 章短期记忆，见 courseware/尚硅谷-09-上下文与记忆.pdf）。
            不传时生成一个一次性的随机 ID。

    返回：
        包含任务名、thread_id、最终输出、消息条数的字典。

    注意：
    thread_id 的命名规则决定了记忆的隔离边界：
        cron-{任务名}：定时任务跨天保留记忆。
        task-{随机}：一次性任务，不留痕。
    """
    agent = get_agent()

    # --------------------------------------------------------
    # 2.1 构造运行时上下文
    # --------------------------------------------------------

    # notify_channel 从全局配置取，
    # 保证所有入口的通知走同一个渠道。
    ctx = AgentContext(
        task_name=task_name or instruction[:30],
        trigger=trigger,
        allow_write=allow_write,
        notify_channel=RUNTIME.notify_channel,
    )

    # --------------------------------------------------------
    # 2.2 构造 config
    # --------------------------------------------------------

    # thread_id 必须放在 config["configurable"] 里，
    # 这是 Checkpointer 识别会话的依据。
    #
    # 不传时用 task-{8位随机} 生成一次性的会话 ID，
    # 保证不同任务之间不会串记忆。
    config = {"configurable": {"thread_id": thread_id or f"task-{uuid.uuid4().hex[:8]}"}}

    # --------------------------------------------------------
    # 2.3 调用 Agent
    # --------------------------------------------------------

    # context 是 invoke 的独立参数，不在 config 里
    #
    # 这是最容易踩的坑之一：
    #
    # agent.invoke(msgs, {"configurable": {...}}, context=ctx)   # ✅
    # agent.invoke(msgs, {"configurable": {..., "context": ctx}})  # ❌
    #
    # 写错的话 invoke 不报错，
    # 但 runtime.context 会是 None，
    # 中间件里的 _ctx() 兜底会走到默认值，
    # 表现为「任务名和写权限都丢了」这种很难查的问题。
    result = agent.invoke(
        {"messages": [HumanMessage(content=instruction)]}, config, context=ctx
    )

    # --------------------------------------------------------
    # 2.4 整理返回值
    # --------------------------------------------------------

    # output 取最后一条消息的 content。
    # Agent 跑完后最后一条一定是 AIMessage（模型的最终回答）。
    return {
        "task_name": ctx.task_name,
        "thread_id": config["configurable"]["thread_id"],
        "output": result["messages"][-1].content,
        # 消息条数：一条用户消息 + 每轮的工具消息 + 最终回答。
        # 可以用来粗略判断这次任务来回了多少轮。
        "messages": len(result["messages"]),
    }
