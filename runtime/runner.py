"""Agent 调用封装。调度器和 HTTP 服务都走这里，保证行为一致。"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import HumanMessage

from agent.config import RUNTIME
from agent.context import AgentContext, Trigger
from agent.graph import build_agent

# 复用同一个 agent 实例，保留 checkpointer 里的记忆
_AGENT = None


def get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT


def run_task(
    instruction: str,
    task_name: str | None = None,
    trigger: Trigger = "cli",
    allow_write: bool = False,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """执行一次任务。

    thread_id 相同则共享记忆（对应课件 09 章短期记忆）。
    """
    agent = get_agent()
    ctx = AgentContext(
        task_name=task_name or instruction[:30],
        trigger=trigger,
        allow_write=allow_write,
        notify_channel=RUNTIME.notify_channel,
    )
    config = {"configurable": {"thread_id": thread_id or f"task-{uuid.uuid4().hex[:8]}"}}
    # context 是 invoke 的独立参数，不在 config 里
    result = agent.invoke(
        {"messages": [HumanMessage(content=instruction)]}, config, context=ctx
    )
    return {
        "task_name": ctx.task_name,
        "thread_id": config["configurable"]["thread_id"],
        "output": result["messages"][-1].content,
        "messages": len(result["messages"]),
    }
