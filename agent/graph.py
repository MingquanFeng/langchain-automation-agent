"""Agent 装配。

对应课件 07 章（create_agent）+ 08 章（中间件栈）+ 09 章（记忆）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.context_editing import ClearToolUsesEdit, ContextEditingMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore

from .config import MODEL, RUNTIME
from .context import AgentContext
from .middleware.custom import CUSTOM_MIDDLEWARE
from .tools import ALL_TOOLS

# 上下文超限时清理旧的工具输出，对应课件 08 章 ContextEditingMiddleware
# trigger=120000 表示输入 token 超过 12 万时开始清理，保留最近 3 条工具结果
CONTEXT_EDITING = ContextEditingMiddleware(
    edits=[ClearToolUsesEdit(trigger=120_000, keep=3, clear_tool_inputs=True)]
)

# 失败重试与故障转移：无人值守场景的兜底
RETRY = [
    ModelRetryMiddleware(max_retries=3, backoff_factor=2.0),
    ToolRetryMiddleware(max_retries=2, on_failure="continue"),
]

# 限次：防止无人值守时跑飞烧钱
LIMITS = [
    ModelCallLimitMiddleware(thread_limit=RUNTIME.max_model_calls),
    ToolCallLimitMiddleware(thread_limit=50),
]


def _build_checkpointer():
    """按配置选持久化后端。

    SqliteSaver 需要一个长连接，用 from_conn_string 会拿到上下文管理器
    （会在 with 块结束时关掉连接），所以直接持有一个 sqlite3 连接。
    """
    if RUNTIME.memory_backend == "sqlite":
        path = Path(RUNTIME.sqlite_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver
    return InMemorySaver()


def build_agent(model=None, tools=None, checkpointer=None, store=None):
    """组装一个带完整守卫层的 Agent。

    换模型、换工具、换记忆后端都从这里注入，不改动调用方。
    """
    middleware = [
        *LIMITS,
        *RETRY,
        TODO,
        CONTEXT_EDITING,
        *CUSTOM_MIDDLEWARE,
    ]

    return create_agent(
        model=model or MODEL.build(),
        tools=tools if tools is not None else ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        checkpointer=checkpointer if checkpointer is not None else _build_checkpointer(),
        store=store if store is not None else InMemoryStore(),
        context_schema=AgentContext,
        name="automation_agent",
    )


# 任务拆解中间件：多步任务的进度追踪（课件 08 章 2.4 节）
TODO = TodoListMiddleware()

SYSTEM_PROMPT = """你是一个自动化日常事务助手。

工作方式：
1. 先用 TodoList 把多步任务拆成清晰的子步骤，然后逐条推进。
2. 每一步做完就用 notify 或 notify_summary 把结果推给用户，不要攒到最后。
3. 通知里给结论，不要复述你的思考过程。
4. 需要读写本地文件时，路径都相对于数据目录。
5. 如果某一步失败，说明失败原因和已经尝试过的办法，不要静默跳过。
6. 缺少信息时按最合理的默认做法继续，并在结果里说明你用了什么默认值。
"""
