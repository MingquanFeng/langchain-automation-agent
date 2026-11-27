"""为 LangChain 执行清单中间件补齐桌面事件。"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import ToolMessage
from langgraph.config import get_stream_writer


class ObservableTodoListMiddleware(TodoListMiddleware):
    """把内置中间件生成的 `write_todos` 错误回执转成桌面事件。

    LangChain 在 `after_model` 中拦截并发 `write_todos` 调用，直接向状态
    写入错误 ToolMessage。此路径不会进入 `wrap_tool_call`，所以事件桥接
    看不到错误。桌面执行清单需要 custom 流事件才能显示失败原因。
    """

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """保留内置校验，并转发它生成的错误 ToolMessage。"""
        update = super().after_model(state, runtime)
        self._emit_errors(update)
        return update

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """异步图执行同样需要转发并发调用错误。"""
        update = super().after_model(state, runtime)
        self._emit_errors(update)
        return update

    @staticmethod
    def _emit_errors(update: dict[str, Any] | None) -> None:
        """将内置校验生成的清单错误映射为客户端 `tool.error` 事件。"""
        if not update:
            return

        for message in update.get("messages", []):
            if not isinstance(message, ToolMessage) or message.status != "error":
                continue

            try:
                writer = get_stream_writer()
            except RuntimeError:
                # 图外直接调用中间件时没有事件流，不影响原有工具校验。
                return

            writer({
                "type": "tool.error",
                "name": "write_todos",
                "call_id": message.tool_call_id,
                "message": str(message.content),
            })
