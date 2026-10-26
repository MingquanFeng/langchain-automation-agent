"""自定义中间件。

对应课件 08 章第 5 节（自定义中间件）与 09 章 4.1 节（中间件中访问上下文）。

两种写法都保留：
- 钩子式（before_model / after_model）：拿得到 state 和 runtime
- 装饰器式（@wrap_model_call / @wrap_tool_call）：能包裹住下游调用
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import AIMessage, SystemMessage
from langgraph.runtime import Runtime

from ..context import AgentContext


def _ctx(runtime: Runtime) -> AgentContext:
    """取运行时上下文。

    context 是 invoke 的可选参数，不传时 runtime.context 为 None。
    这里退化成默认上下文，避免中间件把整个调用打挂。
    """
    ctx = runtime.context
    return ctx if isinstance(ctx, AgentContext) else AgentContext()


class TaskGateMiddleware(AgentMiddleware):
    """任务闸门：只读模式下拦截写操作。

    靠 after_model 检查本轮是否产生了写工具的调用，有则跳掉。
    """

    WRITE_TOOLS = {"write_file", "append_file", "save_json", "notify", "notify_summary"}

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        ctx = _ctx(runtime)
        if ctx.allow_write:
            return None

        called = {
            tc.get("name")
            for msg in state.get("messages", [])[-3:]
            if isinstance(msg, AIMessage)
            for tc in (msg.tool_calls or [])
        }
        blocked = called & self.WRITE_TOOLS
        if not blocked:
            return None

        return {
            "messages": [
                AIMessage(
                    content=f"当前任务为只读模式，已阻止以下写操作：{', '.join(sorted(blocked))}"
                )
            ]
        }


class ContextInjectionMiddleware(AgentMiddleware):
    """动态提示词：把运行时上下文注入系统提示词。

    对应课件 09 章 4.1 节。区别是课件用 before_model 提前退出，
    这里用 wrap_model_call 改写请求的 system prompt，不中断流程。
    """

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        ctx = _ctx(runtime)
        if not ctx.allow_write:
            return {
                "messages": [
                    AIMessage(content="（提示：本轮为只读模式，任何写操作都会被拒绝。）")
                ]
            }
        return None

    def wrap_model_call(self, request, handler):
        ctx = _ctx(request.runtime)
        extra = (
            f"\n\n[运行时上下文]\n"
            f"任务名：{ctx.task_name}\n"
            f"触发方式：{ctx.trigger}\n"
            f"写权限：{'允许' if ctx.allow_write else '禁止'}"
        )
        # system_prompt 只是只读 property，直接赋值已废弃；改用 override 造新 request
        base = request.system_message
        content = base.text if base is not None else ""
        new_request = request.override(
            system_message=SystemMessage(content=(content or "") + extra)
        )
        return handler(new_request)


class TimingMiddleware(AgentMiddleware):
    """记录本轮耗时与模型调用次数，写进日志。"""

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        ctx = _ctx(runtime)
        usage = [
            m.usage_metadata
            for m in state.get("messages", [])
            if isinstance(m, AIMessage) and m.usage_metadata
        ]
        total_tokens = sum(u.get("total_tokens", 0) for u in usage)
        print(
            f"[{ctx.task_name}] 模型调用 {len(usage)} 次，"
            f"累计 token {total_tokens}，消息 {len(state.get('messages', []))} 条"
        )
        return None


# create_agent 接收中间件实例，这里集中构造
CUSTOM_MIDDLEWARE = [
    TaskGateMiddleware(),
    ContextInjectionMiddleware(),
    TimingMiddleware(),
]
