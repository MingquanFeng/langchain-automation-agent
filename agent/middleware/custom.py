"""
自定义中间件。

对应课件：
- 第 08 章第 5 节（courseware/尚硅谷-08-中间件.pdf）：自定义中间件。
- 第 09 章 4.1 节（courseware/尚硅谷-09-上下文与记忆.pdf）：在中间件中访问运行时上下文。

主要职责：
1. 只读闸门：只读模式下拦下写工具，并让模型改用文字回答。
2. 上下文注入：把任务名、触发来源、写权限写进系统提示词。
3. 耗时统计：一轮结束后打印模型调用次数与 token 用量。

两种写法都保留：
- 钩子式（before_model / after_model）：拿得到 state 和 runtime。
- 装饰器式（@wrap_model_call）：能包裹住下游调用，改写请求再放行。

注意：
    create_agent 收的是中间件「实例」，不是类。
    所以 CUSTOM_MIDDLEWARE 里放的是 TaskGateMiddleware() 这种实例。
"""

from __future__ import annotations

# Any：typing 的任意类型标注，用于描述钩子的返回值。
from typing import Any

# AgentMiddleware：所有自定义中间件的基类。
# AgentState：钩子拿到的状态对象，state["messages"] 是消息列表。
from langchain.agents.middleware import AgentMiddleware, AgentState

# AIMessage：模型的输出消息。
# SystemMessage：系统提示词消息。
# ToolMessage：工具执行结果消息。
from langchain.messages import AIMessage, SystemMessage, ToolMessage

# Runtime：LangGraph 的运行时对象，
# runtime.context 就是 invoke 时传进来的 AgentContext。
from langgraph.runtime import Runtime

# AgentContext：单次调用的运行时上下文结构，见 agent/context.py。
from ..context import AgentContext


# ============================================================
# 1. 取运行时上下文的兜底函数
# ============================================================

def _ctx(runtime: Runtime) -> AgentContext:
    """
    取运行时上下文，取不到就退化成默认上下文。

    为什么必须有这个兜底：
        context 是 invoke 的可选参数，不传时 runtime.context 就是 None。
        中间件里直接用 runtime.context，一旦为 None 就会抛 AttributeError，
        把整个调用打挂。

    为什么判断 isinstance 而不是只判 None：
        调用方理论上可以传任意对象当 context，
        不是 AgentContext 就说明类型不对，同样退化成默认值更安全。

    注意：
        新增中间件钩子时一律用 _ctx(runtime)，
        不要直接访问 runtime.context。
    """
    ctx = runtime.context
    return ctx if isinstance(ctx, AgentContext) else AgentContext()


# ============================================================
# 2. 只读闸门
# ============================================================

class TaskGateMiddleware(AgentMiddleware):
    """
    任务闸门：只读模式下拦截写操作。

    靠 after_model 检查本轮是否产生了写工具的调用，有则拦下。

    拦截时有两个讲究：
    1. 每个被拦下的 tool_call 都要补一条对应 tool_call_id 的 ToolMessage。
       少了它，消息历史里就留着没有结果的 tool_call，OpenAI / MiniMax 这类
       兼容接口下一轮请求直接 400（tool call result does not follow tool call）。
    2. 只补 ToolMessage、不补收尾消息。agent 图的路由是看最后一条 AIMessage：
       它的 tool_call 都已有 ToolMessage 回应时，判定为"人工注入的工具消息"，
       跳回模型节点重新回答 —— 于是 agent 会改用文字给结论，而不是直接结束。
       写工具本身不会执行（pending tool call 是空的）。
    """

    # WRITE_TOOLS：算「写操作」的工具名集合。
    #
    # 注意 notify 也在里面：
    # 发消息本身就是副作用，所以只读模式下同样要拦。
    #
    # 新增写类工具时，要同步更新这个集合，
    # 否则新工具会绕过只读闸门。
    WRITE_TOOLS = {"write_file", "append_file", "save_json", "notify", "notify_summary"}

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        模型这一轮输出之后、工具执行之前，检查有没有写操作。

        返回 None 表示放行，什么都不做。
        返回字典表示往消息列表里追加内容。
        """
        ctx = _ctx(runtime)
        if ctx.allow_write:
            return None

        # 只看最近 3 条消息：中间件自己注入的提示消息会夹在中间，
        # 但模型这轮的 tool_calls 一定在最后一条 AIMessage 上。
        blocked = [
            tc
            for msg in state.get("messages", [])[-3:]
            if isinstance(msg, AIMessage)
            for tc in (msg.tool_calls or [])
            if tc.get("name") in self.WRITE_TOOLS
        ]
        if not blocked:
            return None

        # 只在提示文字里列出工具名，不列出每个调用参数，
        # 免得参数里的大段内容又灌进上下文。
        names = ", ".join(sorted({tc["name"] for tc in blocked}))

        # 逐个被拦的 tool_call 补一条 ToolMessage。
        #
        # tool_call_id 必须与原始调用对上，
        # 这是协议要求：每个 tool_call 都要有对应的工具结果。
        #
        # status="error"：告诉下游这是一次失败而不是成功结果，
        # 模型据此知道工具没真的执行。
        return {
            "messages": [
                ToolMessage(
                    content=(
                        f"当前任务为只读模式，已阻止写操作 {names}。"
                        "不要再调用这个工具，改用只读方式把结论写在回复里。"
                    ),
                    name=tc["name"],
                    tool_call_id=tc["id"],
                    status="error",
                )
                for tc in blocked
            ]
        }


# ============================================================
# 3. 上下文注入
# ============================================================

class ContextInjectionMiddleware(AgentMiddleware):
    """
    动态提示词：把运行时上下文注入系统提示词。

    对应课件 09 章 4.1 节（courseware/尚硅谷-09-上下文与记忆.pdf）。区别是课件用 before_model 提前退出，
    这里用 wrap_model_call 改写请求的 system prompt，不中断流程。

    两种钩子各司其职：
        before_model：只读模式下提前给模型一句提醒。
        wrap_model_call：把上下文信息追加到系统提示词末尾。
    """

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        模型调用之前，只读模式下先提醒一句。

        返回的消息会进入消息历史，
        模型下一轮就能看到「这轮不能写」，
        减少它去尝试写操作再被拦的来回。
        """
        ctx = _ctx(runtime)
        if not ctx.allow_write:
            return {
                "messages": [
                    AIMessage(content="（提示：本轮为只读模式，任何写操作都会被拒绝。）")
                ]
            }
        return None

    def wrap_model_call(self, request, handler):
        """
        包裹模型调用：改写请求的系统提示词，再交给下游执行。

        为什么用 wrap_model_call 而不是 before_model：
            before_model 只能往消息列表里追加内容，
            改不了「这一次请求用哪个系统提示词」。
            wrap_model_call 能拿到 request，改造完再交给 handler 继续往下走。

        注意：
            system_prompt 只是只读 property，直接赋值会抛 DeprecationWarning，
            要改得用 override() 造一个新的 request。
        """
        ctx = _ctx(request.runtime)

        # 要追加到系统提示词末尾的上下文信息。
        extra = (
            f"\n\n[运行时上下文]\n"
            f"任务名：{ctx.task_name}\n"
            f"触发方式：{ctx.trigger}\n"
            f"写权限：{'允许' if ctx.allow_write else '禁止'}"
        )

        # system_prompt 只是只读 property，直接赋值已废弃；改用 override 造新 request
        base = request.system_message
        content = base.text if base is not None else ""

        # override 是「替换」语义，不是「追加」语义。
        #
        # 所以必须把原提示词的内容拼回去，
        # 否则原提示词会被冲掉 ——
        # 模型就不知道自己该怎么扮演了。
        new_request = request.override(
            system_message=SystemMessage(content=(content or "") + extra)
        )
        return handler(new_request)


# ============================================================
# 4. 耗时与用量统计
# ============================================================

class TimingMiddleware(AgentMiddleware):
    """记录本轮耗时与模型调用次数，写进日志。"""

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        整个 Agent 跑完之后，打印用量统计。

        注意：
            token 用量来自 AIMessage 上的 usage_metadata，
            而有些供应商的流式响应不返回它（比如 MiniMax）。
            这种情况下宁可不报，也不能报假的 0 ——
            「调用 0 次」会让人误以为模型根本没被调用。
        """
        ctx = _ctx(runtime)
        messages = state.get("messages", [])
        usage = [
            m.usage_metadata
            for m in messages
            if isinstance(m, AIMessage) and m.usage_metadata
        ]
        if usage:
            total_tokens = sum(u.get("total_tokens", 0) for u in usage)
            print(
                f"[{ctx.task_name}] 模型调用 {len(usage)} 次，"
                f"累计 token {total_tokens}，消息 {len(messages)} 条"
            )
        else:
            # 有些供应商的流式响应不返回 usage（如 MiniMax），别打印假的 0
            replies = sum(1 for m in messages if isinstance(m, AIMessage))
            print(
                f"[{ctx.task_name}] 助手消息 {replies} 条，"
                f"token 用量不可用（供应商未返回），消息 {len(messages)} 条"
            )
        return None


# ============================================================
# 5. 中间件清单
# ============================================================

# CUSTOM_MIDDLEWARE：本项目自定义的中间件实例列表。
#
# create_agent 接收中间件实例，这里集中构造。
#
# 顺序的意义（见 agent/graph.py）：
#   中间件栈是 LIMITS → RETRY → TODO → CONTEXT_EDITING → CUSTOM_MIDDLEWARE。
#   放在最后意味着它能看到前面所有中间件处理过的状态，
#   比如只读闸门拦截后，统计中间件统计到的就是被拦之后的结果。
CUSTOM_MIDDLEWARE = [
    TaskGateMiddleware(),
    ContextInjectionMiddleware(),
    TimingMiddleware(),
]
