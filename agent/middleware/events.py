"""
事件桥接中间件：把工具执行过程发成 `custom` 流事件。

对应课件：
- 08 章 5.3（courseware/langChain/尚硅谷-08-中间件.pdf）：wrap-style 钩子。
- 课件未覆盖事件流，本模块按桌面 Agent 的需要自行设计。

背景：
`runtime/events.py` 的归一化层能把 `agent.stream()` 的 `(模式名, 载荷)` 二元组
变成客户端能消费的事件，但工具执行过程**不在** `messages` 流里 ——
那里面只有模型的输出和工具结果，没有「工具什么时候开始、跑了多久、成功还是失败」。

这个中间件补上这块：用 `wrap_tool_call` 钩子包住工具执行，
在执行前后各发一条 `custom` 事件。

发出的事件（字段与 `runtime/events.py` 的事件类一一对应）：

| 事件 | 时机 | 字段 |
|---|---|---|
| `tool.start` | 调用工具前 | `name` / `call_id` / `args` |
| `tool.end` | 调用成功返回后 | `name` / `call_id` / `ok` / `duration_ms` / `result_preview` |
| `tool.error` | 调用抛异常时 | `name` / `call_id` / `message` |

**8 个工具函数一行都不用改** —— 事件全部从中间件这一层发出去。

注意：
1. `get_stream_writer()` 只能在图的执行上下文里调用，
   在外面（比如单测直接调工具）会抛 `RuntimeError`。
   所以取 writer 的地方必须 try/except 兜底，拿不到就静默跳过 ——
   不能因为发不出去一个事件就把整个调用搞崩。
2. `run_id` 由归一化层统一盖上，中间件不用管。
3. 只读闸门拦下写工具时，工具**根本没执行**，
   所以不会有 tool.start / tool.end —— 这是对的，没跑的工具不该有执行事件。
4. **它装配在中间件栈的最外层**（见 `agent/graph.py`）——
   wrap-style 钩子是洋葱结构，第一个定义的最外层。
   放最外层才能包住整个「工具调用 + 重试」过程：
   一次工具调用只发一对事件，失败也只有一条 tool.error，
   而 duration_ms 是用户真正等待的时间（含重试）。

   代价是**看不到每次重试的细节** —— 那是有意的：
   客户端要的是「这次调用失败了，等了 3 秒」，不是「失败了三次」。
"""

from __future__ import annotations

# time：算工具执行的耗时。
import time

# Any：typing 的任意类型标注。
from typing import Any

# AgentMiddleware：所有自定义中间件的基类。
# ToolCallRequest：工具调用的请求对象。
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from ..tools.risk import decide_tool_permission


# ============================================================
# 1. 常量
# ============================================================

# RESULT_PREVIEW_LIMIT：tool.end 里结果预览的最大字符数。
#
# 为什么要截断：
# 工具结果可能很大（读一个几百行的文件），
# 完整塞进事件里会撑爆客户端的界面与网络。
# 客户端要完整结果的话，让它自己去查文件。
RESULT_PREVIEW_LIMIT = 500


# ============================================================
# 2. 辅助函数
# ============================================================

def _preview(result: Any) -> str:
    """从工具执行结果里取一段文本预览。

    参数：
        result：handler 返回的对象（ToolMessage 或 Command）。

    返回：
        最多 RESULT_PREVIEW_LIMIT 个字符的文本。
        拿不到文本时返回空字符串。
    """
    content = getattr(result, "content", None)
    if not isinstance(content, str):
        return ""
    return content[:RESULT_PREVIEW_LIMIT]


# ============================================================
# 3. 事件桥接中间件
# ============================================================

class EventBridgeMiddleware(AgentMiddleware):
    """用 `custom` 流模式把工具执行过程发成事件。

    为什么用 `wrap_tool_call` 而不是 `after_model`：
    `after_model` 只能看到「模型这一轮要调哪些工具」，
    看不到工具什么时候真正跑完、跑了多久、失败没有。
    `wrap_tool_call` 包住的是**实际执行**，所以时序和耗时都准。
    """

    def wrap_tool_call(self, request, handler):
        """包住一次工具调用，在前后发事件。

        参数：
            request：ToolCallRequest，含 tool_call / tool / state / runtime。
            handler：真正执行工具的回调。

        返回：
            handler 的返回值，原样透传。

        注意：
            **失败有两种形态，两种都要处理**：

            1. handler 直接抛异常 —— 没有重试中间件兜着时是这样。
               发 tool.error 之后**继续往外抛**：中间件不吞异常，
               重试策略与错误处理是别人的职责。

            2. handler 正常返回，但结果是 `status="error"` 的 ToolMessage ——
               这是项目里的常态：`ToolRetryMiddleware` 的 `on_failure="continue"`
               会把重试耗尽后的失败包成一条 ToolMessage 继续跑。
               这种情况下不能算成功，要发 tool.error 而不是 tool.end。

            只处理第 2 种会漏报失败，只处理第 1 种会误报成功。
        """
        call = request.tool_call or {}
        name = str(call.get("name", ""))
        call_id = str(call.get("id", ""))
        args = call.get("args", {}) or {}

        # 只读闸门负责阻止没有授权入口的调用；这里处理交互式 WebSocket run。
        # 等待授权发生在工具开始事件之前，用户拒绝时工具 handler 完全不会执行。
        runtime = getattr(request, "runtime", None)
        ctx = getattr(runtime, "context", None)
        decision = decide_tool_permission(
            name,
            allow_write=bool(getattr(ctx, "allow_write", False)),
        )
        if decision != "allow":
            approval_handler = getattr(ctx, "approval_handler", None)
            approved = False
            if decision == "require_approval" and callable(approval_handler):
                try:
                    approved = approval_handler(name, call_id, args) is True
                except Exception:
                    # 授权通道出错时按拒绝处理，不能继续执行有副作用的工具。
                    approved = False

            if not approved:
                if decision == "deny":
                    message = f"工具 {name or '<未知>'} 没有风险登记，按默认拒绝策略未执行。"
                elif callable(approval_handler):
                    message = "用户未批准本次工具调用，工具未执行。"
                else:
                    message = "当前入口不支持单次授权，本次工具调用已拒绝。"
                self._emit({
                    "type": "tool.error",
                    "name": name,
                    "call_id": call_id,
                    "message": message,
                })
                return ToolMessage(
                    content=message,
                    name=name,
                    tool_call_id=call_id,
                    status="error",
                )

        self._emit({
            "type": "tool.start",
            "name": name,
            "call_id": call_id,
            "args": args,
        })

        started = time.perf_counter()
        try:
            result = handler(request)
        except Exception as exc:
            self._emit({
                "type": "tool.error",
                "name": name,
                "call_id": call_id,
                "message": f"{type(exc).__name__}: {exc}",
            })
            # 继续往外抛 —— 中间件不吞异常。
            raise

        duration_ms = int((time.perf_counter() - started) * 1000)

        # 没抛异常，但重试中间件可能把失败包成了一条 error 状态的 ToolMessage。
        # 那种情况要报失败，不能报成功。
        if getattr(result, "status", None) == "error":
            self._emit({
                "type": "tool.error",
                "name": name,
                "call_id": call_id,
                "message": str(getattr(result, "content", "")),
                "duration_ms": duration_ms,
            })
            return result

        self._emit({
            "type": "tool.end",
            "name": name,
            "call_id": call_id,
            "ok": True,
            "duration_ms": duration_ms,
            "result_preview": _preview(result),
        })
        return result

    # ---------- 发事件 ----------

    def _emit(self, payload: dict) -> None:
        """发一条 `custom` 事件。

        注意：
            拿不到 stream writer 时静默跳过 ——
            `get_stream_writer()` 只能在图的执行上下文里调用，
            在外面会抛 RuntimeError。发不出去一个事件不该搞崩整个调用。
        """
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
        except Exception:
            return

        if writer is None:
            return

        try:
            writer(payload)
        except Exception:
            # 事件通道本身出问题也不该影响工具执行。
            pass
