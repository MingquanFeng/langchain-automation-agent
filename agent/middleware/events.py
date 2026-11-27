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

**17 个工具函数一行都不用改** —— 事件全部从中间件这一层发出去。

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
import re
from pathlib import Path

# Any：typing 的任意类型标注。
from typing import Any

# AgentMiddleware：所有自定义中间件的基类。
# ToolCallRequest：工具调用的请求对象。
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from ..tools.risk import decide_tool_permission
from ..tools.edits import (
    discard_approval_preview,
    prepare_approval_preview,
)
from ..context import TaskCancelledError, check_cancelled


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
_REPAIR_EXTENSIONS = frozenset({".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".c", ".h", ".cc", ".cpp", ".rs", ".go", ".java", ".kt", ".swift", ".rb", ".php", ".cs", ".vue", ".svelte", ".html", ".css", ".scss", ".sass", ".less"})
_REPAIR_BLOCKED_PARTS = frozenset({".github", ".gitlab", "alembic", "deploy", "deployment", "infra", "migrations", "migration", "terraform"})
# 明确标记为故意失败的 QA 夹具没有可修复缺陷；防止模型把夹具改绿或绕改其他文件。
_INTENTIONAL_FAILURE_MARKERS = ("QA 固定失败：用于验证三轮上限", "intentional failure fixture for repair limit")


def _repair_payload(session: object | None) -> dict[str, str | None]:
    """将已关闭的修复会话附到工具事件，供工作台立即展示终态。"""
    status = getattr(session, "terminal_status", "") if session is not None else ""
    return {
        "repair_status": status or None,
        "repair_message": getattr(session, "terminal_message", "") or None,
    }


def _close_repair(session: object | None, status: str, message: str) -> None:
    """兼容最小上下文替身；生产 RepairSession 会保留第一个终态。"""
    if session is not None and hasattr(session, "close"):
        session.close(status, message)


def _repair_error(ctx: object, name: str, args: dict[str, Any]) -> str:
    session = getattr(ctx, "repair_session", None)
    if session is None:
        return ""
    if session.terminal_status:
        return session.terminal_message or "当前受控修复已结束，不能继续调用工具。"
    if name == "edit_file":
        output = session.output_summary.casefold()
        # 此规则依据失败命令的原始输出，在生成审批请求前拦截所有编辑目标。
        if any(marker.casefold() in output for marker in _INTENTIONAL_FAILURE_MARKERS):
            return "原检查输出标记为故意失败的 QA 夹具，禁止编辑任何文件来消除该失败。"
        path = args.get("path")
        if session.attempts >= 3:
            return "修复验证已达到 3 次上限，不能再编辑。"
        if session.edit_pending:
            return "修复模式正在等待上一处编辑的批准或完成。"
        if session.edited_since_check:
            return "修复模式每次只允许一次编辑；请先复跑原检查。"
        if not isinstance(path, str):
            return "修复模式只允许修改源码、样式或测试文件。"
        relative = Path(path)
        if relative.suffix.lower() not in _REPAIR_EXTENSIONS or any(part.lower() in _REPAIR_BLOCKED_PARTS for part in relative.parts):
            return "修复模式只允许修改源码、样式或测试文件，不能修改配置、依赖、迁移或部署文件。"
        return ""
    if name == "run_command":
        if session.attempts >= 3:
            return "修复验证已达到 3 次上限，不能再运行检查。"
        if not session.edited_since_check:
            return "修复模式必须先完成一次已批准的编辑，才能复跑检查。"
        if args.get("argv") != session.argv or args.get("cwd", ".") != session.cwd or args.get("timeout_seconds", 60) != session.timeout_seconds:
            return "修复模式只能复跑来源失败检查的原始命令。"
        return ""
    risk = decide_tool_permission(name, allow_write=False)
    if risk != "allow":
        return "修复模式不允许此工具；只允许读取、编辑源码/测试文件和复跑原检查。"
    return ""


def _command_result(ctx: object, call_id: str, result: Any) -> dict[str, Any] | None:
    """读取命令工具写入的终态；兼容中间件重试包装后丢失上下文的旧路径。"""
    recorded = getattr(ctx, "command_results", {}).pop(call_id, None)
    if recorded is not None:
        return recorded
    content = _preview(result)
    match = re.match(r"命令(完成|失败|超时)（退出码 ([-\d]+|None)）", content)
    if match is None:
        return None
    status = {"完成": "succeeded", "失败": "failed", "超时": "timed_out"}[match.group(1)]
    return {"status": status, "exit_code": None if match.group(2) == "None" else int(match.group(2))}


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
    if isinstance(content, str):
        return content[:RESULT_PREVIEW_LIMIT]
    if isinstance(content, list):
        text = "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
        return text[:RESULT_PREVIEW_LIMIT]
    return ""


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
        check_cancelled(ctx)
        if getattr(ctx, "single_command", False):
            # 在参数校验与审批之前占用名额，失败也不得换解释器重试。
            # 同一模型响应可并行请求多个工具，因此检查与占用必须原子化。
            with ctx.command_attempt_lock:
                blocked = ctx.command_attempted
                if name == "run_command" and not blocked:
                    ctx.command_attempted = True
            if blocked:
                message = "本轮单次命令已尝试，禁止继续调用工具或更换解释器重试；请直接报告首次尝试结果。"
                self._emit({"type": "tool.error", "name": name, "call_id": call_id, "message": message})
                return ToolMessage(content=message, name=name, tool_call_id=call_id, status="error")
        decision = decide_tool_permission(
            name,
            allow_write=bool(getattr(ctx, "allow_write", False)),
        )
        repair_error = _repair_error(ctx, name, args)
        if repair_error:
            repair_session = getattr(ctx, "repair_session", None)
            # 模型可能在一条响应里并发发出多个编辑。首个编辑等待审批期间，
            # 其余编辑应单独拒绝；不能因并发请求提前关闭仍待用户批准的会话。
            deferred_repair_call = (
                repair_session is not None
                and not repair_session.terminal_status
                and (
                    name == "edit_file"
                    and (repair_session.edit_pending or repair_session.edited_since_check)
                    or name == "run_command"
                    and repair_session.edit_pending
                    and not repair_session.edited_since_check
                )
            )
            if repair_session is not None and not deferred_repair_call:
                _close_repair(repair_session, "policy_blocked", f"修复已停止：{repair_error}")
            self._emit({"type": "tool.error", "name": name, "call_id": call_id, "message": repair_error, **_repair_payload(repair_session)})
            return ToolMessage(content=repair_error, name=name, tool_call_id=call_id, status="error")
        session = getattr(ctx, "repair_session", None)
        if session is not None and name == "edit_file":
            session.edit_pending = True
        # 结构化编辑在批准前只计算差异并保存内存快照，不产生文件副作用。
        # 发送给客户端的是安全展示参数，避免把完整 old/new 文本重复塞进事件。
        try:
            approval_preview = prepare_approval_preview(name, call_id, args, ctx)
        except Exception as exc:
            approval_preview = None
            preview_error = f"无法生成编辑预览：{type(exc).__name__}: {exc}"
        else:
            preview_error = approval_preview.error
        event_args = approval_preview.args if approval_preview is not None else {}
        if preview_error:
            if session is not None and name == "edit_file":
                session.edit_pending = False
            discard_approval_preview(ctx, call_id)
            self._emit({
                "type": "tool.error",
                "name": name,
                "call_id": call_id,
                "message": preview_error,
            })
            return ToolMessage(
                content=preview_error,
                name=name,
                tool_call_id=call_id,
                status="error",
            )

        if decision != "allow":
            approval_handler = getattr(ctx, "approval_handler", None)
            approved = False
            if decision == "require_approval" and callable(approval_handler):
                # 取消可能让消费器在图提交结果前关闭流；记住这一项，
                # 由消费器在取消收尾阶段确认 ToolMessage 已写入检查点。
                ctx.pending_approval_tool_call = {"id": call_id, "name": name}
                try:
                    approved = approval_handler(name, call_id, event_args) is True
                except Exception:
                    # 授权通道出错时按拒绝处理，不能继续执行有副作用的工具。
                    approved = False

            # 用户批准与取消可能同时到达；等待结束后必须再次检查。
            try:
                check_cancelled(ctx)
            except TaskCancelledError:
                # 取消恰好发生在审批等待期间时，模型已经发出了 tool_call。
                # 必须先返回同一 call_id 的 ToolMessage，让检查点补齐这次调用；
                # 否则下次复用会话时，兼容接口会因缺少工具结果而返回 HTTP 400。
                if session is not None and name == "edit_file":
                    session.edit_pending = False
                discard_approval_preview(ctx, call_id)
                if session is not None and name in {"edit_file", "run_command"}:
                    _close_repair(session, "cancelled", "修复已停止：用户取消了审批。")
                message = "任务已取消，工具未执行。"
                self._emit({
                    "type": "tool.error",
                    "name": name,
                    "call_id": call_id,
                    "message": message,
                    **_repair_payload(session),
                })
                return ToolMessage(
                    content=message,
                    name=name,
                    tool_call_id=call_id,
                    status="error",
                )
            except Exception:
                if session is not None and name == "edit_file":
                    session.edit_pending = False
                discard_approval_preview(ctx, call_id)
                raise
            if not approved:
                if ctx is not None:
                    ctx.pending_approval_tool_call = None
                if session is not None and name == "edit_file":
                    session.edit_pending = False
                discard_approval_preview(ctx, call_id)
                if decision == "deny":
                    message = f"工具 {name or '<未知>'} 没有风险登记，按默认拒绝策略未执行。"
                elif callable(approval_handler):
                    message = "用户未批准本次工具调用，工具未执行。"
                else:
                    message = "当前入口不支持单次授权，本次工具调用已拒绝。"
                if session is not None and name in {"edit_file", "run_command"}:
                    approval = getattr(ctx, "approval_results", {}).pop(call_id, "rejected")
                    terminal = {
                        "timed_out": ("timed_out", "修复已停止：编辑或检查审批超时。"),
                        "cancelled": ("cancelled", "修复已停止：用户取消了审批。"),
                    }.get(approval, ("rejected", "修复已停止：用户拒绝了编辑或检查。"))
                    _close_repair(session, *terminal)
                self._emit({
                    "type": "tool.error",
                    "name": name,
                    "call_id": call_id,
                    "message": message,
                    **_repair_payload(session),
                })
                return ToolMessage(
                    content=message,
                    name=name,
                    tool_call_id=call_id,
                    status="error",
                )

            # 已获批准的调用会由正常工具链写入结果，无需取消补偿。
            if ctx is not None:
                ctx.pending_approval_tool_call = None

        self._emit({
            "type": "tool.start",
            "name": name,
            "call_id": call_id,
            "args": event_args,
        })

        started = time.perf_counter()
        try:
            result = handler(request)
        except Exception as exc:
            if session is not None and name in {"edit_file", "run_command"} and getattr(ctx, "cancel_event", None) is not None and ctx.cancel_event.is_set():
                _close_repair(session, "cancelled", "修复已停止：用户取消了任务。")
            self._emit({
                "type": "tool.error",
                "name": name,
                "call_id": call_id,
                "message": f"{type(exc).__name__}: {exc}",
                **_repair_payload(session),
            })
            # 继续往外抛 —— 中间件不吞异常。
            raise
        finally:
            # handler 正常消费后为空；校验层提前返回或异常时也不能残留快照。
            discard_approval_preview(ctx, call_id)
            if session is not None and name == "edit_file":
                session.edit_pending = False

        duration_ms = int((time.perf_counter() - started) * 1000)

        command_result = _command_result(ctx, call_id, result) if name == "run_command" else None
        if session is not None and name == "edit_file":
            edit_status = getattr(ctx, "edit_results", {}).pop(call_id, "failed")
            if edit_status == "succeeded":
                session.edited_since_check = True
            elif edit_status == "conflicted":
                _close_repair(session, "conflicted", "修复已停止：文件在审批期间发生变化。")
            else:
                _close_repair(session, "edit_failed", "修复已停止：编辑未成功写入文件。")
        if session is not None and name == "run_command" and command_result is not None:
            session.attempts += 1
            session.edited_since_check = False
            status = command_result["status"]
            if status == "succeeded":
                _close_repair(session, "succeeded", "检查通过，修复完成。")
            elif status == "timed_out":
                _close_repair(session, "timed_out", "修复已停止：原检查命令超时。")
            elif status == "cancelled":
                _close_repair(session, "cancelled", "修复已停止：用户取消了原检查。")
            elif session.attempts >= 3:
                _close_repair(session, "failed", "修复未完成：三次验证后原检查仍失败。")

        # 没抛异常，但重试中间件可能把失败包成了一条 error 状态的 ToolMessage。
        # 那种情况要报失败，不能报成功。
        if getattr(result, "status", None) == "error":
            self._emit({
                "type": "tool.error",
                "name": name,
                "call_id": call_id,
                "message": str(getattr(result, "content", "")),
                "duration_ms": duration_ms,
                "command_result": command_result,
                **_repair_payload(session),
            })
            return result

        self._emit({
            "type": "tool.end",
            "name": name,
            "call_id": call_id,
            "ok": True,
            "duration_ms": duration_ms,
            "result_preview": _preview(result),
            "command_result": command_result,
            **_repair_payload(session),
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
