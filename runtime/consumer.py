"""
同步流消费器：把 agent 的流变成事件，投进 asyncio 队列。

对应课件：
- 07 章 6.2（courseware/langChain/尚硅谷-07-智能体.pdf）：create_agent 与流式执行。
- 课件未覆盖事件流，本模块按桌面 Agent 的需要自行设计。

对应桌面版改造计划的**阶段 3**：同步流消费器 + 线程桥接 + 有界队列背压。

为什么用同步 `agent.stream()` 而不是 `astream()`：
1. 现有假模型只在同步路径工作（`tests/fakes.py` 的 `FakeModel` 实现了 `_stream`，
   而 `astream()` 走的异步路径对假模型不友好）。
2. 同步路径跟现有 REPL、HTTP 的行为完全一致，测试不用改。

**为什么要线程桥接**：
`agent.stream()` 是同步的，要跑在**工作线程**里（不能占住事件循环）；
而事件队列是 asyncio 的，跨线程投递要显式处理。

**背压怎么做的**：
两层上限，缺一个就会把内存吃光：

| 层 | 是什么 | 谁在管 |
|---|---|---|
| 队列容量 | `asyncio.Queue(maxsize=N)` —— 已入队的事件 | 事件循环 |
| 在途上限 | `max_pending` —— 已投递但还没入队的事件 | 消费器自己计数 |

在途事件达到上限时，生产者在这里等，等事件循环把队列消化掉、计数降下来。
队列满时 put 协程本身也会 await 等待 —— 两层一起把生产者按住了。

**为什么不 `run_coroutine_threadsafe(...).result()`**：
那个 future 要靠事件循环的回调来 resolve。
如果事件循环在 resolve 之前关闭（进程退出、连接关闭），
future 永远不 resolve，调用方就**永久阻塞** ——
事件其实已经 put 进队列了，但生产者再也回不来。
所以这里只投递、不等结果，背压交给计数。
"""

from __future__ import annotations

# asyncio：跨线程投递事件用。
import asyncio

# logging：记录运行历史的回调失败不能影响事件流。
import logging

# threading：取消标志用（跨线程安全）。
import threading

# time：算耗时。
import time
from contextlib import closing

# Any：typing 的任意类型标注。
from typing import Any, Callable

# AIMessage / HumanMessage：agent 循环的消息类型。
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# BaseEvent / ModelText / RunCancelled / RunEnd / RunError：事件模型。
from runtime.events import (
    BaseEvent,
    ModelText,
    RunCancelled,
    RunEnd,
    RunError,
    StreamNormalizer,
)
from runtime.redaction import redact_sensitive_text

# AgentContext：运行时上下文。
from agent.context import AgentContext, TaskCancelledError, check_cancelled

logger = logging.getLogger(__name__)


# ============================================================
# 1. 常量
# ============================================================

# DEFAULT_QUEUE_SIZE：事件队列的默认容量。
#
# 为什么要有界：
# 客户端消费不过来时（网络慢、界面卡），无界队列会让内存无限涨。
# 有界 + 背压 = 生产者慢下来，而不是把内存吃光。
DEFAULT_QUEUE_SIZE = 256

# DEFAULT_MAX_PENDING：在途事件的默认上限。
#
# 「在途」= 已经投给事件循环、但还没进队列的事件。
# 队列容量管的是「已入队」，这个管的是「在途」——
# 两层加起来才是内存上界。
DEFAULT_MAX_PENDING = 256


# ============================================================
# 2. 消费器
# ============================================================

class StreamConsumer:
    """消费一次 agent 的流，把事件投进 asyncio 队列。

    **用法**：在同步线程里调用 `run()`。

        consumer = StreamConsumer(
            agent=agent,
            run_id="run-a1b2c3d4",
            queue=queue,          # asyncio.Queue(maxsize=N)
            loop=loop,            # 事件循环（在别的线程）
            cancel_event=cancel_event,
        )
        consumer.run("列出数据目录下的文件", ctx, thread_id="t1")

    注意：
    1. `run()` 是**阻塞**的 —— 它要等整个流跑完。
       所以要放在工作线程里跑，不能占住事件循环。
    2. 事件的 `run_id` 由调用方分配（阶段 4 的 RunManager 负责）。
    """

    def __init__(
        self,
        agent: Any,
        *,
        run_id: str,
        queue: "asyncio.Queue[BaseEvent]",
        loop: asyncio.AbstractEventLoop,
        cancel_event: threading.Event | None = None,
        max_pending: int = DEFAULT_MAX_PENDING,
        on_terminal: Callable[[BaseEvent], None] | None = None,
    ) -> None:
        # agent：编译好的 CompiledStateGraph。
        self.agent = agent

        # 归属的执行 id，所有产出的事件都带它。
        self.run_id = run_id

        # 事件队列与它所属的事件循环。
        self.queue = queue
        self.loop = loop

        # 取消标志。跨线程置位，消费循环每轮检查。
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.on_terminal = on_terminal

        # 归一化层：把 (模式名, 载荷) 变成事件。
        self.normalizer = StreamNormalizer(run_id)

        # 待投递事件计数与它的条件变量 —— 背压用。
        #
        # 「待投递」= 已经投给事件循环、但还没进队列的事件数。
        # 队列本身有界（maxsize），这里再限一层「在途」的量，
        # 两层加起来才是内存上界。
        self.max_pending = max_pending
        self._pending = 0
        self._slot_free = threading.Condition(threading.Lock())
        # 连接关闭后停止投递并唤醒背压等待者；put 任务只在事件循环中操作。
        self._closed = False
        self._put_tasks: set[asyncio.Task] = set()

    # ---------- 主流程 ----------

    def run(self, instruction: str, ctx: AgentContext, thread_id: str) -> None:
        """跑一次 agent，把过程与结果都变成事件投进队列。

        参数：
            instruction：给 Agent 的自然语言指令。
            ctx：运行时上下文。
            thread_id：会话 id，决定共享哪份记忆。

        注意：
        **异常路径必须闭合** —— 任何一种退出都要发出一条结束事件，
        否则客户端会一直转圈等下去：

        | 情况 | 发什么 |
        |---|---|
        | 正常跑完 | `run.end` |
        | 收到取消 | `run.cancelled`（不是 run.end） |
        | 模型调用失败等异常 | `run.error` |
        """
        started = time.perf_counter()
        ctx.cancel_event = self.cancel_event
        # 每步的完整状态，留最后一份。
        final_state: dict = {}
        # 累积模型说出来的正文。
        #
        # 为什么不用 state 里最后一条 AIMessage 的 content：
        # 那是**原始**内容，推理模型（MiniMax 等）会把 <think> 思考块混在里面 ——
        # 又脏又长，模型只输出了思考时甚至是空的。
        # model.text 是归一化层滤过思考块的正文，那才是用户该看到的。
        text_parts: list[str] = []

        try:
            check_cancelled(ctx)
            stream = self.agent.stream(
                {"messages": [HumanMessage(content=instruction)]},
                {"configurable": {"thread_id": thread_id}},
                context=ctx,
                # 三种模式：
                #   messages —— 模型的增量输出
                #   values   —— 每步的完整状态，用来在最后取最终答案
                #   custom   —— 中间件发的工具事件（阶段 2 的 EventBridgeMiddleware）
                stream_mode=["messages", "values", "custom"],
            )
            # 显式关闭生成器，让图的工作线程和检查点写入完成后再发送终态。
            with closing(stream):
                for mode, payload in stream:
                    # 审批取消时，先让当前工具节点提交匹配的 ToolMessage。
                    # 立刻抛取消异常会关闭生成器，留下只有 tool_call 的检查点。
                    if ctx.pending_approval_tool_call is None:
                        check_cancelled(ctx)
                    if mode == "values":
                        final_state = payload
                        if self.cancel_event.is_set() and self._pending_approval_result_committed(ctx, payload):
                            ctx.pending_approval_tool_call = None
                            raise TaskCancelledError("审批取消结果已写入检查点")
                        continue

                    for event in self.normalizer.normalize(mode, payload):
                        if isinstance(event, ModelText):
                            text_parts.append(event.text)
                        self._emit(event)
            # 无增量输出或正常返回恰逢取消时，也不能误报成功。
            check_cancelled(ctx)

        except Exception as exc:
            if isinstance(exc, TaskCancelledError) or self.cancel_event.is_set():
                self._persist_cancelled_approval_result(ctx, thread_id)
                if ctx.repair_session is not None:
                    ctx.repair_session.close("cancelled", "修复已停止：用户取消了任务。")
                self._emit_terminal(RunCancelled(run_id=self.run_id, reason="用户取消"))
                return
            # 模型调用失败（API 401、重试耗尽）等异常逃出消费循环。
            # 异常路径必须发出一条结束事件。
            self._emit_terminal(RunError(
                run_id=self.run_id,
                message=redact_sensitive_text(f"{type(exc).__name__}: {exc}"),
                recoverable=False,
            ))
            return

        duration_ms = int((time.perf_counter() - started) * 1000)

        output = "".join(text_parts)
        if not output:
            # 无流式正文时使用最终状态中的文本；模型异常已经走 run.error。
            output = _last_message_text(final_state.get("messages", []))
        if not output.strip():
            # 内部权限提示不能代替模型的答复；正常结束流也不代表任务完成。
            self._emit_terminal(RunError(
                run_id=self.run_id,
                message="模型未返回有效答复，无法确认任务完成。请重试或调整要求。",
                recoverable=True,
            ))
            return

        repair = ctx.repair_session
        self._emit_terminal(RunEnd(
            run_id=self.run_id,
            output=output,
            messages=len(final_state.get("messages", [])),
            duration_ms=duration_ms,
            repair_status=repair.terminal_status or None if repair else None,
            repair_message=repair.terminal_message or None if repair else None,
        ))

    @staticmethod
    def _pending_approval_result_committed(ctx: AgentContext, state: dict) -> bool:
        """确认待审批调用的 ToolMessage 已进入当前完整状态。"""
        pending = ctx.pending_approval_tool_call
        if not pending:
            return False
        call_id = pending.get("id", "")
        return bool(call_id) and any(
            isinstance(message, ToolMessage) and message.tool_call_id == call_id
            for message in state.get("messages", [])
        )

    def _persist_cancelled_approval_result(self, ctx: AgentContext, thread_id: str) -> None:
        """补齐审批取消时未进入检查点的工具结果。"""
        pending = ctx.pending_approval_tool_call
        if not pending:
            return
        call_id = pending.get("id", "")
        name = pending.get("name", "")
        if not call_id:
            return
        config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = self.agent.get_state(config)
            messages = list((getattr(snapshot, "values", None) or {}).get("messages", []))
            answered = any(
                isinstance(message, ToolMessage) and message.tool_call_id == call_id
                for message in messages
            )
            # 当前 pending 由 EventBridge 在模型发出 tool_call 后登记。流被关闭时
            # get_state() 可能暂时还看不到该 AIMessage，不能因此漏补结果。
            if not answered:
                self.agent.update_state(
                    config,
                    {"messages": [ToolMessage(
                        content="任务已取消，工具未执行。",
                        name=name,
                        tool_call_id=call_id,
                        status="error",
                    )]},
                )
        except Exception:
            # 取消本身必须完成；检查点补偿失败不能卡住桌面终态。
            logger.exception("取消时补齐工具结果失败：%s", call_id)
        finally:
            ctx.pending_approval_tool_call = None

    def _emit_terminal(self, event: BaseEvent) -> None:
        """先保存最终状态，再向连接发送对应结束事件。"""
        if self.on_terminal is not None:
            try:
                self.on_terminal(event)
            except Exception:
                logger.exception("运行结果持久化回调失败：%s", self.run_id)
        self._emit(event)

    # ---------- 跨线程投递 ----------

    def emit_event(self, event: BaseEvent) -> None:
        """线程安全地发布运行期间产生的额外事件。"""
        self._emit(event)

    def _emit(self, event: BaseEvent) -> None:
        """把一个事件投进 asyncio 队列。

        注意：
        **不等待投递完成** —— 只把 put 协程排进事件循环就走。

        为什么不 `run_coroutine_threadsafe(...).result()`：
        那个 future 要靠事件循环的回调来 resolve。
        如果事件循环在 resolve 之前关闭（进程退出、连接关闭），
        future 永远不 resolve，调用方就**永久阻塞** ——
        事件其实已经 put 进队列了，但生产者再也回不来。

        背压改由「待投递计数」实现：
        在途事件超过 max_pending 时，生产者在这里等，
        等事件循环把队列消化掉、计数降下来。

        两层上限：
            队列容量（asyncio.Queue 的 maxsize）—— 已入队的事件
            max_pending —— 在途（已投递未入队）的事件
        """
        # 背压：在途事件太多就先停下，等事件循环消化。
        with self._slot_free:
            while self._pending >= self.max_pending:
                if self._closed or self.loop.is_closed():
                    return
                self._slot_free.wait(timeout=0.1)
            if self._closed or self.loop.is_closed():
                return
            self._pending += 1

        try:
            self.loop.call_soon_threadsafe(self._schedule_put, event)
        except RuntimeError:
            # 事件循环已关闭。丢掉这一条，别把计数卡住。
            self._release()

    def _schedule_put(self, event: BaseEvent) -> None:
        """在事件循环里排一个 put 协程。

        队列满时这个协程会 await 等待 —— 那才是真正的背压：
        事件循环停在 put 上，生产者的计数降不下来，于是也停下来。
        """
        if self._closed:
            self._release()
            return
        try:
            task = asyncio.ensure_future(self.queue.put(event))
        except RuntimeError:
            self._release()
            return
        # 无论成功、失败还是被取消，都要释放计数。
        self._put_tasks.add(task)
        task.add_done_callback(self._put_done)

    def _put_done(self, task: asyncio.Task) -> None:
        """在事件循环中回收已投递或被取消的 put。"""
        self._put_tasks.discard(task)
        self._release()

    def _release(self) -> None:
        """释放一个在途名额，并唤醒等待的生产者。"""
        with self._slot_free:
            self._pending -= 1
            self._slot_free.notify()

    def close(self) -> None:
        """关闭消费器。

        唤醒所有等待中的生产者，让它们别再等下去。
        （进程退出或连接关闭时调用。）
        """
        with self._slot_free:
            self._closed = True
            self._slot_free.notify_all()
        try:
            self.loop.call_soon_threadsafe(self._cancel_puts)
        except RuntimeError:
            pass

    def _cancel_puts(self) -> None:
        """连接已不存在，取消仍在等队列空位的投递，不影响终态持久化。"""
        for task in tuple(self._put_tasks):
            task.cancel()


def _last_message_text(messages: list) -> str:
    """取消息列表里最后一条有文本的消息。

    参数：
        messages：agent 结束时的完整消息列表。

    返回：
        最后一条 AIMessage 的文本。没有则返回空字符串。

    注意：
    只看 AIMessage —— 最后一条可能是中间件插入的提示
    （ContextInjectionMiddleware 每轮都插一条），那不是模型的输出。
    """
    for msg in reversed(messages):
        # 只检查本轮消息，避免空回答时复用上一轮的成功结论。
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, AIMessage):
            if msg.name == "context_hint" or msg.tool_calls:
                continue
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else ""
    return ""
