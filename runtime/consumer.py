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

# threading：取消标志用（跨线程安全）。
import threading

# time：算耗时。
import time

# Any：typing 的任意类型标注。
from typing import Any

# AIMessage / HumanMessage：agent 循环的消息类型。
from langchain_core.messages import AIMessage, HumanMessage

# BaseEvent / ModelText / RunCancelled / RunEnd / RunError：事件模型。
from runtime.events import (
    BaseEvent,
    ModelText,
    RunCancelled,
    RunEnd,
    RunError,
    StreamNormalizer,
)

# AgentContext：运行时上下文。
from agent.context import AgentContext


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
        cancelled = False
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
            for mode, payload in self.agent.stream(
                {"messages": [HumanMessage(content=instruction)]},
                {"configurable": {"thread_id": thread_id}},
                context=ctx,
                # 三种模式：
                #   messages —— 模型的增量输出
                #   values   —— 每步的完整状态，用来在最后取最终答案
                #   custom   —— 中间件发的工具事件（阶段 2 的 EventBridgeMiddleware）
                stream_mode=["messages", "values", "custom"],
            ):
                # 取消检查。每轮一次 —— 协作式取消，
                # 正在跑的那一步（模型调用或工具执行）会做完。
                if self.cancel_event.is_set():
                    cancelled = True
                    break

                # values 模式归一化层不处理。
                # 每步推全量状态 —— 留最后一份，正文为空时要拿它兜底。
                if mode == "values":
                    final_state = payload
                    continue

                for event in self.normalizer.normalize(mode, payload):
                    if isinstance(event, ModelText):
                        text_parts.append(event.text)
                    self._emit(event)

        except Exception as exc:
            # 模型调用失败（API 401、重试耗尽）等异常逃出消费循环。
            # 异常路径必须发出一条结束事件。
            self._emit(RunError(
                run_id=self.run_id,
                message=f"{type(exc).__name__}: {exc}",
                recoverable=False,
            ))
            return

        duration_ms = int((time.perf_counter() - started) * 1000)

        if cancelled:
            # 取消发的是 run.cancelled 而不是 run.end ——
            # 客户端据此把界面从「运行中」改到「已取消」。
            self._emit(RunCancelled(run_id=self.run_id, reason="用户取消"))
            return

        output = "".join(text_parts)
        if not output:
            # 没有正文 —— 多数是**模型调用失败**。
            # 重试中间件的 `on_failure="continue"` 会把失败包成一条 AIMessage
            # 写进 state，但那条消息不经过模型节点，
            # **`messages` 模式看不到它** —— 所以既没有 model.text，累积的正文也是空的。
            # 回退到 state 的最后一条消息，让用户看到「模型调用失败了」，
            # 而不是一片空白。
            output = _last_message_text(final_state.get("messages", []))

        self._emit(RunEnd(
            run_id=self.run_id,
            output=output,
            messages=len(text_parts),
            duration_ms=duration_ms,
        ))

    # ---------- 跨线程投递 ----------

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
                self._slot_free.wait(timeout=0.1)
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
        try:
            task = asyncio.ensure_future(self.queue.put(event))
        except RuntimeError:
            self._release()
            return
        # 无论成功、失败还是被取消，都要释放计数。
        task.add_done_callback(lambda _: self._release())

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
            self._pending = 0
            self._slot_free.notify_all()


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
        if isinstance(msg, AIMessage):
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else ""
    return ""
