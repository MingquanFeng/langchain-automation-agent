"""
事件归一化层：把 agent 的流式输出变成客户端能消费的事件。

对应课件：
- 第 07 章（courseware/langChain/尚硅谷-07-智能体.pdf）：stream_mode 的多种形态。
- 课件未覆盖事件模型，本模块按桌面 Agent 的需要自行设计。

分两层：
1. **事件模型** —— BaseEvent 加 9 种事件。
   所有事件统一带 version / run_id / timestamp / type 四个字段，
   客户端只认这四个 + 各事件自己的业务字段。
   version 是协议版本：将来改结构时客户端能识别。

2. **归一化层** —— StreamNormalizer。
   把 `agent.stream()` 吐出的 `(模式名, 载荷)` 二元组变成事件列表。

事件的形状由 LangGraph 的 stream_mode 决定：
- `"messages"` 模式 → 载荷是 `(chunk, metadata)`：
    * AIMessageChunk（增量）→ model.text
    * AIMessage（整条，中间件插入的）→ 滤掉例行提示后 → model.text
    * ToolMessage（工具结果）→ 丢弃 —— 它不是模型输出
- `"custom"` 模式 → 载荷是中间件用 get_stream_writer() 发来的 dict：
    按 `type` 字段分发成 tool.start / tool.end / tool.error

接下来会继续长出：
- 阶段 2 的 EventBridgeMiddleware 用 `custom` 模式发工具事件；
- 阶段 3 的消费器用 `messages` + `custom` 两种模式驱动本模块。
"""

from __future__ import annotations

# re：标准库的正则模块，用来匹配 <think> 思考块。
import re

# time：标准库，取当前时间戳。
import time

# Any：typing 的任意类型标注。
# Literal：类型字面量，用于把 `type` 字段钉死成具体取值。
from typing import Any, Literal

# Pydantic：模型与字段声明。
from pydantic import BaseModel, ConfigDict, Field

# AIMessage / AIMessageChunk / ToolMessage：agent 循环里的三种消息形态。
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage


# ============================================================
# 1. 事件模型
# ============================================================

class BaseEvent(BaseModel):
    """所有事件的基类。

    四个公共字段：
        version：协议版本。将来改结构时客户端靠它识别。
        run_id：属于哪一次执行。同一次执行的所有事件共享这个值。
        timestamp：time.time()，客户端用来排序。
        type：判别字段，取值见各子类。

    注意：
        extra="ignore" —— 中间件发来的 dict 可能带额外字段，
        解析时忽略而不是报错。协议要向前兼容。
    """

    model_config = ConfigDict(extra="ignore")

    # 协议版本，从 1 开始。
    version: int = 1

    # 归属的执行 id，由 RunManager 分配。
    run_id: str

    # 事件产生时刻，秒级浮点。
    timestamp: float = Field(default_factory=time.time)

    # 判别字段，各子类覆盖成 Literal。
    type: str


class RunStart(BaseEvent):
    """任务开始。收到 run.start 命令、分配好 run_id 后发。"""

    type: Literal["run.start"] = "run.start"

    # 任务名，出现在通知标题里。
    task_name: str

    # 会话 id，决定这次执行共享哪份记忆。
    thread_id: str

    # 原始指令。
    instruction: str


class ModelText(BaseEvent):
    """模型文本增量输出。

    只装模型真正说出来的正文 ——
    混在 content 里的 <think> 思考块已经被 ThinkFilter 滤掉。
    """

    type: Literal["model.text"] = "model.text"

    # 增量文本，不是完整答案。
    text: str


class ToolStart(BaseEvent):
    """工具开始执行。"""

    type: Literal["tool.start"] = "tool.start"

    # 工具名。
    name: str

    # 工具调用 id，与工具结果配对。
    call_id: str

    # 调用参数。
    args: dict[str, Any] = Field(default_factory=dict)


class ToolEnd(BaseEvent):
    """工具执行完成。"""

    type: Literal["tool.end"] = "tool.end"

    # 工具名。
    name: str

    # 工具调用 id，与工具结果配对。
    call_id: str

    # 是否成功。失败的情况走 tool.error。
    ok: bool = True

    # 执行耗时，毫秒。
    duration_ms: int | None = None

    # 结果预览。第一版只带一小段，避免大结果灌满界面。
    result_preview: str | None = None


class ToolError(BaseEvent):
    """工具执行失败。"""

    type: Literal["tool.error"] = "tool.error"

    # 工具名。
    name: str

    # 工具调用 id，与工具结果配对。
    call_id: str

    # 错误信息。
    message: str

    # 执行耗时，毫秒。
    #
    # 失败时也要给 —— 用户关心的是「等了多久才失败」。
    duration_ms: int | None = None


class RunProgress(BaseEvent):
    """任务进度。

    只定义不发出 —— 将来接 TodoList 进度时直接用。
    """

    type: Literal["run.progress"] = "run.progress"

    # 进度说明。
    note: str = ""


class RunEnd(BaseEvent):
    """任务正常结束。消费循环跑完时发。"""

    type: Literal["run.end"] = "run.end"

    # 最终答案。
    output: str = ""

    # 消息条数。
    messages: int = 0

    # 执行耗时，毫秒。
    duration_ms: int | None = None


class RunCancelled(BaseEvent):
    """任务取消。收到取消命令或连接断开时发。

    注意：
        取消时发的是 run.cancelled，不是 run.end ——
        客户端据此把界面从「运行中」改到「已取消」。
    """

    type: Literal["run.cancelled"] = "run.cancelled"

    # 取消原因。
    reason: str = ""


class RunError(BaseEvent):
    """任务异常。消费循环抛异常时发（模型调用失败、图崩溃等）。"""

    type: Literal["run.error"] = "run.error"

    # 错误信息。
    message: str

    # 是否可恢复。不可恢复意味着这次执行已经结束。
    recoverable: bool = False


# ============================================================
# 2. 事件类型注册表
# ============================================================

# type → 事件类。
#
# 归一化层靠它把 custom 模式发来的 dict 分发成对应的事件类；
# 反序列化（收到 JSON 还原成事件）也靠它。
EVENT_CLASSES: dict[str, type[BaseEvent]] = {
    "run.start": RunStart,
    "model.text": ModelText,
    "tool.start": ToolStart,
    "tool.end": ToolEnd,
    "tool.error": ToolError,
    "run.progress": RunProgress,
    "run.end": RunEnd,
    "run.cancelled": RunCancelled,
    "run.error": RunError,
}


# ============================================================
# 3. 思考内容过滤
# ============================================================

# MiniMax 这类推理模型会把思考过程以 <think>...</think> 的形式混在 content 里返回，
# 默认滤掉，否则界面会被大段自说自话刷屏。需要看推理过程时由上层决定。
#
# 正则说明：
# .*? 是非贪婪匹配，尽量少匹配。
# (?:</think>|\Z) 表示匹配到闭标签，或者一直匹配到字符串结尾（闭标签还没到的情况）。
# re.S 让 . 能匹配换行符，否则跨行的思考块会被漏掉。
THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)

# 这些中间件注入的消息是每轮的例行提示，界面上显示写权限就够了，不重复刷屏。
#
# 取值是 langgraph_node 名字的前缀，
# 用 startswith 比较是因为同一个节点可能带后缀
# （实测 node 名是 "ContextInjectionMiddleware.before_model"）。
HINT_NODES = ("ContextInjectionMiddleware",)


class ThinkFilter:
    """流式滤掉 <think> 块。

    逐 chunk 拼接后再判断；闭标签还没到时也先把当前这段当思考内容滤掉，
    避免思考过程漏到界面上。

    为什么不能逐 chunk 单独判断：
    思考内容是分片送过来的，
    某个 chunk 可能正好停在 "<think>" 中间，
    单独判断会以为没有思考块而把它原样输出。
    所以要把历史片段都留着，每次拿完整缓冲重新过滤。
    """

    def __init__(self) -> None:
        # 累积收到的原始文本（含思考块）。
        self._buf = ""

        # 已经输出过的干净文本长度。
        # 用它做切片，避免重复输出、也避免漏输出。
        self._printed = 0

    def reset(self) -> None:
        """
        清空缓冲。

        每轮对话开始前调用，
        防止上一轮的思考片段混进这一轮。
        """
        self._buf = ""
        self._printed = 0

    def feed(self, text: str) -> str:
        """吃进一段增量文本，返回可以显示的部分。

        参数：
            text：新收到的增量文本。

        返回：
            本次新产生的、可以显示的干净文本。
            没有新内容时返回空字符串。

        注意：
            返回空字符串不代表出错，
            很可能只是这一段全是思考内容。
        """
        self._buf += text

        # 拿完整缓冲重新过滤。
        clean = THINK_RE.sub("", self._buf)

        # 长度没变说明新来的部分全在思考块里，没有可显示内容。
        if len(clean) <= self._printed:
            return ""

        # 只吐出新增的那一段。
        out = clean[self._printed :]
        self._printed = len(clean)
        return out


def is_hint_node(node: str) -> bool:
    """判断一个 langgraph_node 是不是「例行提示」节点。

    参数：
        node：流式元数据里的 langgraph_node 值。

    返回：
        True 表示这条消息是中间件插入的例行提示，可以滤掉。

    为什么需要：
    ContextInjectionMiddleware 每轮都往消息列表里插一句写权限提示。
    界面本身已经显示了写权限，这句就是噪音；
    但写操作被拦截的说明、模型调用失败之类的消息要显示，不能一起滤掉。
    """
    return str(node).startswith(HINT_NODES)


# ============================================================
# 4. 归一化层
# ============================================================

class StreamNormalizer:
    """把 agent 的流式输出归一化成事件。

    **一次 run 一个实例** —— ThinkFilter 的缓冲状态跨 chunk 累积，
    换个实例就等于把思考片段的过滤状态清零。

    用法：

        normalizer = StreamNormalizer(run_id="run-a1b2c3d4")
        for mode, payload in agent.stream(..., stream_mode=["messages", "custom"]):
            for event in normalizer.normalize(mode, payload):
                ...
    """

    def __init__(self, run_id: str) -> None:
        # 归属的执行 id，所有产出的事件都带它。
        self.run_id = run_id

        # 思考过滤器，跨 chunk 保留缓冲。
        self._think = ThinkFilter()

    def reset(self) -> None:
        """每轮对话前调用，把思考过滤器的缓冲清空。

        防止上一轮的思考片段混进这一轮。
        """
        self._think.reset()

    def normalize(self, mode: str, payload: Any) -> list[BaseEvent]:
        """把一条流式输出归一化成事件。

        参数：
            mode：LangGraph 的 stream_mode 名字。
            payload：该模式对应的载荷。

        返回：
            事件列表。可能是空列表 —— 这一条流没有值得发的事件。

        注意：
            认不出来的 mode 一律返回空列表，不抛异常 ——
            归一化层要能扛住框架新增的流模式。
        """
        if mode == "messages":
            return self._from_messages(payload)
        if mode == "custom":
            return self._from_custom(payload)
        return []

    # ---------- messages 模式 ----------

    def _from_messages(self, payload: Any) -> list[BaseEvent]:
        """处理 `"messages"` 模式的载荷。

        载荷是 `(chunk, metadata)` 二元组：
            chunk —— AIMessageChunk（增量）、AIMessage（整条）或 ToolMessage（工具结果）
            metadata —— 带 langgraph_node 等信息
        """
        if not isinstance(payload, tuple) or len(payload) != 2:
            return []

        chunk, meta = payload

        # ToolMessage 是工具结果，不是模型输出 ——
        # 丢掉它，否则模型的一句话会被当成两段发出去。
        if isinstance(chunk, ToolMessage):
            return []

        # 增量文本：模型的正文走这条路。
        if isinstance(chunk, AIMessageChunk):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if not text:
                return []
            # 过一遍思考过滤器，滤掉混在正文里的 <think> 块。
            text = self._think.feed(text)
            return [ModelText(run_id=self.run_id, text=text)] if text else []

        # 整条消息：中间件插入的提示走这条路。
        if isinstance(chunk, AIMessage):
            # 例行提示（写权限提醒）滤掉，其余的要显示。
            node = str(meta.get("langgraph_node", "")) if isinstance(meta, dict) else ""
            if is_hint_node(node):
                return []
            text = chunk.content if isinstance(chunk.content, str) else ""
            return [ModelText(run_id=self.run_id, text=text)] if text else []

        return []

    # ---------- custom 模式 ----------

    def _from_custom(self, payload: Any) -> list[BaseEvent]:
        """处理 `"custom"` 模式的载荷。

        载荷是中间件用 `get_stream_writer()` 发来的 dict，
        形如 `{"type": "tool.start", "name": ..., "call_id": ..., "args": {...}}`。
        按 `type` 字段分发成对应的事件类。
        """
        if not isinstance(payload, dict):
            return []

        event_type = payload.get("type")
        cls = EVENT_CLASSES.get(event_type) if isinstance(event_type, str) else None
        if cls is None:
            # 未知的事件类型 —— 中间件可能发了客户端还不认识的东西，忽略。
            return []

        # type 由子类自己定死，不从载荷里取。
        data = {k: v for k, v in payload.items() if k != "type"}
        # run_id 由归一化层统一盖上，中间件不用管。
        data["run_id"] = self.run_id

        try:
            return [cls(**data)]
        except Exception:
            # 字段对不上（中间件改了事件形状）就跳过这一条，
            # 不能因为一条坏事件把整次执行搞崩。
            return []
