# 桌面 Agent 改造计划

> 本文最初用于规划事件流后端，后续已补记实施状态与连续对话调试工作台。顶部的 Context、范围和 D1-D6 记录的是**实施前的设计基线**；当前完成情况以文末「实施顺序」和「后续里程碑」为准。

## Context：为什么要做这一步

启动这份计划时，agent 调用是**黑盒**：

- HTTP 层不提供触发入口，交互终端也不存在 —— 只能调 `runner.run_task()` 跑一次等最终结果
- 中途发生了什么完全看不见
- 没有任何办法**打断**一次正在跑的 agent

桌面 Agent 的第一个要求是「运行中可见、可打断」。而 UI 的唯一输入是**事件流** ——
没有它，GUI 只能等最终结果；有了它，终端、网页、桌面壳都能从同一份事件流长出来。

所以计划的第一步是：**把 agent 的内部过程变成结构化事件，用 WebSocket 双向推给客户端，并支持客户端取消**。当时先实现后端；浏览器工作台已在后续提交中补上。

### 已实测验证的技术前提

下面几条是方案成立的关键，均已在本机 langchain 1.2.12 上跑通：

| 前提 | 验证方式 | 结论 |
|---|---|---|
| 中间件 `wrap_tool_call` 里能拿到 `get_stream_writer()` | 在 `wrap_tool_call` 里发 custom 事件 | ✅ 拿到，`custom` 模式收到了 `{'type': 'tool.start', 'name': 'list_files', 'args': {}, 'id': 'c1'}` |
| `stream_mode` 传列表时产出 `(mode, payload)` 二元组 | `stream_mode=["messages","custom"]` | ✅ 已实测 |
| 事件顺序正确 | 观察完整流 | ✅ `tool.start` → `tool.end` → `ToolMessage` → 最终回答 |
| FastAPI WebSocket + uvicorn 在本环境可用 | 起了个最小 ws 端点并用 `websockets` 客户端连 | ✅ 收到全部事件。`websockets` 16.0 已装（`courseware/langChain/requirements.txt:81`），uvicorn 的 `auto` 后端会自动选 `websockets_impl` |
| 跨线程推事件可行 | `asyncio.to_thread` 里跑同步代码 + `loop.call_soon_threadsafe` | ✅ 事件按顺序送达客户端 |

**另外查清了两个约束**（会直接影响实现方式）：

- ⚠️ **`GenericFakeChatModel`（`tests/test_e2e.py` 在用）不能配 `agent.stream()`** ——
  它没实现 `_stream` 的 chunk 格式，流式路径会报 `ValueError: No generations found in stream`。
  `tests/fakes.py` 的 `FakeModel` 实现了 `_stream`，**是流式路径唯一可用的假模型**。
- ⚠️ **`astream()` 异步路径对现有假模型不友好**（`_agenerate_with_cache` → fake stream 为空）。
  所以第一步走**同步 `agent.stream()` + 线程桥接**，不引入异步调用路径。

---

## 范围

### 这一步做

1. `runtime/events.py` —— 事件模型 + 归一化层
2. `agent/middleware/custom.py` —— 加 `EventBridgeMiddleware`（用 `wrap_tool_call` 发工具事件）
3. `runtime/runs.py` —— **Run Manager**：run 生命周期、取消、清理
4. `runtime/server.py` —— 加 WebSocket 端点 `/ws`
5. 同步流的消费器 —— 消费 `agent.stream()`，检查取消标志，归一化后推给客户端
6. 测试 —— `tests/test_events.py`、`tests/test_runs.py`

### 这一步不做

- 本阶段不包含界面（TUI / GUI）；这是当时的实施范围，浏览器工作台已在后续提交中完成
- 权限分级与授权弹窗 —— 第三步
- `interrupt()` / 暂停等输入 —— 只做「取消」
- 工具级超时 —— 见 P1 说明，只在事件模型里预留
- 多任务并发 —— 第一版单连接单任务
- 不改现有 HTTP 端点的行为 —— 纯新增 `/ws`

---

## 设计

### 架构

```
WebSocket 客户端
      │
      │  ← {"type":"run.start","instruction":"...","thread_id":"..."}
      │  ← {"type":"cancel"}
      │
      ▼
┌─ runtime/server.py  @app.websocket("/ws") ─────────────────────────┐
│                                                                   │
│  收命令 ──→ RunManager.start() 分配 run_id、登记 cancel_event      │
│         ──→ 起 asyncio task                                      │
│                                                                   │
│  ┌─ runtime/runs.py  RunManager ────────────────────────────┐     │
│  │  run_id → RunHandle(run_id, thread_id, cancel_event, task) │     │
│  │  start() / cancel() / finish() / get()                    │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                   │
│  事件生产（同步线程）──────────────► 有界 asyncio.Queue            │
│    runtime/events.py                      │  （满了 = 背压，不丢）  │
│      consume_sync_stream()                ▼                        │
│        agent.stream(stream_mode=      推送 websocket.send_json()   │
│          ["messages","custom"])                                  │
│                                                                   │
│  收 "cancel" ──→ RunManager.cancel(run_id) → cancel_event.set()   │
│  连接断开   ──→ 同上 + 清理 run 登记                               │
└───────────────────────────────────────────────────────────────────┘
```

### 核心概念：run_id 与 thread_id

这两个必须分开 —— 原计划里混用了，是缺陷。

| | 含义 | 生命周期 | 生成规则 |
|---|---|---|---|
| **`run_id`** | 一次执行的标识 | 只在本次执行内有效，结束即注销 | `run-{uuid8}`，每次执行唯一 |
| **`thread_id`** | 会话标识，记忆的隔离键 | 跨多次执行复用 | 由客户端指定；不指定则一次性 |

一个会话（thread）里可以连续跑多个 run，每个 run 有独立的 run_id 和独立的取消标志，
但共享同一份记忆。这正是桌面 Agent 的常态：用户在同一个对话里连问三件事。

注意 `runtime/runner.py:148` 现在生成 `task-{uuid}` 当 thread_id —— 那是「一次性会话」的语义，
在桌面端要改成「不指定 thread_id 时，本次 run 用一个临时 thread」。

### 事件模型（`runtime/events.py`）

所有事件继承同一个基类，统一带 `version` / `run_id` / `timestamp`：

```python
class BaseEvent(BaseModel):
    version: int = 1          # 协议版本，将来改结构时客户端能识别
    run_id: str               # 属于哪一次执行
    timestamp: float          # time.time()，客户端用来排序
    type: str                 # 判别字段
```

十种事件：

| 事件 | 作用 | 什么时候发 |
|---|---|---|
| `run.start` | 任务开始 | 收到 `run.start` 命令、分配好 run_id 后 |
| `model.text` | 模型文本增量输出 | `"messages"` 模式里 `AIMessageChunk` 且 content 是非空字符串 |
| `tool.start` | 工具开始执行 | `EventBridgeMiddleware` 的 `wrap_tool_call` 入口 |
| `tool.end` | 工具执行完成 | `wrap_tool_call` 的 handler 正常返回 |
| `tool.error` | 工具执行失败 | `wrap_tool_call` 里 handler 抛异常 |
| `approval.request` | 请求本次工具调用授权 | 有副作用的工具开始执行前；客户端用 `approval.respond` 携带相同 `run_id` / `call_id` 和布尔值 `approved` 应答 |
| `run.progress` | 任务进度 | **预留**，第一版不产生（将来给 TodoList 进度用） |
| `run.end` | 任务正常结束 | 消费循环正常跑完 |
| `run.cancelled` | 任务取消 | 收到 `cancel` 或连接断开 |
| `run.error` | 任务异常 | 消费循环抛异常（模型调用失败、图崩溃等） |

`run.progress` 只定义不发出 —— 用户的原则是「其余功能先预留接口，不提前实现」。

### 三种流模式怎么归一化

| LangGraph `stream_mode` | payload 形态 | 归一化成 |
|---|---|---|
| `"messages"` | `(chunk, metadata)` | `model.text` —— 必须同时满足：`isinstance(chunk, AIMessageChunk)` **且** `isinstance(chunk.content, str)` **且** 非空 |
| `"custom"` | 中间件发的 dict | 按 `type` 字段分发成 `tool.start` / `tool.end` / `tool.error` |
| （消费循环结束） | — | `run.end` |

**content 类型兼容**：content 可能是字符串，也可能是内容块列表（多模态）。
只取字符串形态，非字符串形态直接跳过 —— 多模态内容块列表是已知的坑，别把它当文本发出去。

**ToolMessage 要滤掉** —— 它是工具结果不是模型输出，否则模型的一句话会被当成两段发出去。

### 异常路径必须闭合（这是原计划的漏洞）

三种失败要分别处理，**任何一种都必须发出一条结束事件**，否则 UI 会一直转圈等下去：

| 失败点 | 表现 | 发什么 |
|---|---|---|
| 工具执行抛异常 | `ToolRetryMiddleware` 先重试 2 次；`on_failure="continue"` 会把错误包成 ToolMessage 继续，**异常不逃逸** | `tool.error`（由 EventBridgeMiddleware 在 `wrap_tool_call` 的 except 里发），流程继续 |
| 模型调用失败（API 401、超时耗尽重试） | 异常逃出消费循环 | `run.error` |
| 客户端取消 | cancel_event 被置位 | `run.cancelled` |

消费器的形状：

```python
try:
    for chunk in ...:
        if cancel_event.is_set(): break
        await emit(normalize(chunk))
except asyncio.CancelledError:
    await emit(RunCancelled(...))
    raise
except Exception as exc:
    await emit(RunError(message=str(exc), recoverable=False))
finally:
    manager.finish(run_id)     # 注销 run，释放资源
```

### Run Manager（`runtime/runs.py`）

职责单一：**run 的生命周期管理**。不碰 agent、不碰事件格式。

```python
@dataclass
class RunHandle:
    run_id: str
    thread_id: str
    cancel_event: threading.Event
    task: asyncio.Task | None = None
    started_at: float = 0.0

class RunManager:
    def start(self, ...) -> RunHandle      # 分配 run_id，建 cancel_event，登记
    def get(self, run_id) -> RunHandle | None
    def cancel(self, run_id) -> bool        # 置 cancel_event；返回是否命中
    def finish(self, run_id) -> None        # 注销（幂等）
    def active(self) -> list[RunHandle]
```

**并发限制：第一版单连接单任务。**
每个 WebSocket 连接同时只允许一个活跃 run。收到 `run.start` 时若已有活跃 run ——
**拒绝并回一条 `run.error`**（说明「上一轮还在运行」），而不是静默取消旧的。
理由：静默取消会让用户以为自己看到的是新一轮的结果。将来要做多任务，
Run Manager 按 run_id 管理已经能承接，届时再加「事件按 run_id 路由」。

### 取消语义

**协作式取消**，明确定义：

- 置位 `cancel_event` 后，消费循环在**下一次迭代**时检查到并 `break`
- 当前正在执行的那一步（模型调用或工具执行）**会做完**
- 发出 `run.cancelled`（不是 `run.end`），客户端据此把界面从「运行中」改到「已取消」

不做瞬时硬中断，也不做 `interrupt()`（暂停等输入再恢复）—— 后者需要设计暂停点并处理
checkpoint 恢复，是另一件事。**这一点要如实写进文档和 UI 提示**，否则用户会以为点了停就立刻停。

### 有界队列与背压

事件队列有界（`asyncio.Queue(maxsize=N)`），防止客户端消费不过来时无限堆积内存。

关键：**满了要背压，不能丢事件**。生产者在同步线程里，不能直接 `await queue.put()`。
做法是跨线程投递一个协程：

```python
loop.call_soon_threadsafe(lambda: asyncio.ensure_future(queue.put(event)))
```

队列满时这个协程会 await 等待，从而对生产者形成背压；事件不会丢。
（`call_soon_threadsafe` 在 loop 已关闭时会抛 `RuntimeError`，要 try/except 兜住，
否则关闭过程中的最后一个事件会把消费协程搞崩。）

### 关键决策

**D1 · 用同步 `agent.stream()` + 线程桥接，不用 `astream()`**
现有假模型只在同步路径工作（见上）。走同步路径还有个好处：当时既有测试不需要改动；目前测试套件已扩展，数量以实际运行结果为准。

**D2 · 取消用 `threading.Event` + `break`，不 hack 生成器**
在消费循环里检查标志位，命中就 `break`。生成器随之被关闭，LangGraph 自行清理。
不采用 `gen.close()` 从外部中断 —— 那要求生成器不在阻塞调用中，语义脆弱。

**D3 · 用中间件发工具事件，不改工具函数**
`EventBridgeMiddleware` 走 `wrap_tool_call` 钩子，在工具执行前后各发一条 custom 事件。
**8 个工具函数一行都不用改**。加进 `agent/graph.py` 的 `CUSTOM_MIDDLEWARE` 列表即可
（注意 `build_agent()` 不接受 `middleware` 参数，中间件栈是它内部固定组装的）。

`get_stream_writer()` 必须 try/except 兜底 —— 万一不在图的执行路径里（比如单测直接调工具），
拿不到就静默跳过，不能让整个调用崩掉。

**D4 · 现有 HTTP 端点不动**
`/health`、`/tasks`、`/jobs` 保留。新增 `/ws`。桌面版只用 `/ws` 做交互，
HTTP 侧退成探活与定时任务管理。

**D6 · WebSocket 端点只有一个发送点**

Starlette 的 `WebSocket.send` **没有锁** —— 并发发帧会交错。
所以所有要发给客户端的消息（事件的 + 端点自己产生的 `run.start` / `run.error`）
都进同一个队列，由唯一的推送任务发出去。端点自己不直接 send。

**D5 · 归一化层独立成 `runtime/events.py`**
归一化层是 `runtime/events.py`（目前只有 `ThinkFilter` 与 `is_hint_node` 两个过滤器，
事件模型与流式解析在这一步长出来）。

### 复用与新增

| 复用 | 来自 |
|---|---|
| `_AGENT` 单例 | `runtime/runner.py:58` `get_agent()` |
| `AgentContext` 构造规则 | `runtime/runner.py:132-137` |
| `config` 与 thread_id 规则 | `runtime/runner.py:148` |
| 流式假模型 | `tests/fakes.py` 的 `FakeModel` |
| 端点测试写法 | `tests/test_runtime.py`、`tests/fakes.py` 的 `build_agent(model=..., checkpointer=InMemorySaver())` |

新增：

- `runtime/events.py`
- `runtime/runs.py`
- `agent/middleware/custom.py` 里的 `EventBridgeMiddleware`
- `runtime/server.py` 的 `@app.websocket("/ws")`
- `tests/test_events.py`、`tests/test_runs.py`

---

## 实施顺序

| 阶段 | 内容 | 优先级 | 状态 |
|---|---|---|---|
| 1 | 事件模型（`BaseEvent` + 10 种事件）+ 归一化层 | P0 | ✅ 已完成（`runtime/events.py`） |
| 2 | `EventBridgeMiddleware`（`wrap_tool_call` 发 tool.start/end/error） | P0 | ✅ 已完成（`agent/middleware/events.py`） |
| 3 | 同步流消费器 + 线程桥接 + 有界队列背压 | P0 | ✅ 已完成（`runtime/consumer.py`） |
| 4 | Run Manager（run_id / cancel_event / start / cancel / finish） | P0 | ✅ 已完成（`runtime/runs.py`） |
| 5 | WebSocket 端点 `/ws`（收 run.start / cancel，单连接单任务） | P0 | ✅ 已完成（`runtime/server.py`） |
| 6 | 异常处理 + 连接断开清理 | P0 | ✅ 已完成（消费器、Run Manager、WebSocket 均处理结束/取消/异常路径；断连会取消活跃 run） |
| 7 | 单元测试 + 集成测试 | P0 | ✅ 后端覆盖已完成（事件、消费器、Run Manager、WebSocket 均有测试；聊天页面尚无自动化回归测试） |
| 8 | 文档同步 | P1 | ✅ 已完成（README、CLAUDE、LEARNING、KNOWLEDGE 与本计划已对齐当前实现） |
| 9 | 连续对话调试工作台（GET `/`） | P1 | ✅ 已完成（`runtime/static/index.html`，通过 WebSocket 展示事件并复用 `thread_id`） |

阶段 3 和 4 有依赖（消费器要用 cancel_event），其余可独立推进。

### 阶段 3 的实现与本文的两处出入

**一、背压不是「等 put 完成」，而是「在途计数」。**

本文原方案写的是 `loop.call_soon_threadsafe(lambda: asyncio.ensure_future(queue.put(event)))`，
它不阻塞生产者 —— 队列满了待投递的协程会一直挂着，事件确实没丢，但内存照涨，
跟「有界队列防堆积」的本意相反。

改成用 `asyncio.run_coroutine_threadsafe(...).result()` 能真正背压，
但**实测会永久阻塞**：那个 future 靠事件循环的回调来 resolve，
事件循环在 resolve 之前关闭（进程退出、连接关闭）时，future 永远不 resolve ——
事件其实已经 put 进队列了，生产者却再也回不来。实测表现为消费线程卡死 8 秒。

现在的实现是两层上限：
- 队列容量（`asyncio.Queue` 的 `maxsize`）管「已入队」
- 在途上限（`max_pending`）管「已投递未入队」

生产者只在计数上等，不等任何 future，事件循环关闭时不会卡死。

**二、模型调用失败不会逃出消费循环。**

本文的「异常路径必须闭合」一节写的是「模型调用失败 → 异常逃出消费循环 → `run.error`」。
实测不成立：`ModelRetryMiddleware` 的默认 `on_failure="continue"`
会把重试耗尽后的失败包成一条 AIMessage 继续跑 —— 图本身是「成功」跑完的。

所以现状是：
- 模型失败 → 包成一条 AIMessage → 消费循环正常结束 → 发 `run.end`，错误信息出现在 `output` 里
- `run.error` 只在**不被任何中间件兜住**的异常时出现（图崩溃、checkpointer 出错等）

要让模型失败也触发 `run.error`，得把 `agent/graph.py` 里 `ModelRetryMiddleware` 的
`on_failure` 改成 `"error"` —— 那是行为变更，会影响 REPL 与 HTTP，等要做的时候再决定。
阶段 6 的异常处理分散在 3 和 5 里，单独列出来是因为它最容易漏 —— 必须在每条退出路径上
都保证发出一条结束事件。

---

## 验证

```bash
# 1. 运行完整测试套件（当前 README 记录为 101 项；以实际 pytest 结果为准）
$PY -m pytest tests/ -q

# 2. 运行事件流后端的重点测试
$PY -m pytest tests/test_events.py tests/test_event_bridge.py tests/test_consumer.py tests/test_runs.py tests/test_ws.py -q

# 3. 服务起得来，WebSocket 端点可用
$PY -m runtime --no-schedule --port 8125
```

第 3 步的手工验证（`websockets` 已装）：

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8125/ws") as ws:
        await ws.send(json.dumps({"type": "run.start",
                                  "instruction": "列出数据目录下的文件",
                                  "allow_write": False}))
        async for raw in ws:
            print(raw)

asyncio.run(main())
```

**期望看到的序列**（每条都带 `version` / `run_id` / `timestamp`）：

```
{"version":1,"run_id":"run-a1b2c3d4","timestamp":1762...,"type":"run.start","task_name":"...","thread_id":"...","instruction":"..."}
{"version":1,"run_id":"run-a1b2c3d4","timestamp":1762...,"type":"tool.start","name":"list_files","call_id":"c1","args":{}}
{"version":1,"run_id":"run-a1b2c3d4","timestamp":1762...,"type":"tool.end","name":"list_files","call_id":"c1","ok":true,"duration_ms":3,"result_preview":"..."}
{"version":1,"run_id":"run-a1b2c3d4","timestamp":1762...,"type":"model.text","text":"数据目录下有 ..."}
{"version":1,"run_id":"run-a1b2c3d4","timestamp":1762...,"type":"run.end","output":"...","messages":4,"duration_ms":812}
```

**取消路径**：发 `run.start` 后立刻发 `cancel`，应收到 `run.cancelled` 且**没有** `run.end`。

**异常路径**：区分两种情况验证：未被中间件处理的异常应收到 `run.error` 且没有 `run.end`；模型调用失败在当前 `ModelRetryMiddleware(on_failure="continue")` 配置下会作为回答内容结束，收到 `run.end`。不要把后一种行为误判成 WebSocket 异常路径失效。

---

## P1 预留项（不在这一步实现）

| 项 | 为什么现在不做 |
|---|---|
| 工具级超时 | `ToolCallRequest` 没有 timeout 字段，同步工具的超时需要每个工具自己处理。等第三步做工具集扩张时一起做 |
| `run.progress` 事件 | 事件模型里定义了但不产生。将来接 TodoList 进度时直接用 |
| 多任务并发 | Run Manager 按 run_id 管理已能承接，届时再加「事件按 run_id 路由」 |
| 事件结果的完整回传 | 第一版 `tool.end` 只带 500 字符的 `result_preview`，避免大结果灌满 UI |

---

## 后续里程碑

| 步骤 | 内容 | 依赖 |
|---|---|---|
| **已完成：1. 事件流后端**（本文） | WebSocket、事件模型、Run Manager、取消机制 | — |
| **已完成：连续对话调试工作台** | 浏览器聊天界面、实时事件展示、同一会话复用 `thread_id` | 1 |
| 2. 权限分级 | 工具声明 `risk_level`；`TaskGateMiddleware` 从二值改策略式；授权请求事件 → 客户端应答 | 1 |
| 3. Textual TUI | 接上事件流；diff 视图、工具卡片、授权弹窗 | 1、2 |
| 4. 工具集扩张 | 先只读（目录树、grep、读任意源码），再可逆写（结构化编辑），最后 shell | 2 |
| 5. Tauri GUI | Vue 3 + TypeScript 的前端 + Tauri 壳 + Python Agent 后端 | 1-4 |
| 6. 长期记忆 | `store` 真用起来；SQLite 持久化、项目级记忆、跨会话上下文 | 4 |

**TUI 排在 GUI 前面**：Claude Code 本身是终端应用，真正的分界线是「能不能展示结构化信息」——
纯文本终端只能打印文本流，TUI 能渲染工具卡片和 diff。GUI 是包装，不是能力。

**开发原则**：先保证 Agent Runtime 稳定，再做 TUI，最后做 GUI。

### 下一阶段任务：阶段 2 权限分级与写操作确认

**目标**：把当前 WebSocket 工作台的整轮 `allow_write` 开关升级为最小权限策略。只读操作直接运行；有副作用的工具在执行前逐次请求用户确认。用户拒绝、确认超时、任务取消或连接断开时，相关工具不得执行。

#### 工具风险分类

| 风险等级 | 当前工具 | 处理方式 |
|---|---|---|
| `read_only` | `list_files`、`read_file`、`read_json` | 自动放行 |
| `workspace_write` | `write_file`、`append_file`、`save_json` | 执行前请求本次调用确认 |
| `external_effect` | `notify`、`notify_summary` | 执行前请求本次调用确认 |
| 未声明等级 | 后续新增工具 | 默认拒绝并记录错误，不能静默放行 |

#### 实施任务

1. **[x] 为工具声明风险等级**：在 `agent/tools/risk.py` 集中维护风险等级；工具注册表导入时校验工具清单与风险表完全对应，避免遗漏或陈旧登记。
2. **[x] 把权限策略集中到中间件**：新增基于风险等级的 `allow` / `require_approval` / `deny` 决策，并由 `TaskGateMiddleware` 统一执行；旧入口的 `allow_write` 作为整轮授权兼容保留。WebSocket 确认交互由后续协议任务接入。
3. **[x] 扩展 WebSocket 确认协议**：服务端发 `approval.request`，客户端回 `approval.respond`；请求和应答至少关联 `run_id`、`call_id` 与允许/拒绝决定。工具必须等到匹配的允许应答后才开始执行。
4. **[ ] 处理确认生命周期**：拒绝、超时、取消、断连、重复应答、未知 `call_id` 和并发待确认调用均有明确结果；等待确认时仍能接收 `cancel`，所有退出路径都清理等待状态。
5. **[ ] 更新浏览器工作台**：展示待确认的工具、简要参数和操作说明；提供单次允许与拒绝；显示处理结果。移除或调整整轮“允许写入”选项，避免它绕过逐次确认。
6. **[ ] 补齐测试与文档**：覆盖工具分类、策略判断、WebSocket 请求/应答、拒绝与超时、取消/断连清理、旧调用方兼容，并同步更新协议和使用说明。

#### 验收标准

- 所有已注册工具均有风险等级；未分类工具默认拒绝（旧入口显式授予整轮 `allow_write=True` 时按兼容规则放行）。
- 未收到与当前 `run_id`、`call_id` 匹配的允许应答前，写工具和通知工具没有副作用。
- 拒绝、超时、取消或连接断开后工具均未执行，Agent 与 WebSocket 连接不会因等待确认而卡住。
- 旧 Python API 与定时任务仍按原有 `allow_write` 配置工作。
- 后端新增自动化测试覆盖允许、拒绝、错误应答和退出清理；完整测试套件通过。

**本阶段暂不做**：记住用户的长期授权、多任务并行审批、新工具扩展、Textual TUI 与 Tauri GUI。先完成浏览器工作台中的单次确认闭环。
