# 桌面 Agent 改造计划

> 本文最初用于规划事件流后端，后续已补记实施状态与连续对话调试工作台。顶部的 Context、范围和 D1-D6 记录的是**实施前的设计基线**；当前状态以文末「实施顺序」「后续里程碑」和「当前推进清单」为准。

## Context：为什么要做这一步

启动这份计划时，agent 调用是**黑盒**：

- HTTP 层不提供触发入口，也没有交互式客户端 —— 只能调 `runner.run_task()` 跑一次等最终结果
- 中途发生了什么完全看不见
- 没有任何办法**打断**一次正在跑的 agent

桌面 Agent 的第一个要求是「运行中可见、可打断」。而 UI 的唯一输入是**事件流** ——
没有它，GUI 只能等最终结果；有了它，浏览器工作台和桌面壳都能使用同一份事件流。

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

- 本阶段不包含界面；这是当时的实施范围，浏览器工作台已在后续提交中完成
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
当时已有的 **14 个工具函数一行都不用改**；阶段 9 加入记忆工具后当前共 17 个，仍由同一中间件统一发事件。加进 `agent/graph.py` 的 `CUSTOM_MIDDLEWARE` 列表即可
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
| 9 | 连续对话调试工作台 | P1 | ✅ 已完成（现位于 `desktop/web/workbench.html`，通过 WebSocket 展示事件并复用 `thread_id`） |

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

**二、模型调用失败的终态（已在运行可靠性改造中修正）。**

本文的「异常路径必须闭合」一节写的是「模型调用失败 → 异常逃出消费循环 → `run.error`」。
实测不成立：`ModelRetryMiddleware` 的默认 `on_failure="continue"`
会把重试耗尽后的失败包成一条 AIMessage 继续跑 —— 图本身是「成功」跑完的。

当前已设置 `ModelRetryMiddleware(on_failure="error")`：模型重试耗尽后发 `run.error`，
运行历史保存 `failed`；同步入口也保存失败并抛出异常。普通工具失败仍交给模型处理，
取消异常则直接终止，不作为工具失败重试。
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

**异常路径**：未被中间件处理的异常及模型重试耗尽均应收到 `run.error`，没有 `run.end`；取消后到达的返回值或异常应成为 `run.cancelled`。普通工具失败可交给模型解释，客户端同时收到 `tool.error`。

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
| **已完成：2. 权限分级** | 工具风险登记、策略决策、逐次授权协议与浏览器确认界面 | 1 |
| **已完成：3. 桌面 GUI MVP** | Tauri 桌面窗口复用现有浏览器工作台与 `/ws` 协议 | 1、2 |
| **已完成：4. 桌面后端托管与打包** | 桌面应用启动/关闭 Python 服务，处理配置、日志和 macOS 应用打包 | 3 |
| **已完成：5. 任务与运行记录持久化** | 任务配置、执行结果和状态写入 SQLite；通过 `/runs` 分页查询 | 4 |
| **已完成：6. 工作台历史与会话恢复** | 查看历史运行、恢复对应 `thread_id` 与对话 | 5 |
| 7. 运行可靠性 | 模型与工具超时、取消语义、并发与故障恢复 | 5、6 |
| **已完成：8. 工具集扩张** | 只读目录树和代码搜索；可审批、可撤销的结构化编辑；完成通用 shell 评估并决定暂不加入 | 2、7 |
| 9. 长期记忆 | 持久化 `store`、项目级记忆、跨会话上下文与删除控制 | 6、8 |
| 10. 分发加固 | 鉴权、审计、数据保留策略、签名与公证 | 7、9；面向外部分发时 |

**当前建议顺序**：阶段 6 的历史与会话恢复、阶段 7 的主要可靠性边界、阶段 8 的安全搜索和结构化编辑已经接入；通用 shell 评估结论为暂不加入。阶段 9 已完成 SQLite Store、用户/项目隔离、显式管理入口以及来源、过期和冲突治理；下一步按优先级补阶段 7 的故障边界，外部发行加固按实际分发计划推进。

### 当前推进清单

#### 阶段 5：任务与运行记录持久化

阶段已完成。运行数据使用独立的 `runtime.sqlite3`，与 LangGraph 会话 checkpointer 分开。2026-11-16 验证结果：针对性持久化测试 10 项通过，完整测试套件 144 项通过。

- [x] 服务首次启动时从 `runtime/tasks.py` 默认项初始化任务表；动态任务增删改持久化。
- [x] 记录 WebSocket 流式任务与 `run_task()` 同步任务的开始、结束、输出、错误和状态。
- [x] 服务重启时把遗留的 `running` 标记为 `interrupted`。
- [x] 提供 `GET /runs?limit=50&offset=0`，限制单页最多 100 条。
- [x] POSIX 系统下将运行数据库文件权限设为仅当前用户可读写。
- [x] 回归验证：动态任务重启后保留；成功、失败、取消和中断状态正确；分页排序稳定；运行结果不影响既有 WebSocket 事件；运行记录写入失败时 WebSocket 不启动 Agent。

**验收标准**：重启前后 `/tasks` 返回一致；`/runs` 能查到 WebSocket、CLI/API 与定时任务执行结果；服务重启后旧的运行中记录显示为 `interrupted`；数据库写入失败时 WebSocket 任务不会在无记录状态下启动。

#### 阶段 6：工作台历史与会话恢复

阶段已完成。2026-11-17 完成最终回归：后端覆盖会话与运行记录分页、消息读取、同一 `thread_id` 继续对话、内部提示过滤、删除/清空记录保留记忆、删除会话联动清理，以及运行中记录保护；真实浏览器验证两类历史翻页、`interrupted` 展示、会话恢复和删除请求。

- [x] 工作台增加历史运行入口，支持按时间浏览、查看任务指令、输出、状态和耗时。
- [x] 增加会话内容查询接口，从 `thread_id` 读取 Checkpointer 消息；列表记录和消息内容分层返回。
- [x] 选择历史会话后恢复对应 `thread_id`，后续消息沿用该会话记忆，不自动重发旧任务。
- [x] 删除会话时同步清理会话检查点与关联运行记录；运行记录支持单条删除或清空。
- [x] 断线或页面重载后可以刷新查询 run 状态，并区分 `interrupted` 与已完成结果。
- [x] 回归验证历史分页、消息读取、会话恢复以及删除语义。

**验收标准**：页面重载后可浏览历史并继续选定会话；运行中断后能区分 `interrupted` 和已完成结果；删除操作能清理用户可见记录及对应持久化数据。

#### 阶段 7：运行可靠性

- [x] 为模型请求配置 `LLM_REQUEST_TIMEOUT`（默认 60 秒），关闭 SDK 重试叠加，重试耗尽记录 `failed` 和异常；现有 webhook 工具保持 15 秒 I/O 超时。
- [x] 定义取消边界：等待授权时立即拒绝；阻塞模型/工具调用时展示“正在取消”，在当前同步调用返回后结束，不再发起下一次调用或重试。
- [x] WebSocket、同步 / 定时任务和会话删除共用进程内会话占用表；验证不同会话并发使用 SQLite checkpointer 和运行记录库不会串线，工作线程退出前不释放占用。
- [x] 扩展跨进程 SQLite 锁竞争与磁盘故障注入测试：运行记录库在另一进程短暂持有写锁时，会在 10 秒 busy timeout 内等待后成功；`disk I/O error` 会回滚，不留下 `running` 半条记录。LangGraph SQLite Checkpointer 在生产路径完成 schema 初始化后，也能等待另一进程短暂写锁并完成会话写入（2026-11-19，`tests/test_reliability.py`）。
- [x] 统一服务退出、进程崩溃、事件队列关闭和记录写入失败的终态：首个 `succeeded` / `failed` / `cancelled` 终态不可被迟到事件覆盖；worker 无法启动时立即写入 `failed`；断连或队列关闭会先持久化 `cancelled`，再停止投递；进程退出前未能结束的 `running` 在下次启动统一收敛为 `interrupted`；终态写入暂时失败不改变客户端事件，数据库恢复后的下一次启动会完成该收敛（2026-11-19，`tests/test_runs.py`、`tests/test_storage.py`、`tests/test_consumer.py`）。
- [ ] 展示可恢复错误和最终错误，保留必要诊断信息，同时避免把凭据写入运行记录。

**验收标准**：模型或工具超时后 run 在限定时间内结束；取消状态与实际工具副作用一致；服务重启后无 run 永久停留在 `running`；并发任务不会串会话或破坏运行历史。

**本轮已验证的故障边界**：断线或发送队列满时关闭待发送任务，唤醒背压等待者，终态持久化仍执行；审批取消不执行工具；模型失败不再记为成功；重复启动和非法 JSON 是可恢复的命令错误。首次 Agent 初始化加锁，避免重复模型客户端和数据库连接。

**时间与部署边界**：I/O 超时是连接 / 读取等待上限，不是整轮任务的总时限。同步文件调用或持续有流式数据的连接不具备可强杀的截止时间；已发生的工具副作用不回滚。硬性总时限和任意工具强制中断需要独立进程执行；多服务进程共享同一数据库也需要额外协调。记录库写入失败仍记录诊断日志，服务重启把残留 `running` 标为 `interrupted`；磁盘故障下的终态持久化补偿留在后续存储加固中处理。

#### 阶段 8：工具集扩张

阶段已完成。阶段 8 交付的 14 个工具覆盖目录浏览、名称/正文搜索、受限文件读写、带 diff 审批的结构化编辑、备份恢复和显式通知；阶段 9 另增加 3 个长期记忆工具。通用 shell 已完成评估，现阶段不加入。

**通用 shell 评估结论（2026-11-17）**：现有真实任务没有必须执行任意命令的场景；搜索、读取、编辑、恢复与通知均有边界明确的专用工具。加入 shell 会让单次工具审批承载任意文件访问、进程启动、网络请求和破坏性命令，能够绕过工作目录沙箱、敏感文件过滤与结构化编辑备份；当前协作式取消也不能保证立即终止同步子进程。桌面 WebView 继续不开放通用 Shell capability，Python Agent 也不注册命令执行工具。

只有在 formatter、测试、构建或诊断命令成为稳定且高频的明确需求，并且专用工具不足时才重新评估。届时优先实现结构化 `run_command(argv, cwd, timeout)`，不接受 shell 字符串且不使用 `shell=True`；同时要求默认关闭、命令/参数 allowlist、独立高风险审批、工作目录约束、环境变量清洗、输出上限、进程组超时终止和完整审计。旧入口的整轮 `allow_write=True` 不得自动放行该能力。

- [x] 添加受工作目录约束的目录树、文件名搜索和源码内容搜索工具。
- [x] 保持隐藏目录、依赖目录、凭据文件过滤和大文件截断；每个新工具登记风险等级。
- [x] 实现结构化编辑预览并接入逐次审批；批准后重新校验原文摘要，原子保存前创建可恢复备份，恢复操作也展示反向差异并保留恢复前状态。
- [x] 根据真实使用场景评估通用 shell；结论为现阶段不加入，继续使用边界明确的专用工具，并保留重新评估条件与安全前置要求。

**验收标准**：搜索不会越出工作目录或读取敏感文件；编辑在批准前不产生副作用；失败或撤销后文件可以恢复到操作前状态。

#### 阶段 9：长期记忆

第一项已完成。2026-11-17 验证结果：Store 重开持久化、文件权限、内存后端、默认装配、用户/项目命名空间稳定性与隔离测试通过；完整测试套件 202 项通过。当前不会自动提取对话内容，避免在缺少管理入口时积累不可见记忆。

第二项已完成。2026-11-17 验证结果：显式记忆 CRUD、用户/项目精确查询、输入限制、Agent Store 注入、无授权拒绝、HTTP 管理接口与桌面托管鉴权回归通过；真实页面验证设置入口、新增、编辑和删除二次确认，完整测试套件 217 项通过。

第三项已完成。2026-11-17 验证结果：长期记忆现在记录创建来源与最近修改来源、Store 创建/更新时间、修订号和固定过期策略；过期记录不再返回并会惰性清理。设置页和 Agent 工具均基于修订号更新或删除，重复内容及旧修订请求明确返回冲突，不会静默覆盖。针对性测试 109 项、完整测试套件 220 项通过；真实页面验证 30/90 天策略、新增、修订更新、来源展示和二次确认删除。

- [x] 将当前内存 `Store` 替换为 SQLite 持久后端，定义用户级与项目级命名空间。Store 使用独立的 `store.sqlite`，项目标识由规范化工作目录哈希生成；测试默认隔离到内存后端。
- [x] 只保存用户明确需要跨会话复用的偏好和事实，提供记忆查看、修改和删除入口。新增 3 个记忆工具、`/memories` CRUD API 和设置页管理界面；写入与删除工具使用独立持久状态风险等级并逐次审批。
- [x] 明确记忆写入来源、更新时间、过期策略和冲突处理；避免把整段对话无差别复制为记忆。来源由可信入口写入；支持永不过期或 30/90/365 天；更新与删除使用修订号乐观并发控制，相同内容不重复创建。

**验收标准**：服务重启后授权范围内的记忆仍可用；不同项目之间不串记忆；用户能查看并删除任意一条长期记忆。

#### 阶段 10：分发加固

- [ ] 若对外分发，增加本机服务访问校验，限制非预期网页或本机进程调用。
- [ ] 增加工具审批、外部通知和文件修改的审计记录；补充历史清理与数据库备份/恢复流程。
- [ ] 将 API Key 迁移到系统凭据存储，评估数据库与运行记录的敏感信息保护。
- [ ] 配置 macOS 签名、公证和升级流程；需要其他平台时再增加对应构建与验收。

**验收标准**：安装包可在目标机器干净启动；密钥不写入安装包或普通日志；敏感操作可追溯；签名、公证和升级按目标发行渠道验证。

### 已完成阶段记录：阶段 2 权限分级与写操作确认

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
2. **[x] 把权限策略集中到中间件**：新增基于风险等级的 `allow` / `require_approval` / `deny` 决策，并由 `TaskGateMiddleware` 统一执行；旧入口的 `allow_write` 作为整轮授权兼容保留，WebSocket 工作台通过授权回调逐次确认。
3. **[x] 扩展 WebSocket 确认协议**：服务端发 `approval.request`，客户端回 `approval.respond`；请求和应答至少关联 `run_id`、`call_id` 与允许/拒绝决定。工具必须等到匹配的允许应答后才开始执行。
4. **[x] 处理确认生命周期**：拒绝、超时、取消、断连、重复应答、未知 `call_id` 和并发待确认调用均有明确结果；等待确认时仍能接收 `cancel`，所有退出路径都清理等待状态。
5. **[x] 更新浏览器工作台**：展示待确认的工具、简要参数和操作说明；提供单次允许与拒绝；显示处理结果。移除或调整整轮“允许写入”选项，避免它绕过逐次确认。
6. **[x] 补齐测试与文档**：覆盖工具分类、策略判断、WebSocket 请求/应答、拒绝与超时、取消/断连清理、旧调用方兼容，并同步更新协议和使用说明。

#### 验收标准

- 所有已注册工具均有风险等级；未分类工具默认拒绝（旧入口显式授予整轮 `allow_write=True` 时按兼容规则放行）。
- 未收到与当前 `run_id`、`call_id` 匹配的允许应答前，写工具和通知工具没有副作用。
- 拒绝、超时、取消或连接断开后工具均未执行，Agent 与 WebSocket 连接不会因等待确认而卡住。
- 旧 Python API 与定时任务仍按原有 `allow_write` 配置工作。
- 后端新增自动化测试覆盖允许、拒绝、错误应答和退出清理；完整测试套件通过。

**本阶段暂不做**：记住用户的长期授权、多任务并行审批和新工具扩展。浏览器工作台的单次确认闭环已完成。

---

## 下一阶段任务：阶段 3 桌面 GUI MVP

### 目标

提供一个可独立打开的桌面窗口，让用户在桌面应用里使用现有聊天工作台。桌面版复用
FastAPI 服务、`/ws` 事件协议、会话记忆和逐次授权逻辑；不复制 Agent 执行与权限判断。

第一版使用共享工作台，由 FastAPI 根路径提供；桌面资源迁移后，工作台已放入 `desktop/web/workbench.html` 并随 Tauri 应用打包。
开发期间先由用户单独启动 `python -m runtime --no-schedule`，桌面应用连接本机服务。Python
后端托管、安装包和一键启动放在下一阶段处理。

### 首版范围

- 仅支持本机桌面应用与本机 Agent 服务连接；首个交付目标为 macOS。
- 展示现有聊天工作台，支持连续对话、实时模型输出、工具调用卡片、逐次授权和取消。
- 阶段 3 当时保留浏览器访问 `http://127.0.0.1:8000/`；当前版本已停止提供浏览器工作台，聊天界面只随桌面应用分发。
- 服务地址可配置；服务未启动、连接中断和 Agent 错误都要在界面中明确显示。
- 首版不在 Tauri 窗口中启动或托管 Python 服务，不打可分发安装包。

### 集成方案结论

- 阶段 3 的方案曾由连接页打开 FastAPI 根路径；当前 Tauri 主窗口加载随应用打包的连接页和同目录工作台，启动服务后通过 `postMessage` 把 loopback 地址交给工作台。
- 工作台直接访问 sidecar 的 HTTP API 与 `/ws`；FastAPI 只对 Tauri 本地 WebView 源开放必要的 GET/DELETE CORS。Python 服务不再托管 HTML 或图片资源。
- 服务地址只接受 `127.0.0.1` / `localhost` 的 HTTP URL，支持自定义端口并保存在桌面 WebView 的 `localStorage`。
- Tauri CSP 只允许应用资源及 loopback HTTP/WebSocket 连接；工作台 iframe 加载应用包内的兄弟资源。Tauri capability 不开放通用文件系统或 Shell，只使用系统目录选择对话框。参考 [Tauri 配置文档](https://v2.tauri.app/reference/config/) 和 [Tauri CSP 文档](https://v2.tauri.app/security/csp/)。

### 实施任务

1. **[x] 桌面集成方案**：以打包的连接页承载地址配置与错误提示，iframe 加载应用包内工作台，由工作台直连 sidecar API 与 `/ws`。
2. **[x] Tauri 项目骨架**：在独立 `desktop/` 目录建立 Tauri v2 应用，创建主窗口、应用名称与临时图标、开发命令和本地运行配置；Rust 编译检查已通过，未改动 Agent 核心模块。
3. **[x] 连接本地服务**：新增随应用打包的连接页，提供可配置 loopback 服务地址、工作台握手、失败展示与重试入口。工作台可手动重连；所有重试只建立连接，不自动重发 Agent 命令。
4. **[x] 桌面交互验收**：完整 Python 套件 131 项通过；随后在 macOS Tauri WebView 通过隔离模拟服务验证连续对话与 `thread_id` 复用、文本流、工具事件、授权允许/拒绝、取消和 Agent 错误展示。模拟服务与测试窗口均已关闭。
5. **[x] 桌面回归与文档**：阶段 3 曾保留浏览器入口；当前工作台和图标均随 Tauri 静态资源打包，不再提供浏览器工作台。

### 验收标准

- 本地启动 FastAPI 后，可以通过开发命令打开 Tauri 桌面窗口并进入现有聊天工作台。
- 桌面窗口可以完成连续对话，实时呈现模型文本与工具事件；同一会话继续使用服务端返回的 `thread_id`。
- 有副作用的工具仍须针对当前 `run_id` / `call_id` 明确批准；拒绝、取消、断连和错误应答不会误执行工具。
- 桌面窗口中的工作台可访问 sidecar HTTP API 与 `/ws`，连续对话仍复用服务端返回的 `thread_id`。
- 桌面集成相关验证通过；完整项目测试套件通过，测试不调用真实模型 API。

### 本阶段暂不做

- Python Agent 后端随桌面应用自动启动、后台托管或随包分发。
- 面向其他操作系统的安装包、签名和自动更新。
- Vue 重写、工具集扩张、Shell 工具和长期记忆界面。

## 阶段 4：桌面后端托管与打包

### 已确定的打包方式

- **PyInstaller 单文件 sidecar + Tauri `externalBin`**：构建机使用项目运行环境生成与当前 macOS 架构匹配的 Python 可执行文件，由桌面应用随包携带。Tauri sidecar 按目标架构命名；构建流程写入 `desktop/src-tauri/binaries/`。
- sidecar 由 Rust 侧调用并管理进程，不通过 WebView 的 Shell 插件 API 启动；连接页只读取运行状态和最近日志，并可选择文件工具工作目录、请求重启。
- 桌面运行时配置和会话记忆放到系统应用支持目录；文件工具可选工作目录，默认仍是应用支持目录下的 `data/`；日志写入同目录的 `logs/runtime.log`。现有终端启动方式继续使用项目根目录的 `.env` 与 `data/`。

### 实施任务

1. **[x] 确定 Python 托管原型**：添加 PyInstaller 打包入口与 macOS 目标架构脚本，并配置 Tauri sidecar。
2. **[x] 隔离桌面配置和数据**：支持通过 `AUTOMATION_AGENT_HOME` 指向应用支持目录，首次启动创建 `.env`；SQLite 留在应用支持目录，文件工具可配置独立工作目录。
3. **[x] 启动、监控和停止服务**：Rust 侧分配空闲 loopback 端口、启动 sidecar、检查 `/health`；退出时发 `SIGTERM`，超时后回收进程组。
4. **[x] 日志、配置位置和恢复界面**：展示 `.env` 与日志路径、读取最近日志，并在启动失败时允许重启托管服务。
5. **[x] macOS `.app` 启动验收**：上一轮验收中，arm64 App 重启后成功连接本机服务，`/health` 返回 200，工作台执行清单完成 4/4；49 项针对性测试与 134 项完整回归通过。冒烟检查确认假凭据读取被拦截，`.next`、`node_modules` 与敏感文件被过滤，越界探测被拒绝。

### 验收标准

- 不依赖已安装的系统 Python 即可启动随包 Python sidecar。
- 桌面应用只绑定 loopback 空闲端口；失败时显示原因和日志路径，不终止已有的服务进程。
- 关闭桌面窗口或退出应用时有序停止托管服务。
- API Key、SQLite 与日志均留在用户应用支持目录；源码目录和安装包不保存用户密钥。
- macOS `.app` 可以构建和启动；签名、公证与自动更新另行安排。
