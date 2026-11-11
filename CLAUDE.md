# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目性质

这是一个**学习驱动**的项目，不是生产系统。`automation_agent/` 是边学边搭的自动化 Agent 骨架，课件在仓库内的 `courseware/` —— 两套讲义：尚硅谷 LangChain 1.x 的 10 章 PDF（`courseware/langChain/`），和 **LangGraph 专项课件 6 篇 Markdown**（`courseware/langGraph/`，章节号 1-13 连续），见 `courseware/README.md` 的来源说明与章节索引。

课件对每个知识点都有对应章节，源码注释里用「对应课件 XX 章 X.X 节」标注了出处。**改代码前先看对应课件章节**，保持实现与课件 API 一致；只有当课件 API 与 langchain 1.2 实际行为冲突时，才按实际行为写并在注释里说明差异。

PDF 文件名即章节号（`courseware/langChain/尚硅谷-05-Tools.pdf` → 课件 05 章）。`LEARNING.md` 是「章节 ↔ 源码位置」的索引，`courseware/` 是原文，两者配合使用。

> `README.md` 是对外的项目说明（快速开始、目录结构），本文件是给 Claude 的工作指南 —— 两者互补，**改动以本文件为准**。

## 环境

固定使用 conda 环境 `langchain1.2`（Python 3.13.12），不要用系统 Python：

```bash
PY=/opt/miniconda3/envs/langchain1.2/bin/python
```

`conda` shell 函数在 zsh 里是坏的（`CONDA_EXE` 未设置，报 permission denied），一律用绝对路径调用：

```bash
/opt/miniconda3/bin/conda env list
```

依赖基线来自 `courseware/langChain/requirements.txt`（57 项，LangChain/LangGraph/MCP/模型供应商/Web/配置/日志/记忆，RAG 及以后章节被注释掉）。本项目在此之上额外装了：`apscheduler`（定时）、`langgraph-cli`（服务化）、`langgraph-checkpoint-sqlite`（持久化记忆）。

已装关键版本：`langchain 1.2.12`、`langchain-core 1.2.18`、`langgraph 1.1.2`、`langgraph-checkpoint-sqlite 3.1.2`、`apscheduler 3.11.3`、`fastapi 0.135.1`、`uvicorn 0.46.0`、`httpx 0.28.1`。

> **课件的 requirements.txt 是 UTF-16 LE 编码。** 直接 `pip install -r` 会乱码，需先 `iconv -f UTF-16LE -t UTF-8` 转码。

`MAX_MODEL_CALLS`（`.env` 默认 30）是限次阈值的唯一来源 —— `graph.py` 里 `ModelCallLimitMiddleware(thread_limit=RUNTIME.max_model_calls)`。调限次只改这个环境变量，不用改代码。

## 常用命令

所有命令都在本目录（`automation_agent/`）下执行。

```bash
# 跑完整测试套件（当前记录为 101 项，全部用假模型，不调真实模型 API）
$PY -m pytest tests/ -q

# 跑单个测试
$PY -m pytest tests/test_e2e.py::test_write_tool_actually_runs -q

# 起服务（含定时任务）
$PY -m runtime

# 只起服务不起定时任务
$PY -m runtime --no-schedule

# 开发模式热重载
$PY -m runtime --reload

# 换端口/监听地址
$PY -m runtime --host 0.0.0.0 --port 8080

# serve 是默认子命令，$PY -m runtime serve 与 $PY -m runtime 等价
```

本项目目前没有单独的 lint、type-check 或 build 步骤。提交前运行 `pytest tests/ -q`，确认全部通过；测试数量随测试增删变化，以实际输出为准。涉及服务或界面时，再启动服务做相应手工验证。

服务起来后的 HTTP 端点：

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/health` | 健康检查 + 调度器开关状态 |
| GET | `/tasks` | 列出定时任务表 |
| POST | `/tasks` | 动态添加定时任务（cron + 指令） |
| DELETE | `/tasks/{name}` | 删除定时任务 |
| GET | `/jobs` | 列出调度器已注册的 job 与下次运行时间 |
| GET | `/` | 连续对话调试工作台 |
| WS | `/ws` | 桌面版事件流。收 `run.start` / `cancel`，推结构化事件 |

浏览器工作台通过 WebSocket `/ws` 启动和取消交互式任务；Python 调用方可用 `runtime/runner.py` 的 `run_task()`，定时任务由 `runtime/scheduler.py` 触发。当前已有浏览器调试工作台；权限分级、Textual TUI 与 Tauri GUI 仍在路线图中，见 `DESKTOP_AGENT_PLAN.md`。

首次使用需 `cp .env.example .env` 并填 `LLM_API_KEY`。环境变量全清单见 `.env.example`。

## 调试：LangSmith 追踪

LangSmith 能看到每次 invoke 的**完整 trace** —— 每一步的输入输出、耗时、token 用量、工具调用。
代码里 print 的信息远没有它全，是调试 agent 最快的利器。

**开启：只改 `.env` 三行，代码零改动**（langchain 自动读环境变量）：

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2-...     # 到 https://smith.langchain.com 申请
LANGSMITH_PROJECT=automation-agent
```

**坑**
- 🧩 **值必须是精确的小写 `true`** —— 框架做的是 `== "true"` 精确匹配，写成 `1` 或 `True` 都不生效。
- 🧩 **默认关是有原因的**：判定**只看开关、不看有没有 API key**。没填 key 就开着，
  每次调用都会刷一条 `401 Unauthorized`，但**不影响运行**（追踪失败被 langchain 吞掉，只打日志）。
- 🧩 **追踪失败不会让调用失败** —— 所以「代码跑通了」不等于「追踪通了」。
  验证方法：跑一次，日志里没有 `Failed to multipart ingest runs` 就说明通了；
  然后去 LangSmith 网站按项目名找最新的 run。
- 🧩 **追踪开着会改变模型的调用路径** —— 追踪要记录增量 token，
  于是 LangChain 让模型走流式路径而不是非流式路径。**这会打破测试**：
  假模型给的响应是一次性的有限资源，流式路径消耗得更快，消息不够就 `StopIteration`
  （表现为「Model call failed after 4 attempts」）。
  `tests/conftest.py` 里有个 autouse fixture 在测试期间关掉追踪 —— 实测同一组用例
  开着 6.5 秒、关掉 0.18 秒。**改测试替身或中间件时留意这条。**
- 🧩 **只能改 `.env`，`export` 不管用**。项目用 `load_dotenv(override=True)`，
  `.env` 里的值会盖掉 shell 里的同名环境变量（见陷阱 10）。
  `set_cli_override()` 同样无效 —— 它写的是 `__CLI_LANGSMITH_TRACING__` 这种独立前缀的键，
  而框架只认 `LANGSMITH_TRACING`。**追踪开关只有 `.env` 一个入口。**
- 🧩 **隐私**：追踪会把提示词与模型输出发到 LangSmith 云端。涉及敏感内容先想清楚，
  或用 `PIIMiddleware` 脱敏（见「可选增强」）。
- 📘 03 章是 LangChain 侧的讲法，LangGraph 课件 04 章 9.1 也有（顺带讲了 `langgraph dev` 与 Studio）。

## 学习材料

- `LEARNING.md` — **源码索引**：课件章节 ↔ 源码位置的对照表。想学这个项目先看它。共 6 条主线，第 6 条是 LangGraph 底层。
- `KNOWLEDGE.md` — **知识点总结**：只讲本项目实际用到的 API 是什么、怎么写、坑在哪。当速查手册用。
- `courseware/` — 课件原文，两个子目录：`langChain/`（LangChain 1.x 的 10 章 PDF）+ `langGraph/`（LangGraph 专项课件 6 篇 Markdown）。索引见 `courseware/README.md`。
- `DESKTOP_AGENT_PLAN.md` — 事件流后端与连续对话调试工作台的设计、现状及后续路线图。

## 架构

分两层，边界是**"Agent 做什么" vs "谁在什么时候让它做"**：

```
agent/     — 能力与状态层（课件 05/07/08/09 章）
runtime/   — 运行时外壳（课件未覆盖，需自行实现）
```

### agent/ — 能力与状态层

- `config.py` — 唯一的配置出口。`MODEL` 和 `RUNTIME` 两个 frozen dataclass 在模块导入时从环境变量读一次。改配置只改这里 + `.env`。`MEMORY_BACKEND` 决定记忆后端（`sqlite` 落盘到 `data/memory.sqlite`，`memory` 只在内存）。
- `context.py` — `AgentContext`，每次调用的静态运行时配置（任务名、触发来源、是否允许写）。`trigger` 用 `Literal["schedule", "api", "cli", "lab"]` 限定，新增触发方式要同步改这个类型别名。
- `graph.py` — **装配中心**。`build_agent()` 是唯一创建 Agent 的地方，模型/工具/记忆后端都从这里注入。中间件栈在这里组装：`EVENT_BRIDGE`（事件桥接，**最外层**）→ `LIMITS`（限次）→ `RETRY`（重试）→ `TODO`（任务拆解）→ `CONTEXT_EDITING`（上下文清理）→ `PII`（脱敏）→ `CUSTOM_MIDDLEWARE`（自定义守卫）。
- `tools/` — L1 能力层。`files.py` 的文件工具被沙箱限制在 `data/` 目录内；读文件有 **200,000 字节硬上限**（`MAX_READ_BYTES`，超出即截断并把「还有多少行、传什么 offset」告诉模型），默认每次返回 200 行（`limit` 参数模型可传大，但字节上限始终生效）；二进制文件拒绝；记忆库（`memory.sqlite` 及其 WAL/SHM）禁读 —— 否则 agent 一句「看下这个文件」就能撑爆上下文窗口；`notify.py` 是通知出口，webhook 未配置时降级为终端输出。
  - **沙箱越界不抛异常，返回一句给模型看的中文说明**（`_resolve_safe`）。`_resolve()` 仍然抛 `ValueError`，但那是给 Python 调用方用的；工具层用它会让 `ToolRetryMiddleware`（默认 `retry_on=(Exception,)`）把同一个错参数重试 3 次、空耗约 3 秒，模型最后收到的还是 `failed after 3 attempts ... Please try again.` 这种没法照做的英文提示。
- `middleware/custom.py` — L4 守卫层的自定义部分。三个中间件分别负责只读拦截、上下文注入、耗时统计。
- `middleware/pii.py` — PII 脱敏。配置 langchain 内置的 `PIIMiddleware`，脱敏四种类型：邮箱、信用卡、手机号、身份证号。开关是 `.env` 的 `PII_MASKING`（默认开）。
- `middleware/events.py` — 事件桥接。用 `wrap_tool_call` 把工具执行过程发成 `custom` 流事件（`tool.start` / `tool.end` / `tool.error`），供桌面版的工具卡片用。**8 个工具函数一行都不用改**。它装配在栈的**最外层** —— 包住整个「工具调用 + 重试」，所以一次调用只发一对事件，`duration_ms` 是用户真正等待的时间。

### runtime/ — 运行时外壳

- `runner.py` — `run_task()` 同步调用入口；`get_agent()` 提供共享 Agent 单例。定时任务和 Python 调用方走 `run_task()`，WebSocket 消费器则用同一单例运行 `agent.stream()`。
- `scheduler.py` — APScheduler 封装。**LangGraph 1.x 本地没有内置 cron**（`langgraph_sdk` 也未导出 cron，那是云端 LangGraph Platform 的能力），所以定时触发必须自己接 APScheduler。cron 表达式按 `SCHEDULER_TIMEZONE` 解释，用 `CronTrigger.from_crontab`，即标准 5 段格式。
- `server.py` — FastAPI。定时任务增删查、调度状态、探活、根路径工作台与 **`/ws` 事件流端点**。`/ws` 收 `run.start` / `cancel` / `approval.respond` 命令，推 `run.start` / `approval.request` / `model.text` / `tool.*` / `run.end` / `run.cancelled` / `run.error` 事件。**单一发送点** —— Starlette 的 `WebSocket.send` 没有锁，并发发帧会交错，所以所有消息都进同一个队列由唯一的推送任务发出。
- `events.py` — 事件模型 + 归一化层。`BaseEvent` 加 10 种事件构成桌面版的事件协议；`StreamNormalizer` 把 `agent.stream()` 的 `(模式名, 载荷)` 二元组变成事件。
- `consumer.py` — 同步流消费器（桌面版阶段 3）。在**工作线程**里跑 `agent.stream()`，把归一化后的事件投进 asyncio 队列。两层上限做背压：队列容量管「已入队」、`max_pending` 管「在途未入队」。**不要改成等 `future.result()`** —— 事件循环在 resolve 前关闭时那会永久阻塞。
- `runs.py` — Run Manager（桌面版阶段 4）。管 run 的生命周期：`start` / `get` / `cancel` / `finish` / `active`。**run_id 是一次执行的身份、thread_id 是会话的记忆隔离键，两回事**。第一版单连接单任务 —— 同一 owner 第二次 `start` 抛 `RunBusyError`，由调用方回一条 `run.error`，不静默取消旧的。
- `static/index.html` — 根路径提供的连续对话调试工作台，复用 `thread_id` 展示对话与运行事件。
- `tasks.py` — 定时任务表。改这个文件就改 Agent 每天自动做什么。**注意 `TASKS` 是模块级内存字典，不落盘** —— `POST /tasks` 动态加的任务进程重启即消失，只有改代码里的 `TASKS` 列表才是持久定义。
- `__main__.py` — CLI 入口，唯一子命令 `serve`（默认，起 HTTP 服务 + 可选调度器）。`--no-schedule` / `--reload` / `--host` / `--port` 都在这里解析。**它同时是陷阱 10 的落地文件** —— 调度开关必须在 `import runtime.server` 之前生效，顺序不能改。

### LangGraph 依赖面

项目只用 LangGraph 的这几个面，改代码时别越界：

| API | 位置 | 用途 |
|---|---|---|
| `create_agent()` | `agent/graph.py` | 返回 `CompiledStateGraph`（底层 Pregel）。**项目不手写 `StateGraph`** |
| `InMemorySaver` / `SqliteSaver` | `agent/graph.py` | checkpointer（短期记忆），二选一 |
| `InMemoryStore` | `agent/graph.py` | store（长期记忆）。**目前只挂了没用** —— 全项目没有 `store.put`/`get`/`search` 调用 |
| `Runtime` | `agent/middleware/custom.py` | 运行时上下文，`runtime.context` |
| `langgraph_node` 元数据 | `runtime/events.py` | 过滤中间件插入的例行提示 |
| `thread_id` | `runtime/runner.py` | checkpointer 的会话键 |

**别做的事**：

- 别改成 `create_react_agent` 或手写 `StateGraph` —— 会绕开现有中间件栈（限次 / 重试 / 只读闸门 / 上下文注入）。
- 别指望 `langgraph_sdk` 提供 cron —— 本地没有内置 cron（实测 `langgraph_sdk 0.3.9` 未导出 cron），定时靠 APScheduler。
- `interrupt()` / `Command` / `Send` / `RetryPolicy` 这些高级 API 项目**当前一个都没用**，引入前先看 `DESKTOP_AGENT_PLAN.md` 的设计。

## 换模型须知

换模型只改 `.env` 三行，不用碰代码：

```bash
LLM_PROVIDER=deepseek      # provider 名
LLM_MODEL=deepseek-chat    # 模型名
LLM_BASE_URL=              # 自带端点留空；走兼容端点时填地址
```

**五个前提**（踩到才会坏）：

| 前提 | 说明 |
|---|---|
| **必须支持 function calling** | 硬门槛。没有 `tool_calls`，agent 循环第一步就断了，模型只能当聊天机器人 |
| **provider 包要装** | `init_chat_model` 内置 24 个 provider，但每个要装对应的 `langchain-xxx`。当前环境装好的只有 5 个：deepseek / openai / anthropic / ollama / openrouter |
| **模型名要跟 provider 一起给** | provider 做成必填配置项，不靠自动推断 —— 有些模型名（如 `qwen-plus`）推断会直接报错 |
| **自带端点走 `LLM_BASE_URL`** | OpenAI 兼容的自建端点都走这条 |
| **思考块形态因模型而异** | MiniMax 混在 `content` 里（`runtime/events.py` 的 `ThinkFilter` 处理）；DeepSeek 放 `additional_kwargs.reasoning_content`（content 干净）。换模型后 ThinkFilter 可能变成空转 —— 无害 |

**已实测跑通的组合**：

| 模型 | provider | 备注 |
|---|---|---|
| deepseek-chat | deepseek | 默认配置 |
| MiniMax-M3 | openai + `LLM_BASE_URL=https://api.minimaxi.com/v1` | 走 OpenAI 兼容端点 |
| qwen3.5:9b-mlx | ollama | 本地离线 |

**坑**

- 🧩 **流式响应可能不返回 `usage_metadata`**（MiniMax 就是）。`TimingMiddleware` 有 fallback ——
  打「token 用量不可用（供应商未返回）」而不是假的 0。
- 🧩 **SYSTEM_PROMPT 的服从度因模型而异**。「先用 TodoList 拆解」「每步做完就 notify」
  是靠模型自觉的，不是代码强制的。弱模型可能只回一句话就完事 —— 代码层看不出异常，
  界面体验却差很多。换模型后第一件事就是看它听不听话。
- 🧩 **没有 fallback 模型**。`ModelFallbackMiddleware` 没启用（见「可选增强」），主模型挂了就直接失败。

## 关键 API 陷阱

这几处是实际调试踩出来的，与直觉不符，改代码时注意：

1. **`context` 是 `invoke` 的独立参数，不在 config 里。**
   ```python
   agent.invoke(msgs, {"configurable": {...}}, context=ctx)   # ✅
   agent.invoke(msgs, {"configurable": {...}, "context": ctx})  # ❌ runtime.context 为 None
   ```

2. **`runtime.context` 可能是 None。** context 是可选参数，不传时中间件里取到的是 `None`。`middleware/custom.py` 的 `_ctx()` 做了兜底 —— **新增中间件钩子必须用 `_ctx(runtime)` 而不是直接 `runtime.context`**。

3. **`SqliteSaver.from_conn_string()` 是上下文管理器**，直接传给 `create_agent` 会报 `Invalid checkpointer`（连接会被关掉）。`graph.py` 里直接持有一个 `sqlite3` 连接。

4. **`ContextEditingMiddleware` 参数是 `edits=[ClearToolUsesEdit(...)]`**，不是 `token_limit`/`truncate`。

5. **`create_agent` 收中间件实例，不是类。**

6. **`hook_config` 只用于 `can_jump_to`**，无参装饰器写法是错的。

7. **测试用的假模型必须实现 `bind_tools`** —— `BaseChatModel.bind_tools` 默认抛 `NotImplementedError`，任何带工具的 Agent 都会走到这。`FakeListChatModel` 不支持 tool_calls，要测工具调用链得用 `GenericFakeChatModel`（接受 `AIMessage` 迭代器）。`tests/test_e2e.py` 里的 `ToolCapableFakeModel` 已经处理好了。

8. **`GenericFakeChatModel._generate` 必须返回 `ChatResult`**，不是 `AIMessage`。

9. **`ModelRequest` 是不可变的，改字段要用 `override()`。** `system_prompt` 只是只读 property（返回 `system_message.text`），直接赋值会抛 `DeprecationWarning`。改系统提示词要造新 request：
   ```python
   new_request = request.override(
       system_message=SystemMessage(content=(request.system_message.text or "") + extra)
   )
   return handler(new_request)
   ```
   注意 `override` 是**替换**语义 —— 要在原提示词后追加，得自己把原文拼进去，否则原提示词会被冲掉。`ContextInjectionMiddleware.wrap_model_call` 是正例。

10. **`SCHEDULER_ENABLED` 必须在 import `runtime.server` 之前生效。** `server.py` 在模块导入时就读了 `RUNTIME.scheduler_enabled` 做启动钩子，所以 `runtime/__main__.py` 里 `--no-schedule` 要先写覆盖再 `importlib.reload(agent.config)` 和 `reload(server)`。改启动逻辑时别把这个顺序弄丢了。

    **写覆盖要用 `agent.config.set_cli_override("SCHEDULER_ENABLED", "0")`，不能直接 `os.environ["SCHEDULER_ENABLED"] = "0"`。** 因为 `config.py` 用的是 `load_dotenv(override=True)`（.env 覆盖环境变量），直接设进 `os.environ` 的值会在 reload 时重新读 `.env` 的动作里被顶掉 —— `.env` 里默认带着 `SCHEDULER_ENABLED=1`，`--no-schedule` 形同虚设（`/health` 仍返回 `scheduler: true`）。`set_cli_override` 写的是 `__CLI_SCHEDULER_ENABLED__` 这种独立前缀的键，`.env` 里不会有它，reload 也不会清掉它。

11. **FastAPI 的 `@app.on_event` 已废弃**（fastapi 0.135），要用 `lifespan` 上下文管理器。`server.py` 里是 `lifespan` 写法：启动逻辑放 `yield` 前，清理逻辑放 `yield` 后。

12. **1.x 的 `response_format` 必须显式包策略类**：`create_agent(response_format=Schema)` 直接传 Pydantic 类已不支持，要写 `ToolStrategy(Schema)` / `ProviderStrategy(Schema)` / `AutoStrategy()`。（项目目前没用 `response_format`，但加结构化输出时会撞上。）


13. **只读闸门拦下工具时必须补 ToolMessage。** `TaskGateMiddleware` 拦掉写工具后，要给每个被拦的 `tool_call` 补一条对应 `tool_call_id` 的 `ToolMessage`。少了它，消息历史里留着没有结果的 `tool_call`，下一轮请求被 OpenAI / MiniMax 兼容接口直接 400（`tool call result does not follow tool call`）。**只补 ToolMessage、不补收尾消息** —— agent 图的路由（`langchain/agents/factory.py`）看的是最后一条 AIMessage 的 `tool_calls`：都已回应就判定为「人工注入的工具消息」，跳回模型节点重新回答，于是 agent 会改用文字给结论。这跟官方 `HumanInTheLoopMiddleware` 的拒绝路径一致。

14. **`build_agent(checkpointer=None)` 不是「不要记忆」，是「用默认的」。** 源码写的是 `checkpointer if checkpointer is not None else _build_checkpointer()`，传 None 会建**真实的 sqlite checkpointer**。测试里要显式传 `InMemorySaver()`，否则测试会话会落进 `data/memory.sqlite` 并跨次累积，最后把测试搞挂。

## 可选增强

langchain 1.x 的 `ModelFallbackMiddleware`（模型降级）、`HumanInTheLoopMiddleware`（人工审批）、`SummarizationMiddleware`（摘要压缩）都可用，但当前**没有启用**：

- 降级需要显式给 fallback 模型（`ModelFallbackMiddleware("openai:gpt-4o-mini")`），要先决定备用供应商和 key 从哪来，属于配置决策。
- 人工审批适合交互式场景，与"无人值守定时任务"的项目定位冲突。
- 摘要压缩与已有的 `ContextEditingMiddleware`（`ClearToolUsesEdit`）功能部分重叠。
- ~~`PIIMiddleware`~~ —— **已启用**，见 `agent/middleware/pii.py`。用 `redact` 而不是 `mask`：mask 只保留末尾几位（手机号是 `****5678`），那几位仍是真实个人信息，照样会进供应商和 LangSmith。

要启用时在 `agent/graph.py` 的中间件栈里加，不要直接改 `RETRY` 列表的语义。

## 约定

- **记忆隔离靠 `thread_id`**。定时任务用 `cron-{任务名}` 作为 thread_id，保留跨天记忆；`runner.run_task()` 不传 thread_id 时生成随机 `task-{uuid}`。**同一个 thread_id 就共享同一份记忆**，客户端想续聊就显式传同一个值。
- **新增工具要登记两处**：加入 `agent/tools/__init__.py` 的 `ALL_TOOLS`，并在 `agent/tools/risk.py` 声明风险等级。工具清单与风险表在导入时自动校验；不能再维护独立的 `WRITE_TOOLS` 名单。
- **权限决策由 `agent/tools/risk.py` 集中定义**：只读等级自动放行；有副作用的等级返回 `require_approval`；无风险登记返回 `deny`。`TaskGateMiddleware` 负责拦截未授权调用。浏览器工作台通过 `approval.request` / `approval.respond` 逐次确认；旧 Python API、定时任务和旧 WebSocket 客户端显式设置 `allow_write=True` 时仍保持整轮授权兼容。
- **通知出口是 `notify` / `notify_summary`**，不是直接 print。SYSTEM_PROMPT 要求 Agent 每步做完就推送，不要攒到最后。
- **PII 脱敏默认开着**（`.env` 的 `PII_MASKING=1`）。它脱敏**用户指令**与**工具结果**
  （文件工具读出来的内容走后者，是最主要的泄露通道），但**不脱敏模型输出** ——
  模型输出多半来自用户自己的输入，用户自己知道那是什么，
  脱敏输出会让「提醒我邮箱是多少」这种场景失效。脱敏是全局的，
  想让 agent 处理一份本来就该被它看到的通讯录时把开关关掉。
- **事件协议在 `runtime/events.py`** —— 桌面版的事件协议。新增事件类型要同时改三处：
  事件类、`EVENT_CLASSES` 注册表、`StreamNormalizer` 的分派规则。
  `version` 是协议版本，改事件结构时要动它 —— 客户端靠它识别新旧格式。
- **中间件顺序有意义**：`LIMITS` 放最前（先挡住跑飞），`CUSTOM_MIDDLEWARE` 放最后（能看到前面中间件的结果）。
- **测试会真写 `data/` 目录**（`data/out/report.md` 等），`data/` 已在 `.gitignore` 里，跑测试不必清理。
- 提交前跑 `pytest tests/ -q`，全部测试必须通过；测试数量以当次输出为准。
