# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目性质

这是一个**学习驱动**的项目，不是生产系统。`automation_agent/` 是边学边搭的自动化 Agent 骨架，课件在上一级的 `../langchain学习/01-课件/`（尚硅谷 LangChain 1.x 的 10 章 PDF + 配套资料）。

课件对每个知识点都有对应章节，源码注释里用「对应课件 XX 章 X.X 节」标注了出处。**改代码前先看对应课件章节**，保持实现与课件 API 一致；只有当课件 API 与 langchain 1.2 实际行为冲突时，才按实际行为写并在注释里说明差异。

PDF 文件名即章节号（`尚硅谷-05-Tools.pdf` → 课件 05 章）。本仓库没有 README，这份文件就是唯一的项目说明。

> 当前目录**尚未 git init**（`git status` 报「不是 Git 仓库」）。要提交需先初始化仓库。

## 环境

固定使用 conda 环境 `langchain1.2`（Python 3.13.12），不要用系统 Python：

```bash
PY=/opt/miniconda3/envs/langchain1.2/bin/python
```

`conda` shell 函数在 zsh 里是坏的（`CONDA_EXE` 未设置，报 permission denied），一律用绝对路径调用：

```bash
/opt/miniconda3/bin/conda env list
```

依赖基线来自 `../langchain学习/01-课件/requirements.txt`（57 项，LangChain/LangGraph/MCP/模型供应商/Web/配置/日志/记忆，RAG 及以后章节被注释掉）。本项目在此之上额外装了：`apscheduler`（定时）、`langgraph-cli`（服务化）、`langgraph-checkpoint-sqlite`（持久化记忆）。

已装关键版本：`langchain 1.2.12`、`langchain-core 1.2.18`、`langgraph 1.1.2`、`langgraph-checkpoint-sqlite 3.1.2`、`apscheduler 3.11.3`、`fastapi 0.135.1`、`uvicorn 0.46.0`、`httpx 0.28.1`。

> **课件的 requirements.txt 是 UTF-16 LE 编码。** 直接 `pip install -r` 会乱码，需先 `iconv -f UTF-16LE -t UTF-8` 转码。

## 常用命令

所有命令都在本目录（`automation_agent/`）下执行。

```bash
# 跑测试（35 项，全部用假模型，不调真实 API）
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

# 交互式终端（REPL）
$PY -m runtime chat

# 跑学习实验（不需要 API key，用假模型）
$PY lab/01_agent_loop.py
$PY lab/02_middleware.py
$PY lab/03_memory.py
```

**本项目只有 pytest 一道验证关卡**，没有 lint、type-check、build 步骤（纯 Python 包，无打包产物）。提交前的验证底线就是 `pytest tests/ -q` 35 项全过 + 按实际需求跑一次服务确认能起来。

服务起来后的 HTTP 端点：

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/health` | 健康检查 + 调度器开关状态 |
| GET | `/tasks` | 列出定时任务表 |
| POST | `/tasks` | 动态添加定时任务（cron + 指令） |
| DELETE | `/tasks/{name}` | 删除定时任务 |
| GET | `/jobs` | 列出调度器已注册的 job 与下次运行时间 |
| POST | `/run` | 同步触发一次任务（默认只读，`allow_write=true` 才开写权限） |
| POST | `/run/stream` | SSE 流式触发（同上，默认只读） |

首次使用需 `cp .env.example .env` 并填 `LLM_API_KEY`。环境变量全清单见 `.env.example`。

## 学习材料

- `LEARNING.md` — **学习地图**：课件章节 ↔ 源码位置 ↔ lab 实验 ↔ 练习的对照表。想学这个项目先看它。
- `lab/` — 动手实验脚本，**不需要 API key**（用脚本化的假模型编排，不花钱也不限流）：
  - `lab/01_agent_loop.py` — 看清 agent 循环：模型输出的 tool_calls 如何变成 ToolMessage 回灌
  - `lab/02_middleware.py` — 看清中间件如何改变行为（只读闸门如何拦截写操作）
  - `lab/03_memory.py` — 看清 thread_id 如何隔离记忆

## 架构

分两层，边界是**"Agent 做什么" vs "谁在什么时候让它做"**：

```
agent/     — 能力与状态层（课件 05/07/08/09 章）
runtime/   — 运行时外壳（课件未覆盖，需自行实现）
```

### agent/ — 能力与状态层

- `config.py` — 唯一的配置出口。`MODEL` 和 `RUNTIME` 两个 frozen dataclass 在模块导入时从环境变量读一次。改配置只改这里 + `.env`。`MEMORY_BACKEND` 决定记忆后端（`sqlite` 落盘到 `data/memory.sqlite`，`memory` 只在内存）。
- `context.py` — `AgentContext`，每次调用的静态运行时配置（任务名、触发来源、是否允许写）。`trigger` 用 `Literal["schedule", "api", "cli", "lab"]` 限定，新增触发方式要同步改这个类型别名。
- `graph.py` — **装配中心**。`build_agent()` 是唯一创建 Agent 的地方，模型/工具/记忆后端都从这里注入。中间件栈在这里组装：`LIMITS`（限次）→ `RETRY`（重试）→ `TODO`（任务拆解）→ `CONTEXT_EDITING`（上下文清理）→ `CUSTOM_MIDDLEWARE`（自定义）。
- `tools/` — L1 能力层。`files.py` 的文件工具被沙箱限制在 `data/` 目录内（`_resolve` 越界抛 `ValueError`）；读文件有 200KB 上限、二进制文件拒绝、记忆库（`memory.sqlite`）禁读 —— 否则 agent 一句「看下这个文件」就能撑爆上下文窗口；`notify.py` 是通知出口，webhook 未配置时降级为终端输出。
- `middleware/custom.py` — L4 守卫层的自定义部分。三个中间件分别负责只读拦截、上下文注入、耗时统计。

### runtime/ — 运行时外壳

- `runner.py` — 唯一的 Agent 调用入口，调度器和 HTTP 服务都走它，保证行为一致。`_AGENT` 单例复用，保留 checkpointer 里的记忆。
- `scheduler.py` — APScheduler 封装。**LangGraph 1.x 本地没有内置 cron**（`langgraph_sdk` 也未导出 cron，那是云端 LangGraph Platform 的能力），所以定时触发必须自己接 APScheduler。cron 表达式按 `SCHEDULER_TIMEZONE` 解释，用 `CronTrigger.from_crontab`，即标准 5 段格式。
- `server.py` — FastAPI。`/run` 同步，`/run/stream` 走 SSE 流式。
- `repl.py` — 交互式终端，`python -m runtime chat`。斜杠命令控会话（`/new` `/write` `/think`），其余输入当指令发给 Agent。走 `stream_mode=["messages","values"]`：增量文本从 messages 流取，工具调用从 values 流里的 `tool_calls` 认（messages 模式下非流式模型不吐 `tool_call_chunks`）。
- `tasks.py` — 定时任务表。改这个文件就改 Agent 每天自动做什么。**注意 `TASKS` 是模块级内存字典，不落盘** —— `POST /tasks` 动态加的任务进程重启即消失，只有改代码里的 `TASKS` 列表才是持久定义。

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

12. **1.x 的 `response_format` 必须显式包策略类**：`create_agent(response_format=Schema)` 直接传 Pydantic 类已不支持，要写 `ToolStrategy(Schema)` / `ProviderStrategy(Schema)` / `AutoStrategy()`。这是 0.x 到 1.x 常见踩坑点（项目目前没用 `response_format`，但加结构化输出时会撞上）。

13. **别从网上抄 0.x 的写法。** 0.x 的老 API 没删，都挪进了 `langchain-classic` 包 —— 所以能 import 但不该用：`AgentExecutor`、`create_react_agent`、`create_tool_calling_agent`、`create_structured_chat_agent`、`ConversationBufferMemory` 这类 `xxxMemory` 类、`JsonOutputParser` 拼提示词。1.x 对应的新写法见 `LEARNING.md` 开头的版本对照表。

14. **只读闸门拦下工具时必须补 ToolMessage。** `TaskGateMiddleware` 拦掉写工具后，要给每个被拦的 `tool_call` 补一条对应 `tool_call_id` 的 `ToolMessage`。少了它，消息历史里留着没有结果的 `tool_call`，下一轮请求被 OpenAI / MiniMax 兼容接口直接 400（`tool call result does not follow tool call`）。**只补 ToolMessage、不补收尾消息** —— agent 图的路由（`langchain/agents/factory.py`）看的是最后一条 AIMessage 的 `tool_calls`：都已回应就判定为「人工注入的工具消息」，跳回模型节点重新回答，于是 agent 会改用文字给结论。这跟官方 `HumanInTheLoopMiddleware` 的拒绝路径一致。

15. **`build_agent(checkpointer=None)` 不是「不要记忆」，是「用默认的」。** 源码写的是 `checkpointer if checkpointer is not None else _build_checkpointer()`，传 None 会建**真实的 sqlite checkpointer**。测试里要显式传 `InMemorySaver()`，否则测试会话会落进 `data/memory.sqlite` 并跨次累积，最后把测试搞挂。

## 可选增强

langchain 1.x 的 `ModelFallbackMiddleware`（模型降级）、`HumanInTheLoopMiddleware`（人工审批）、`SummarizationMiddleware`（摘要压缩）、`PIIMiddleware`（敏感信息脱敏）都可用，但当前**没有启用**：

- 降级需要显式给 fallback 模型（`ModelFallbackMiddleware("openai:gpt-4o-mini")`），要先决定备用供应商和 key 从哪来，属于配置决策。
- 人工审批适合交互式场景，与"无人值守定时任务"的项目定位冲突。
- 摘要压缩与已有的 `ContextEditingMiddleware`（`ClearToolUsesEdit`）功能部分重叠。
- `PIIMiddleware(pii_type="email", strategy="mask")` 值得加 —— 防止手机号、邮箱这类个人信息被发到模型供应商那边。

要启用时在 `agent/graph.py` 的中间件栈里加，不要直接改 `RETRY` 列表的语义。

## 约定

- **记忆隔离靠 `thread_id`**。定时任务用 `cron-{任务名}` 作为 thread_id，保留跨天记忆；`POST /run` 每次生成随机 `task-{uuid}`；`POST /run/stream` 默认生成随机 `stream-{uuid}`，只有显式传 `thread_id` 才复用会话。
- **写操作受 `AgentContext.allow_write` 控制，默认关闭**（只读是默认态，写权限要显式开启：`AgentContext(allow_write=True)` / `/run` 传 `allow_write=true` / `ScheduledTask(allow_write=True)`）。只读模式由 `TaskGateMiddleware` 拦截。注意 `notify` / `notify_summary` 也算写工具，需要推送的任务必须显式开权限。新增写类工具时，要同步更新 `TaskGateMiddleware.WRITE_TOOLS`。
- **通知出口是 `notify` / `notify_summary`**，不是直接 print。SYSTEM_PROMPT 要求 Agent 每步做完就推送，不要攒到最后。
- **中间件顺序有意义**：`LIMITS` 放最前（先挡住跑飞），`CUSTOM_MIDDLEWARE` 放最后（能看到前面中间件的结果）。
- **测试会真写 `data/` 目录**（`data/out/report.md` 等），`data/` 已在 `.gitignore` 里，跑测试不必清理。
- 提交前跑 `pytest tests/ -q`，35 项必须全过。
