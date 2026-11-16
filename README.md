# langchain-automation-agent

基于 **LangChain 1.x / LangGraph 1.x** 的自动化 Agent 学习骨架。

把「Agent 做什么」和「谁在什么时候让它做」拆成两层：

| 层 | 职责 | 对应内容 |
|---|---|---|
| `agent/` | 能力与状态层 | 工具、中间件栈、运行时上下文、记忆后端 |
| `runtime/` | 运行时外壳 | APScheduler 定时任务、FastAPI 服务、事件流、连续对话工作台 |

源码带有教程章节索引，是一份**可运行的学习教材**，而不是业务系统。

## 特点

- **1.x 新写法**：`create_agent` + 中间件体系 + `state`/`store`/`checkpointer`
- **中间件栈完整**：事件桥接 → 限次 → 重试 → 任务拆解 → 上下文清理 → 脱敏 → 自定义守卫，顺序可考
- **测试零成本**：测试使用假模型，不调用真实模型 API
- **学习材料齐备**：`LEARNING.md` 源码索引，课件章节 ↔ 源码位置一一对应；`KNOWLEDGE.md` 把相关知识点浓缩成速查手册

## 快速开始

### 1. 装依赖

需要 Python 3.13+：

```bash
pip install langchain langchain-core langgraph langgraph-checkpoint-sqlite \
            fastapi uvicorn httpx pydantic python-dotenv apscheduler pytest
```

按需补装模型供应商包与追踪工具：

```bash
pip install langchain-deepseek langchain-openai   # 模型供应商
pip install langsmith                             # 可观测性（LangSmith）
```

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env` 填入 `LLM_API_KEY`。其余配置项都有默认值，见 `.env.example`。

### 3. 跑测试

```bash
python -m pytest tests/ -q
```

测试全部使用假模型，不需要 API key；运行结果以本地实际 pytest 输出为准。

### 4. 起服务

```bash
python -m runtime                # 含定时任务
python -m runtime --no-schedule  # 启动服务，但不启动定时任务
python -m runtime --reload       # 开发模式热重载
```

服务默认监听 `http://127.0.0.1:8000`，提供桌面工作台使用的 HTTP API 与 WebSocket。聊天界面随 Tauri 应用打包；`python -m runtime` 单独启动时不再提供浏览器工作台。

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/tasks` | 列出定时任务 |
| POST | `/tasks` | 动态添加定时任务 |
| DELETE | `/tasks/{name}` | 删除定时任务 |
| GET | `/jobs` | 已注册 job 与下次运行时间 |
| GET | `/runs?limit=50&offset=0` | 分页查询最近执行记录及结果 |
| GET | `/threads/messages?thread_id=...` | 查询会话检查点中的对话消息 |
| DELETE | `/runs/{run_id}` / `/runs` | 删除单条或清空运行记录 |
| DELETE | `/threads?thread_id=...` | 删除会话检查点及关联运行记录 |

任务表在首次启动时从 `runtime/tasks.py` 的默认项初始化，之后的动态增删改会保存在运行时 SQLite 数据库，重启后继续生效。运行记录包含指令、最终输出或错误、触发来源、状态、耗时和会话 ID，默认保存在 `data/runtime.sqlite3`（桌面版位于应用支持目录；自定义 `SQLITE_PATH` 时使用同目录）。默认每页 50 条，最多 100 条。工作台“运行历史”可按时间浏览状态、指令、结果和耗时；恢复会话会读取 Checkpointer 消息并沿用原 `thread_id`，不会自动重发任务。单条运行记录删除不会删除会话记忆；删除整个会话会一并删除该会话的所有运行记录与 Checkpointer 数据；清空运行记录会保留会话记忆。

WebSocket `/ws` 是桌面工作台使用的实时事件流接口。只读工具会直接运行；写文件或发送通知时，聊天窗口会显示工具用途和参数，并等待你选择“仅允许本次”或“拒绝本次”。每次批准只对当前工具调用生效，取消任务或关闭连接时，等待中的工具不会执行。

文件工具支持目录树、按名称查找文件和按正文定位源码。扫描始终限制在当前工作目录内，跳过隐藏目录、依赖与构建产物、敏感配置和密钥文件；正文搜索不解释正则表达式，并限制单文件读取量、单行预览和结果条数。

现有文本可通过结构化编辑工具精确替换。工作台在审批卡中展示统一 diff，不把完整 `old_text` / `new_text` 重复写入审批事件；批准后会再次校验文件未在等待期间变化，覆盖前把原文保存到应用数据目录的隐藏备份库。备份列表不返回正文，恢复操作同样先展示反向 diff，并为恢复前状态创建新的备份。

模型请求默认使用 `LLM_REQUEST_TIMEOUT=60` 秒的网络 I/O 超时；SDK 重试关闭，由图统一执行最多 3 次额外重试。模型重试耗尽后记录为失败。通知 webhook 保留 15 秒 I/O 超时；这些超时不代表整轮任务的总时限。取消是协作式的：工作台先显示“正在取消”，当前同步调用退出后才显示“已取消”，已完成的操作不会回滚。同一 Python 进程中，桌面、WebSocket、同步任务及会话删除共用 `thread_id` 占用，避免同时修改同一会话；多进程共用数据库仍需额外协调。

授权协议的事件形状如下。客户端必须把请求中的 `run_id` 和 `call_id` 原样带回；服务端只接受发起该 run 的连接所作的应答。

```json
{"type":"approval.request","run_id":"run-a1b2c3d4","call_id":"call-1","name":"write_file","args":{"path":"out/report.md","content":"..."},"risk_level":"workspace_write"}
{"type":"approval.respond","run_id":"run-a1b2c3d4","call_id":"call-1","approved":true}
```

桌面托管服务要求本次启动的连接凭证，拒绝 WebSocket 请求中的整轮授权。有副作用的工具须逐次审批。独立启动的服务继续兼容旧客户端显式传递的布尔值 `allow_write: true`，Python API 和定时任务也保留原有设置。

REST 接口管理定时任务、查询运行状态和历史；桌面工作台通过 WebSocket `/ws` 触发交互式任务。也可以从 Python 直接调用：

```bash
# Python API（allow_write 默认 false，即只读，写工具会被闸门拦下）
python -c "
from runtime.runner import run_task
print(run_task(instruction='列出数据目录下的文件', allow_write=True)['output'])
"
```

更完整的桌面版路线见 [DESKTOP_AGENT_PLAN.md](DESKTOP_AGENT_PLAN.md)。

## 代码导览

从 [LEARNING.md](LEARNING.md) 开始。它把教程章节和源码位置串成六条主线：

1. **模型怎么变成 Agent** —— `create_agent` 的参数注入模型、工具、中间件、记忆
2. **工具** —— `@tool` 装饰器、docstring 如何影响模型选工具、沙箱边界
3. **中间件** —— Node-style 与 Wrap-style 两种钩子，以及执行顺序规则
4. **上下文与记忆** —— `AgentContext` / `state` / `store` / `checkpointer` 四者分工
5. **运行时外壳** —— 定时任务、HTTP 服务、事件归一化
6. **LangGraph 底层** —— checkpointer / store / 运行时上下文 / 图的路由

每条主线给出源码位置、对应课件章节和这个知识点的关键认知。
只想拿结论、不想通读课件的话，看 [KNOWLEDGE.md](KNOWLEDGE.md)。

## 目录结构

```
agent/
  config.py          唯一配置出口（环境变量 → frozen dataclass）
  context.py         AgentContext，本次调用的静态运行时配置
  graph.py           装配中心，build_agent() 是唯一创建 Agent 的地方
  tools/             L1 能力层：文件工具（沙箱限制在 data/）+ 通知出口
  middleware/        中间件：只读闸门、上下文注入、耗时统计 + PII 脱敏
runtime/
  runner.py          同步 run_task() 入口；提供共享的 get_agent() 单例
  scheduler.py       APScheduler 封装（LangGraph 1.x 本地没有内置 cron）
  server.py          FastAPI，定时任务与运行历史 API、WebSocket 事件流
  storage.py         SQLite 持久化定时任务与运行记录
  events.py          事件模型与流归一化
  consumer.py        工作线程中的 agent.stream() 消费器与跨线程背压
  runs.py             WebSocket run 生命周期、取消和清理
  tasks.py           定时任务表，改这里就改 Agent 每天自动做什么
desktop/web/
  index.html         Tauri 本机服务配置与生命周期连接页
  workbench.html     随桌面应用打包的聊天与运行历史工作台
  assets/            桌面品牌资源
tests/               单元测试与集成测试（含假模型）
```

## 实现要点

几个容易踩的坑，代码里都做了处理，也记录在 [CLAUDE.md](CLAUDE.md)：

- `context` 是 `invoke` 的独立参数，**不在 config 里**；`runtime.context` 可能为 `None`，钩子里必须兜底
- `ModelRequest` 不可变，改字段用 `override()`，且 `override` 是替换语义
- `SqliteSaver.from_conn_string()` 是上下文管理器，不能直接传给 `create_agent`
- `create_agent` 收中间件**实例**，不是类
- 假模型必须实现 `bind_tools`，否则 `NotImplementedError`
- 1.x 的 `response_format` 必须显式包 `ToolStrategy(Schema)`

## 教程对应

源码注释里的「对应课件 XX 章 X.X 节」指向一套 LangChain 1.x 中文课程（PDF 共 10 章：概述 / 模型调用 / LangSmith / Message 与提示词 / Tools / 结构化输出 / 智能体 / 中间件 / 上下文与记忆 / RAG）。

课件 PDF 原件就在本仓库的 `courseware/langChain/` 目录（10 章），章节索引见 `courseware/README.md`。

## 说明

这是一个**学习用途**的项目：代码精简、注释偏教学，没有生产级的鉴权、限流、审计与错误重试策略，请勿直接用于生产环境。
