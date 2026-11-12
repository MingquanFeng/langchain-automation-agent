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

服务起在 `http://127.0.0.1:8000`。打开根地址即可与 Agent 连续聊天：后续消息自动沿用当前会话记忆，并可实时查看工具调用或取消运行。

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/` | 本地 Agent 调试工作台 |
| GET | `/health` | 健康检查 |
| GET | `/tasks` | 列出定时任务 |
| POST | `/tasks` | 动态添加定时任务 |
| DELETE | `/tasks/{name}` | 删除定时任务 |
| GET | `/jobs` | 已注册 job 与下次运行时间 |

WebSocket `/ws` 是工作台使用的实时事件流接口。只读工具会直接运行；写文件或发送通知时，聊天窗口会显示工具用途和参数，并等待你选择“仅允许本次”或“拒绝本次”。每次批准只对当前工具调用生效，取消任务或关闭连接时，等待中的工具不会执行。

授权协议的事件形状如下。客户端必须把请求中的 `run_id` 和 `call_id` 原样带回；服务端只接受发起该 run 的连接所作的应答。

```json
{"type":"approval.request","run_id":"run-a1b2c3d4","call_id":"call-1","name":"write_file","args":{"path":"out/report.md","content":"..."},"risk_level":"workspace_write"}
{"type":"approval.respond","run_id":"run-a1b2c3d4","call_id":"call-1","approved":true}
```

浏览器工作台不会发送整轮 `allow_write`。旧 WebSocket 客户端仍可显式传 `allow_write: true`，但这会按旧规则一次性允许本轮写操作；Python API 和定时任务也继续支持原有设置。

REST 接口只管理定时任务和查询状态；浏览器工作台通过 WebSocket `/ws` 触发交互式任务。也可以从 Python 直接调用：

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
  server.py          FastAPI，定时任务接口、浏览器工作台、WebSocket 事件流
  events.py          事件模型与流归一化
  consumer.py        工作线程中的 agent.stream() 消费器与跨线程背压
  runs.py             WebSocket run 生命周期、取消和清理
  static/index.html  连续对话调试工作台
  tasks.py           定时任务表，改这里就改 Agent 每天自动做什么
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
