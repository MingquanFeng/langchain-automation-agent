# langchain-automation-agent

基于 **LangChain 1.x / LangGraph 1.x** 的自动化 Agent 学习骨架。

把「Agent 做什么」和「谁在什么时候让它做」拆成两层：

| 层 | 职责 | 对应内容 |
|---|---|---|
| `agent/` | 能力与状态层 | 工具、中间件栈、运行时上下文、记忆后端 |
| `runtime/` | 运行时外壳 | APScheduler 定时任务、FastAPI 服务、任务表 |

整个项目约 1100 行，每段源码都标注了对应的教程章节，是一份**可运行的学习教材**，而不是业务系统。

## 特点

- **1.x 新写法**：`create_agent` + 中间件体系 + `state`/`store`/`checkpointer`，不混用 0.x 的 `AgentExecutor`
- **中间件栈完整**：限次 → 重试 → 任务拆解 → 上下文清理 → 自定义守卫，顺序可考
- **测试零成本**：35 项端到端测试全部用假模型，不调用任何真实 API
- **学习材料齐备**：`LEARNING.md` 学习地图 + `lab/` 三个免 API key 的动手实验

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

预期输出 `35 passed`。全部使用假模型，不需要 API key。

### 4. 起服务

```bash
python -m runtime                # 含定时任务
python -m runtime --no-schedule  # 只起 HTTP 服务
python -m runtime --reload       # 开发模式热重载
python -m runtime chat           # 交互式终端（REPL）
```

服务起在 `http://127.0.0.1:8000`：

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/tasks` | 列出定时任务 |
| POST | `/tasks` | 动态添加定时任务 |
| DELETE | `/tasks/{name}` | 删除定时任务 |
| GET | `/jobs` | 已注册 job 与下次运行时间 |
| POST | `/run` | 同步触发一次任务 |
| POST | `/run/stream` | SSE 流式触发 |

```bash
# allow_write 默认 false：只读模式，写工具（notify / write_file 等）会被拦截。
# 需要写权限时显式传 allow_write=true
curl -X POST http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"列出数据目录下的文件并汇总通知","allow_write":true}'
```

## 学习路径

从 [LEARNING.md](LEARNING.md) 开始。它把教程章节、源码位置和练习串成五条主线：

1. **模型怎么变成 Agent** —— `create_agent` 的参数注入模型、工具、中间件、记忆
2. **工具** —— `@tool` 装饰器、docstring 如何影响模型选工具、沙箱边界
3. **中间件** —— Node-style 与 Wrap-style 两种钩子，以及执行顺序规则
4. **上下文与记忆** —— `AgentContext` / `state` / `store` / `checkpointer` 四者分工
5. **运行时外壳** —— 定时任务、HTTP 服务、会话隔离

每个实验脚本都不需要 API key：

```bash
python lab/01_agent_loop.py   # 看清 agent 循环：tool_calls 如何变成 ToolMessage 回灌
python lab/02_middleware.py   # 看清中间件如何改变行为（只读闸门）
python lab/03_memory.py       # 看清 thread_id 如何隔离记忆
```

## 目录结构

```
agent/
  config.py          唯一配置出口（环境变量 → frozen dataclass）
  context.py         AgentContext，本次调用的静态运行时配置
  graph.py           装配中心，build_agent() 是唯一创建 Agent 的地方
  tools/             L1 能力层：文件工具（沙箱限制在 data/）+ 通知出口
  middleware/        自定义中间件：只读闸门、上下文注入、耗时统计
runtime/
  runner.py          唯一 Agent 调用入口，调度器和 HTTP 都走它
  scheduler.py       APScheduler 封装（LangGraph 1.x 本地没有内置 cron）
  server.py          FastAPI，/run 同步、/run/stream 流式
  repl.py            交互式终端，python -m runtime chat
  tasks.py           定时任务表，改这里就改 Agent 每天自动做什么
lab/                 动手实验脚本
tests/               端到端测试
```

## 交互式终端

不想用 HTTP 接口时，直接在终端里跟 Agent 聊：

```bash
python -m runtime chat
```

```
› 用一句话介绍你自己
🤖 我是一个自动化日常事务助手，会用 Todo 清单拆解任务、逐条推进……

› 上面那个助手叫什么？
🤖 它没有名字，我刚才只做了自我介绍……
```

斜杠命令做控制，其余输入都是给 Agent 的指令：

| 命令 | 作用 |
|---|---|
| `/help` | 显示帮助 |
| `/new` | 开新会话（换 thread_id，上下文清空） |
| `/thread` | 显示当前会话 ID |
| `/write on\|off` | 开关写权限（默认 off，只读） |
| `/think on\|off` | 是否显示模型思考过程（默认 off） |
| `/exit` | 退出（Ctrl-D、Ctrl-C 同效） |

会话 ID 默认是 `cli-chat`，重启终端还能接着上次聊。

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

本仓库只含代码，课程资料需自行获取。

## 说明

这是一个**学习用途**的项目：代码精简、注释偏教学，没有生产级的鉴权、限流、审计与错误重试策略，请勿直接用于生产环境。
