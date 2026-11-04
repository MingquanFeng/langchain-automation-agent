# 学习地图

这个项目只有 4200 行（去掉空行和纯注释约 1900 行），但它把 **langchain 1.x agent 开发的每个知识点都落到了可运行的代码上**。
它的价值不在业务功能，而在于「每段代码都标了对应课件章节，每个知识点都有对应的实现」。

## 怎么用这张地图

源码注释里的「对应课件 XX 章 X.X 节」就是索引。本文档把两样东西串起来：

```
课件小节  →  该读哪几个文件（按顺序）
```

每条主线先给源码位置，再给这个知识点的关键认知和容易踩的坑 —— 看代码时对照着看，
比通读课件更快定位到自己要用的那部分。

课件原文就在本仓库的 `courseware/` 目录（尚硅谷 LangChain 1.x 的 10 章 PDF 原件），
文件名的数字即章号 —— `courseware/langChain/尚硅谷-05-Tools.pdf` 就是课件 05 章。
章节与源码的对应索引见 `courseware/README.md`。

另外 `courseware/langGraph/` 是一套 **LangGraph 专项课件**（6 篇 Markdown，章节号 1-13 连续），
讲的是 `create_agent` 底下那层图运行时 —— 对应关系见下面「主线 6」。

> **不想通读 10 章课件、只想拿结论**：直接看 `KNOWLEDGE.md` —— 那是把课件里与本项目相关的部分
> 压缩后的知识本体（API 签名 + 项目用法 + 坑），按「一次调用发生了什么」组织，不是章节摘抄。

---

## 主线 1：模型怎么变成 Agent（课件 02 / 04 / 07 章）

**读源码**：`agent/graph.py` → `agent/config.py`

| 位置 | 看什么 | 对应课件 |
|---|---|---|
| `agent/graph.py:422` `build_agent()` | 唯一创建 Agent 的地方。`create_agent(model, tools, system_prompt, middleware, checkpointer, store, context_schema, name)` | 07 章 1.4.2「全新的调用」 |
| `agent/graph.py:634` `SYSTEM_PROMPT` | 系统提示词就是普通字符串，Agent 的一切行为从它开始 | 07 章 6 |
| `agent/config.py:187` `ModelConfig.build()` | `init_chat_model(model=..., model_provider=...)` | 02 章 2.2「兼容用法」 |
| `runtime/runner.py:166` | `HumanMessage` 构造输入 | 04 章 1.5.2 |

**关键认知**：「Agent」不是一个需要实例化的类，而是一次 `create_agent()` 调用返回的**编译好的图**。
模型、工具、中间件、记忆全是**注入参数** —— 所以换模型/换工具不用改调用方，这就是 `build_agent()` 存在的意义。

**延伸**：课件 07 章 8.2 讲了 7 种 `stream_mode`（values / updates / messages / tasks / debug / checkpoints / custom）。
**项目目前没有流式消费者** —— 桌面版做事件流时会重新用起 `values` + `messages` + `custom`（见主线 6）。

---

## 主线 2：工具（课件 05 章）

**读源码**：`agent/tools/files.py` → `agent/tools/notify.py` → `agent/tools/__init__.py`

| 位置 | 看什么 | 对应课件 |
|---|---|---|
| `agent/tools/files.py:307` `@tool(parse_docstring=True)` | 装饰器把函数变成工具。**docstring 就是给模型看的工具说明**；开了 `parse_docstring` 后 `Args:` 段会进 schema 的参数说明 | 05 章 2.2「工具描述的各部分」/ 3.1 |
| `agent/tools/files.py:105` `_resolve()` / `:126` `_resolve_safe()` | 沙箱。越界时工具层返回中文说明（不抛异常，避免被重试 3 次） | 05 章 1.3「工具调用的整体流程」 |
| `agent/tools/files.py:418` `save_json` | 参数标了 `dict | list`，langchain 自动转成 JSON schema | 05 章 2.2.4「参数类型」 |
| `agent/tools/notify.py:99` `_deliver()` | 通知有 webhook 和终端两条路，未配置时降级 | — |

**关键认知**：模型**不执行你的 Python 函数**。它只输出一段结构化的 `tool_calls`（名字 + JSON 参数 + id），
由框架代为执行，再把返回值包成 `ToolMessage` 回灌进对话 —— 这个来回就是 agent 循环。

**两条工具设计规范**：
- 工具数量 **2-5 个最佳**（**07 章** P21：「只给 Agent 需要的工具，工具太多会混淆。一般 2-5 个工具最佳」）—— 本项目有 8 个（6 个文件 + 2 个通知），已经偏多，模型选错工具的概率会上升。
- docstring 要写 **Google 风格**（参数逐个说明，**05 章** 3.1）—— 项目现在 8 个工具全部开了 `parse_docstring=True` 且写成 Google 风格，参数说明会进 schema 的 `properties.<arg>.description`，模型不用靠参数名猜。

---

## 主线 3：中间件（课件 08 章）—— 本项目精华

**读源码**：`agent/graph.py:470-491`（内置栈的组装处）→ `agent/middleware/custom.py`（三个自定义）

内置中间件栈：

| 位置 | 中间件 | 作用 | 对应课件 |
|---|---|---|---|
| `agent/graph.py:249` `LIMITS` | `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` | **最前**，先挡住跑飞，防止无人值守时烧钱 | 08 章 3.1 / 3.2 |
| `agent/graph.py:192` `RETRY` | `ModelRetryMiddleware` / `ToolRetryMiddleware` | 模型/工具失败时重试 | 08 章 3.6 / 3.5 |
| `agent/graph.py:619` `TODO` | `TodoListMiddleware` | 注入 `write_todos` 工具，让模型拆解多步任务 | 08 章 2.4 |
| `agent/graph.py:153` `CONTEXT_EDITING` | `ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=120_000, keep=3)])` | 上下文超限时清理旧工具结果 | 08 章 3.8 |
| `agent/middleware/custom.py:271` `CUSTOM_MIDDLEWARE` | TaskGate / ContextInjection / Timing | **最后**，能看到前面所有中间件的结果 | 08 章 5 |

自定义中间件覆盖了课件 08 章第 5 节的两种写法：

| 中间件 | 钩子 | 演示的能力 | 对应课件 |
|---|---|---|---|
| `TaskGateMiddleware` | `after_model` | 检查本轮有没有写工具调用，有就拦下 | 08 章 5.3 Node-style hooks |
| `ContextInjectionMiddleware` | `before_model` + `wrap_model_call` | 把运行时上下文注入系统提示词 | 08 章 5.3 + 5.4（两种钩子对比） |
| `TimingMiddleware` | `after_agent` | 统计 token 与耗时 | 08 章 5.3 |

**关键认知 —— 执行顺序（课件 08 章 5.6）**：
> `before_*` 从前到后执行，`after_*` 从后往前执行，`wrap_*` 是洋葱式包裹（最外层先进入）。

这条规则解释了为什么 `LIMITS` 必须放最前：它的限次检查是 `before_model` 型，
放最前 = 最先检查 = 能在模型跑飞之前拦住。放最后就晚了。

**可加的内置中间件**（课件 08 章 2.x 都讲过）：
- `PIIMiddleware(pii_type="email", strategy="mask")` —— 自动脱敏，防止把敏感信息发给模型
- `SummarizationMiddleware(model=..., trigger=..., keep=...)` —— 上下文满了自动摘要
- `HumanInTheLoopMiddleware(interrupt_on={...})` —— 危险工具调用前人工审批（和「无人值守」定位冲突，但做本地工具很有用）

---

## 主线 4：上下文与记忆（课件 09 章）

**读源码**：`agent/context.py` → `agent/graph.py:290` `_build_checkpointer()` → `runtime/runner.py:148`

| 位置 | 看什么 | 对应课件 |
|---|---|---|
| `context.py` `AgentContext` | 每次调用的**静态**运行时配置 | 09 章 4「Static runtime Context」 |
| `agent/graph.py:290` | 记忆后端二选一：`sqlite`（`data/memory.sqlite`，进程级持久）或 `memory`（重启即失） | 09 章 2.1 / 2.2 / 2.3 |
| `runtime/runner.py:148` | thread_id 决定记忆隔离 | 09 章 2.1.4「工作原理」 |
| `agent/middleware/custom.py:48` `_ctx()` | `runtime.context` **可能是 None**，所有钩子都要兜底 | 09 章 4.1 |

**关键认知 —— 三个层次别混淆（课件 09 章 1.2.2）**：

| 层次 | 是什么 | 本项目的载体 |
|---|---|---|
| 动态运行时上下文 | state，随对话变化 | `AgentState`（messages / jump_to） |
| 动态跨会话上下文 | store，跨 thread 共享 | `InMemoryStore`（当前几乎没用） |
| **静态运行时上下文** | context，每次调用传入、不持久 | `AgentContext` |

**课件 09 章 4.1 讲了三种在钩子里读 context 的方式**，项目实现了两种：
- ✅ Node-style hooks（`before_model` / `after_model`）—— `TaskGateMiddleware` 用了
- ✅ Wrap-style hooks（`wrap_model_call` / `wrap_tool_call`）—— `ContextInjectionMiddleware` 用了
- ❌ `dynamic_prompt` —— **项目没实现**，这是动态改写系统提示词的官方方式

**记忆治理（课件 09 章 2.4）**：消息裁剪 / 消息删除 / 摘要。
项目做的是**工具结果清理**（`ContextEditingMiddleware` + `ClearToolUsesEdit`，属于 08 章 3.8），
而不是消息裁剪 —— 这条路还没走。

---

## 主线 5：运行时外壳（课件未覆盖，自研部分）

**读源码**：`runtime/runner.py` → `runtime/scheduler.py` → `runtime/server.py` → `runtime/tasks.py`

| 位置 | 看什么 |
|---|---|
| `runtime/runner.py:58` `get_agent()` | `_AGENT` 单例，保留 checkpointer 里的记忆。调度器和 Python 调用方都走 `run_task()`，保证行为一致 |
| `runtime/scheduler.py:141` | `CronTrigger.from_crontab` —— 5 段 cron，按 `SCHEDULER_TIMEZONE` 解释。**LangGraph 1.x 本地没有内置 cron**，`langgraph_sdk` 也没导出，那是云端 Platform 的能力 |
| `runtime/server.py:123` `/health` | 探活 + 调度器开关状态。**HTTP 层现在只管定时任务与状态查询，不提供触发 Agent 的入口** —— 触发走定时任务 / Python API / 桌面壳 |
| `runtime/tasks.py:109` `TASKS` | 改这个文件就改 Agent 每天自动做什么。注意它是**内存字典，不落盘** |

---

## 主线 6：LangGraph 底层（`courseware/langGraph/`，6 篇）

**读源码**：`agent/graph.py`（checkpointer / store 注入）→ `agent/middleware/custom.py`（Runtime）→ `runtime/events.py`（流式元数据处理）

LangChain 课件讲的是「怎么用 `create_agent`」，这套课件讲的是「`create_agent` 底下跑的是什么」。
本项目每个 Agent 都是一张 **LangGraph 状态图**，只是图是 `create_agent` 替你画的。

> 概念细节（State/Reducer、条件边、checkpointer、节点容错）见 `KNOWLEDGE.md` 第八章「LangGraph 图运行时」。

| 位置 | 看什么 | 对应课件 |
|---|---|---|
| `agent/graph.py:422` `build_agent()` | `create_agent()` 返回 `CompiledStateGraph`（底层 Pregel）。项目用的每一张图都是它画的 | 01 章 2.2「编译状态图」 |
| `agent/graph.py:290` `_build_checkpointer()` | 短期记忆：`SqliteSaver` / `InMemorySaver` 二选一 | 03 章 6.2「启用可恢复执行」/ 7.1 |
| `agent/graph.py:102` `InMemoryStore` | 长期记忆：跨 thread 的键值存储。**目前只挂了没用**（全项目没有 `store.put`/`get`/`search` 调用） | 03 章 7.2「长期记忆」 |
| `agent/middleware/custom.py:38` `Runtime` | `runtime.context` 就是 LangGraph 的运行时上下文 | 03 章 7.3「运行时上下文」 |
| `agent/middleware/custom.py:98` `after_model` | 钩子插在 `model` 与 `tools` **两个节点之间**，所以它能拦下写工具 | 02 章 4.2「分支结构」 |
| `runtime/events.py` `ThinkFilter` | 滤掉模型混在正文里的 `<think>` 思考块（MiniMax 这类推理模型） | 05 章 11.2 `stream`/`astream` |
| `runtime/events.py` `is_hint_node` | 流式元数据里的 `langgraph_node`，用来滤掉中间件插入的例行提示 | 05 章 11.2 |

**关键认知 —— 项目跑的那张图长这样**（实测 `agent.get_graph()`）：

```
__start__ → model → tools → model → … → __end__
              │        └──── 条件边：model 这轮有 tool_calls 就走 tools
              └───────────── 条件边：model 给了最终答案就走 __end__
```

顺序、分支、循环三件事 `create_agent` 全替你办了 —— 这正是课件 01 章「图的构建」和 02 章「控制流」讲的，
只是项目没手写 `StateGraph`。想看节点和边，直接 `agent.get_graph().draw_mermaid()`。

**学习顺序建议**（先补原理，再补没做过的）：

1. **03 章 6.1-6.3 + 7.1-7.3** —— 与项目重合度最高。读完能把 CLAUDE.md 陷阱 3/6/14 的成因讲清楚。
   *跳过附录 A（PostgreSQL）*，项目用 SQLite。
2. **01 章 2.1-2.4 + 3.2 reducer** —— 自己手写一张最小 `StateGraph`（放 scratch，别改项目），
   再用 `get_graph()` 把项目这张图画出来对比。**跳过 00 章**：项目已有 `langchain1.2` 环境，模型走 `.env` 的 `LLM_API_KEY`。
3. **04 章中断 + 05 章流式** —— 正好是 `DESKTOP_AGENT_PLAN.md` 第一步要的东西（事件流 + 取消）。

**这套课件里项目还没碰的**：02 章的 `RetryPolicy` / 节点超时 / 节点缓存（项目重试走中间件）、
04 章的 `interrupt()` 中断与 `langgraph dev` 部署、05 章的子图与设计模式。

**抄代码注意**：课件手写 `StateGraph`、项目用 `create_agent` 封装，概念相通但 API 不同。
别抄 `create_react_agent` —— 那会绕开项目现有的中间件栈。

---

## 课件里有、这个项目还没接入的

| 课件章 | 项目现状 | 怎么接 |
|---|---|---|
| 03 LangSmith | 没配追踪 | 加 `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` + `LANGSMITH_PROJECT`（**1.x 是 `LANGSMITH_` 前缀，不是老版本的 `LANGCHAIN_TRACING_V2`**）。看每次 invoke 的完整 trace，这是最快的调试利器 |
| 06 结构化输出 | 只有手写的 `save_json` | 用 `model.with_structured_output(PydanticModel)` 让模型直接返回结构化对象，比让模型「输出 JSON 字符串再解析」可靠得多。四种模式见课件 06 章 2.1-2.4（Pydantic / TypedDict / JSON Schema / dataclass） |
| 07 章 7.3 `ToolStrategy` | 没用 `response_format` | 让 Agent 直接返回结构化对象。注意 1.x **必须显式** `ToolStrategy(Schema)`，直接传 `Schema` 会报错 |
| 09 章 3 长期记忆 | store 只挂了 `InMemoryStore` 没实际用 | 用 `store.put(namespace, key, value)` / `store.search()` 实现跨会话记忆 |
| 10 RAG | requirements 里被注释掉了 | 需要向量库 + 切分，最后再碰 |

LangGraph 专项课件（`courseware/langGraph/`）里还没接入的：

| 课件小节 | 项目现状 | 怎么接 |
|---|---|---|
| 02 章 5.2 重试 / 5.3 超时 / 5.5 节点缓存 | 失败重试走 `ModelRetryMiddleware` / `ToolRetryMiddleware`（中间件层），没用图节点级的 `RetryPolicy` | 节点级重试更细（可按节点配），但中间件已够用；想换就改 `graph.py` 的 `RETRY` |
| 04 章 8.1 动态中断 / 8.2 静态断点 | 没有任何 `interrupt()` 调用 | `DESKTOP_AGENT_PLAN.md` 第一步的「取消」要么用 `interrupt()` 做暂停点，要么走独立的取消通道 |
| 04 章 9.1 本地部署（`langgraph dev` / AgentChatUI） | 用 FastAPI 自建服务（`runtime/server.py`） | `langgraph dev` 需要模块导出图对象，本项目导出的是 FastAPI `app`，套不上 |
| 05 章 12 子图 | 单张图，没有子图 | 现在没必要；等任务表长出多步骤工作流再拆 |
| 05 章 13 运行图设计模式（路由 / 编排者-工人 / 评估者-优化者） | 单一 ReAct 循环 | 同一个 `create_agent` 循环就是「路由」模式；编排者-工人需要手写 `StateGraph` |

---

## 常见卡点速查

完整版在 `CLAUDE.md` 的「关键 API 陷阱」，这里只列最容易撞的：

1. `context` 是 `invoke` 的独立参数，**不在 config 里**。放错位置 `runtime.context` 就是 `None`。
2. `runtime.context` 可能为 `None`，钩子里必须兜底。
3. `ModelRequest` 不可变，改字段用 `override()`；`override` 是**替换**语义，追加内容要自己拼。
4. `create_agent` 收中间件**实例**，不是类。
5. 假模型必须实现 `bind_tools`，否则 `NotImplementedError`。
6. `SqliteSaver.from_conn_string()` 是上下文管理器，直接传给 `create_agent` 会报 `Invalid checkpointer`。
7. `hook_config` 只用于 `can_jump_to`，不能写成无参装饰器。

---

