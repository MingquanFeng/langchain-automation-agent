# 学习地图

这个项目只有 1100 行，但它把 **langchain 1.x agent 开发的每个知识点都落到了可运行的代码上**。
它的价值不在业务功能，而在于「每段代码都标了对应课件章节，每个知识点都有对应的实现」。

## 怎么用这张地图

源码注释里的「对应课件 XX 章 X.X 节」就是索引。本文档把两样东西串起来：

```
课件小节  →  该读哪几个文件（按顺序）
```

每条主线先给源码位置，再给这个知识点的关键认知和容易踩的坑 —— 看代码时对照着看，
比通读课件更快定位到自己要用的那部分。

课件原文就在本仓库的 `courseware/` 目录（尚硅谷 LangChain 1.x 的 10 章 PDF 原件），
文件名的数字即章号 —— `courseware/尚硅谷-05-Tools.pdf` 就是课件 05 章。
章节与源码的对应索引见 `courseware/README.md`。

> **不想通读 10 章课件、只想拿结论**：直接看 `KNOWLEDGE.md` —— 那是把课件里与本项目相关的部分
> 压缩后的知识本体（API 签名 + 项目用法 + 坑），按「一次调用发生了什么」组织，不是章节摘抄。

---

## ⚠️ 第一件事：分清 1.x 和 0.x

网上关于 LangChain Agent 的教程**绝大多数是 0.x 的**，照着抄会写出跑不通的代码。先记住这张表：

| 知识 | 1.x 写法（本项目在用） | 0.x 老写法（已过时，别抄） |
|---|---|---|
| 创建 Agent | `create_agent(...)` | `AgentExecutor` / `initialize_agent` |
| 预置 Agent | `create_agent` 自带 | `create_react_agent`、`create_tool_calling_agent`、`create_structured_chat_agent` |
| 定制 Agent | **中间件体系** | `llm_agent` + `agent_kwargs` + `allowed_tools` |
| 记忆 | `state` + `store` + `checkpointer` | `ConversationBufferMemory` 等 `xxxMemory` 类 |
| 结构化输出 | `with_structured_output()` | `JsonOutputParser` + 提示词拼格式（课件明确「不推荐」） |
| Agent 的结构化响应 | `response_format=ToolStrategy(Schema)` | 直接传 `response_format=Schema`（**1.x 已不支持**） |

0.x 的老 API 没被删除，而是挪进了 `langchain-classic` 包 —— 所以你搜到的代码**能 import 但不该用**。

---

## 主线 1：模型怎么变成 Agent（课件 02 / 04 / 07 章）

**读源码**：`agent/graph.py` → `agent/config.py`

| 位置 | 看什么 | 对应课件 |
|---|---|---|
| `graph.py:65` `build_agent()` | 唯一创建 Agent 的地方。`create_agent(model, tools, system_prompt, middleware, checkpointer, store, context_schema, name)` | 07 章 1.4.2「全新的调用」 |
| `graph.py:93` `SYSTEM_PROMPT` | 系统提示词就是普通字符串，Agent 的一切行为从它开始 | 07 章 6 |
| `config.py:32` `ModelConfig.build()` | `init_chat_model(model=..., model_provider=...)` | 02 章 2.2「兼容用法」 |
| `runtime/runner.py:46` | `HumanMessage` 构造输入 | 04 章 1.5.2 |

**关键认知**：1.x 里「Agent」不再是一个需要 `initialize_agent` 的类，而是一次 `create_agent()` 调用返回的**编译好的图**。
模型、工具、中间件、记忆全是**注入参数** —— 所以换模型/换工具不用改调用方，这就是 `build_agent()` 存在的意义。

**延伸**：课件 07 章 8.2 讲了 7 种 `stream_mode`（values / updates / messages / tasks / debug / checkpoints / custom）。
项目只用了 `values`（`server.py:113`）和 `messages`（`repl.py:458`）。

---

## 主线 2：工具（课件 05 章）

**读源码**：`agent/tools/files.py` → `agent/tools/notify.py` → `agent/tools/__init__.py`

| 位置 | 看什么 | 对应课件 |
|---|---|---|
| `files.py` `@tool(parse_docstring=True)` | 装饰器把函数变成工具。**docstring 就是给模型看的工具说明**；开了 `parse_docstring` 后 `Args:` 段会进 schema 的参数说明 | 05 章 2.2「工具描述的各部分」/ 3.1 |
| `files.py` `_resolve()` / `_resolve_safe()` | 沙箱。越界时工具层返回中文说明（不抛异常，避免被重试 3 次） | 05 章 1.3「工具调用的整体流程」 |
| `files.py` `save_json` | 参数标了 `dict | list`，langchain 自动转成 JSON schema | 05 章 2.2.4「参数类型」 |
| `notify.py:23` `_deliver()` | 通知有 webhook 和终端两条路，未配置时降级 | — |

**关键认知**：模型**不执行你的 Python 函数**。它只输出一段结构化的 `tool_calls`（名字 + JSON 参数 + id），
由框架代为执行，再把返回值包成 `ToolMessage` 回灌进对话 —— 这个来回就是 agent 循环。

**两条工具设计规范**：
- 工具数量 **2-5 个最佳**（**07 章** P21：「只给 Agent 需要的工具，工具太多会混淆。一般 2-5 个工具最佳」）—— 本项目有 8 个（6 个文件 + 2 个通知），已经偏多，模型选错工具的概率会上升。
- docstring 要写 **Google 风格**（参数逐个说明，**05 章** 3.1）—— 项目现在 8 个工具全部开了 `parse_docstring=True` 且写成 Google 风格，参数说明会进 schema 的 `properties.<arg>.description`，模型不用靠参数名猜。

---

## 主线 3：中间件（课件 08 章）—— 本项目精华

**读源码**：`agent/graph.py:32-47`（内置栈）→ `agent/middleware/custom.py`（三个自定义）

内置中间件栈：

| 位置 | 中间件 | 作用 | 对应课件 |
|---|---|---|---|
| `graph.py:43` `LIMITS` | `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` | **最前**，先挡住跑飞，防止无人值守时烧钱 | 08 章 3.1 / 3.2 |
| `graph.py:37` `RETRY` | `ModelRetryMiddleware` / `ToolRetryMiddleware` | 模型/工具失败时重试 | 08 章 3.6 / 3.5 |
| `graph.py:91` `TODO` | `TodoListMiddleware` | 注入 `write_todos` 工具，让模型拆解多步任务 | 08 章 2.4 |
| `graph.py:32` `CONTEXT_EDITING` | `ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=120_000, keep=3)])` | 上下文超限时清理旧工具结果 | 08 章 3.8 |
| `custom.py:116` `CUSTOM_MIDDLEWARE` | TaskGate / ContextInjection / Timing | **最后**，能看到前面所有中间件的结果 | 08 章 5 |

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

**读源码**：`agent/context.py` → `agent/graph.py:49` `_build_checkpointer()` → `runtime/runner.py:43`

| 位置 | 看什么 | 对应课件 |
|---|---|---|
| `context.py` `AgentContext` | 每次调用的**静态**运行时配置 | 09 章 4「Static runtime Context」 |
| `graph.py:49` | 记忆后端二选一：`sqlite`（`data/memory.sqlite`，进程级持久）或 `memory`（重启即失） | 09 章 2.1 / 2.2 / 2.3 |
| `runner.py:43` | thread_id 决定记忆隔离 | 09 章 2.1.4「工作原理」 |
| `custom.py:21` `_ctx()` | `runtime.context` **可能是 None**，所有钩子都要兜底 | 09 章 4.1 |

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
| `runner.py:18` `get_agent()` | `_AGENT` 单例，保留 checkpointer 里的记忆。调度器和 HTTP 都走 `run_task()`，保证行为一致 |
| `scheduler.py:52` | `CronTrigger.from_crontab` —— 5 段 cron，按 `SCHEDULER_TIMEZONE` 解释。**LangGraph 1.x 本地没有内置 cron**，`langgraph_sdk` 也没导出，那是云端 Platform 的能力 |
| `server.py:113` `/run/stream` | SSE 流式。`stream_mode="values"` 每步给全量状态 |
| `tasks.py:22` `TASKS` | 改这个文件就改 Agent 每天自动做什么。注意它是**内存字典，不落盘** |

---

## 课件里有、这个项目还没接入的

| 课件章 | 项目现状 | 怎么接 |
|---|---|---|
| 03 LangSmith | 没配追踪 | 加 `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` + `LANGSMITH_PROJECT`（**1.x 是 `LANGSMITH_` 前缀，不是老版本的 `LANGCHAIN_TRACING_V2`**）。看每次 invoke 的完整 trace，这是最快的调试利器 |
| 06 结构化输出 | 只有手写的 `save_json` | 用 `model.with_structured_output(PydanticModel)` 让模型直接返回结构化对象，比让模型「输出 JSON 字符串再解析」可靠得多。四种模式见课件 06 章 2.1-2.4（Pydantic / TypedDict / JSON Schema / dataclass） |
| 07 章 7.3 `ToolStrategy` | 没用 `response_format` | 让 Agent 直接返回结构化对象。注意 1.x **必须显式** `ToolStrategy(Schema)`，直接传 `Schema` 会报错 |
| 09 章 3 长期记忆 | store 只挂了 `InMemoryStore` 没实际用 | 用 `store.put(namespace, key, value)` / `store.search()` 实现跨会话记忆 |
| 10 RAG | requirements 里被注释掉了 | 需要向量库 + 切分，最后再碰 |

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

