# 知识点总结 —— 本项目用到的 LangChain 1.x 知识

> **这份文档只讲项目代码里真实用到的知识点。**
> 课件原文在 `courseware/`，两套讲义：LangChain 1.x 的 10 章 PDF（`courseware/langChain/`）+
> LangGraph 专项 6 篇 Markdown（`courseware/langGraph/`）；本文把它们压缩成「一次调用到底发生了什么」这一条主线。
>
> 与 [`LEARNING.md`](LEARNING.md) 的分工：
> - `LEARNING.md` 是**索引** —— 告诉你每个知识点在哪一章、对应哪些源码文件。
> - 本文是**知识本体** —— API 是什么、怎么写、坑在哪。可以当速查手册用。
>
> 标注说明：
> - 📘 = 课件原文出处（章节号 / 小节号）。`LG XX 章` 指 LangGraph 专项课件
>   （`courseware/langGraph/`，章节号 1-13），不带前缀的指 LangChain PDF 课件（01-10 章）
> - 🧩 = 项目自己的做法或踩出来的坑（课件没讲或讲得不准）

---

## 目录

| 章 | 内容 | 主要课件来源 |
|---|---|---|
| [一](#一一次调用的完整数据流) | 一次调用的完整数据流 | — |
| [二](#二包结构) | 包结构 | 01 章 |
| [三](#三模型与消息02--04-章) | 模型与消息 | 02 / 04 章 |
| [四](#四工具05-章) | 工具 | 05 章 |
| [五](#五agent07-章) | Agent | 07 章 |
| [六](#六中间件08-章-本项目精华) | 中间件（本项目精华） | 08 章 |
| [七](#七上下文与记忆09-章) | 上下文与记忆 | 09 章 |
| [八](#八langgraph-图运行时coursewarelanggraph) | LangGraph 图运行时 | LG 01-05 章 |
| [九](#九项目自研的运行时外壳课件未覆盖) | 运行时外壳（课件未覆盖） | — |
| [十](#十课件里有项目还没用的) | 课件里有、项目还没用的 | 03 / 06 / 10 章 + LG |
| [十一](#十一踩坑速查按症状索引) | 踩坑速查（按症状索引） | — |

---

## 一、一次调用的完整数据流

理解了这个流程图，下面的知识点就都有挂靠点了。

```
①  用户指令  "整理一下 data 目录"
        │
        ▼
②  runtime/runner.py  run_task()
    ├─ 构造 AgentContext（任务名 / 触发源 / 写权限）      ← 静态运行时上下文
    ├─ 构造 config = {"configurable": {"thread_id": ...}} ← 会话隔离键
    └─ agent.invoke({"messages":[HumanMessage(...)]}, config, context=ctx)
                                    ↑                    ↑
                        context 是独立参数，不在 config 里  ← 最容易踩的坑
        │
        ▼
③  LangGraph agent 图开始跑
    │
    │   ┌─────────────────────────────────────────────┐
    │   │  中间件栈（洋葱式，见第六章）                 │
    │   │  事件桥接 → LIMITS → RETRY → TODO → 清理 → PII │
    │   │        → CUSTOM_MIDDLEWARE                  │
    │   └─────────────────────────────────────────────┘
        │
        ▼
④  模型节点  system_prompt + 运行时上下文 + 工具清单 → 模型
        │                                     ↑ agent/config.py  init_chat_model
        │
        ├──▶ 模型返回 AIMessage（**没有 tool_calls**）──▶ 结束，取最后一条消息当答案
        │
        └──▶ 模型返回 AIMessage（**带 tool_calls**）
                │
                ▼
⑤  工具节点  框架执行 tool_calls，把返回值包成 ToolMessage
                │
                │   🧩 只有写操作会被 TaskGateMiddleware 拦下，
                │      拦下后必须补 ToolMessage（见 4.6）
                ▼
⑥  ToolMessage 回灌 → 回到 ④，直到模型不再要工具
        │
        ▼
⑦  记忆落盘  checkpointer 按 thread_id 存下完整消息历史
```

**一句话记住**：模型**不执行你的 Python 函数**，它只输出一段结构化的 `tool_calls`（名字 + JSON 参数 + id），
框架代为执行，再把返回值包成 `ToolMessage` 回灌 —— 这个来回就是 agent 循环。（📘 05 章 1.3）

---

## 二、包结构

📘 01 章 2.3 —— 1.x 把包结构拆清了：装哪个用哪个，不必把整个生态都装上。

| 包 | 定位 | 本项目用到的 |
|---|---|---|
| `langchain-core` | 核心 API（Runnable、BaseMessage 等） | `HumanMessage`、`@tool` |
| `langchain-community` | 第三方集成，按需安装 | `langchain-deepseek`（provider=deepseek 时自动装） |
| `langgraph` | 图编排层：状态、checkpointer、store、中间件运行时 | `Runtime`、`InMemorySaver`、`SqliteSaver`、`InMemoryStore` |

主 `langchain` 包保持轻量，Agent 的能力都从 `langchain.agents` 进。

---

## 三、模型与消息（02 / 04 章）

### 3.1 `init_chat_model`：1.x 统一的模型创建入口

**是什么**：按模型标识自动挑选底层驱动类（`ChatOpenAI` / `ChatDeepSeek` …），不必记每个供应商各自的类名和参数写法。

📘 02 章 2.2：

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "provider:model_name",   # 或拆成 model= + model_provider=
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=1000,
    **kwargs
)
```

**项目里怎么用**（`agent/config.py:200-227`，`ModelConfig.build()`）：

```python
from langchain.chat_models import init_chat_model

kwargs: dict = {
    "model": self.model,              # LLM_MODEL，默认 deepseek-chat
    "model_provider": self.provider,  # LLM_PROVIDER，默认 deepseek
    "temperature": self.temperature,  # LLM_TEMPERATURE，默认 0.3
}
if self.api_key:
    kwargs["api_key"] = self.api_key
if self.base_url:
    kwargs["base_url"] = self.base_url

return init_chat_model(**kwargs)
```

**换模型只改 `.env` 三行**（`LLM_PROVIDER` / `LLM_MODEL` / `LLM_BASE_URL`），代码不用动。
五个前提与已实测的模型组合见 `CLAUDE.md` 的「换模型须知」。

**坑**
- 🧩 `api_key` / `base_url` 是「配了才传」。不传 `base_url` 时由供应商集成决定默认地址。
- 📘 02 章：**并非所有模型名都支持自动推断**（如 `qwen-plus` 会报错）。所以项目把 provider 单独做成必填配置项，不靠推断。
- 📘 02 章：阿里 `dashscope` 尚未纳入统一注册体系，只能用 `model_provider="openai"` + OpenAI 兼容地址。
- 🧩 `config.py` 自注：**部分模型不接受 `temperature`**。
  实测 `langchain_openai` 有兜底（`validate_temperature` 校验器：gpt-5 把非 1 的值丢掉，
  o1 没传时自动设 1），但非 OpenAI 兼容的原生集成不一定有这层保护。
- 📘 02 章的 temperature 分档：`0.0-0.3` 用于要一致性/准确性的任务（数据提取、分类、代码生成）；`0.5-0.7` 平衡；`0.8-1.5` 创造性。项目取 0.3。

### 3.2 `.invoke()` 与 `.stream()`

| | `.invoke()` | `.stream()` |
|---|---|---|
| 返回 | 一次性返回完整 `AIMessage` | 迭代器，逐块返回 |
| 项目用在 | `runtime/runner.py` 同步调用；`runtime/consumer.py` 通过 `agent.stream()` 消费流式输出 | WebSocket 工作台使用流式路径 |

📘 02 章 31 页签名：

```python
def invoke(self, input, config=None, *, stop=None, **kwargs) -> AIMessage: ...
```

**坑**
- 📘 02 章：流式输出**依赖模型供应商是否支持**。
- 🧩 `chunk.content` 可能是字符串也可能是内容块列表，项目两处都先 `isinstance(..., str)` 判断再取
  （过滤逻辑现在在 `runtime/events.py` 的 `ThinkFilter`）。

### 3.3 消息四角色

📘 04 章第 4 页角色表：

| 角色 | 对象 | 用途 |
|---|---|---|
| System | `SystemMessage(...)` | 设定 AI 的行为、角色、规则 |
| User | `HumanMessage(...)` | 用户输入 |
| Assistant | `AIMessage(...)` | AI 的回复（**带 `tool_calls` 时是工具请求**） |
| Tool | `ToolMessage(...)` | 工具执行结果 |

**项目里怎么用**
- `runtime/runner.py:28` 从 `langchain_core.messages` 导入 `HumanMessage`，在 `runner.py:166` 构造 `HumanMessage(content=instruction)`。
- `agent/middleware/custom.py:34` 从 `langchain.messages` 导入 `AIMessage, SystemMessage, ToolMessage`。
  （1.x 里这两条导入路径指向同一套消息类型，项目两处都在用。）

**坑**
- 📘 04 章：`content` 是唯一必填字段，**字段名可以省略** —— `SystemMessage("你是助手")` 等价于 `SystemMessage(content="...")`。
- 📘 04 章：`name` / `id` 元数据字段**是否生效取决于模型供应商** —— DeepSeek 官方声明支持 `name`，实测模型识别不了。别把名字字段当可靠能力用。

### 3.4 `.content` / `.text` / `.content_blocks`

| 属性 | 含义 | 项目用在哪 |
|---|---|---|
| `.content` | 内容字段，**弱类型**：字符串或内容块列表 | `runner.py:178` 取最终答案；流式增量取法见 `runtime/events.py` |
| `.text` | 纯文本形态的内容 | `custom.py:206` 取 `SystemMessage` 文本 |
| `.content_blocks` | 1.x 新增的**跨供应商标准化多模态结构**（推理 / 文本 / 工具调用统一对象） | **项目未使用** |

**坑**
- 📘 04 章：`content_blocks` 是**懒加载**的，调用时才解析；需要拿「思维链」或「引用」信息时**优先检查 `content_blocks` 而不是 `content`** —— 不同供应商的思维链位置不同（DeepSeek 在 `additional_kwargs.reasoning_content`）。
- 🧩 课件对 `.text` 与 `.content` 的差别**没有明确说明**（02 章流式示例用 `chunk.text`，04 章流式示例用 `chunk.content`，两处不一致）。项目只在取系统提示词纯文本这一处用 `.text`，其余一律 `.content`。

### 3.5 提示词模板（项目未使用）

📘 04 章 28-31 页的 `ChatPromptTemplate`：

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的AI机器人，你的名字是{name}。"),
    ("human", "你好，最近怎么样？"),
    ("ai", "我很好，谢谢！"),
    ("human", "{user_input}"),
])
prompt = chat_template.invoke({"name": "小明", "user_input": "你叫什么名字？"})
```

**项目里怎么用**：**完全没用**。全项目零 `ChatPromptTemplate` / `PromptTemplate` 引用。
系统的提示词是纯字符串常量 `agent/graph.py:653`，需要随运行时变化的内容由中间件在调用前改写请求（见 6.8）。

**为什么这样可行**：📘 04 章说纯字符串拼接的缺点是「可读性差、不易维护、无变量校验」——
但项目的提示词里**根本没有变量**（任务名、触发方式、写权限是运行时动态追加的，不是模板占位符）。

**坑**：📘 04 章 —— 在 `XxxMessage` 对象里不能有占位符，`SystemMessage(content="{word}")` 喂给 `from_messages` 时 `{word}` 不会被替换。

### 3.6 模型是无状态的

📘 04 章 16 页的关键规则：**每次调用必须传递完整的对话历史**。

**项目里怎么用**：项目**不手写消息列表**，历史由 checkpointer 承担。`runner.py:165` 每次只传一条新消息：

```python
result = agent.invoke(
    {"messages": [HumanMessage(content=instruction)]}, config, context=ctx
)
```

checkpointer 自动完成「读历史 → 追加 → 调模型 → 存历史」四步（📘 09 章 12-13 页）。

---

## 四、工具（05 章）

### 4.1 工具调用的四步流程

📘 05 章 1.3：

```
1. 模型绑定工具：model.bind_tools([...])
2. 模型生成工具调用请求：返回带 tool_calls 的 AIMessage
3. 开发者执行工具：tool.invoke()
4. 把结果包成 ToolMessage 回灌，模型据此生成最终回答
```

**项目里怎么用**：项目不手写这个循环，交给 `create_agent`（`agent/graph.py:516`）。
只在**闸门拦截**这一处手工构造 ToolMessage（`custom.py:134`）。

### 4.2 `@tool` 装饰器

📘 05 章第 2 页：

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息
    参数:
        city: 城市名称，如"北京"、"上海"
    返回:
        天气信息字符串
    """
    return city + "晴天，温度 15°C"

result = get_weather.invoke({"city": "北京"})
```

装饰之后：函数名 → 工具名；docstring → 工具描述；类型注解 → 参数 JSON Schema；函数体多出 `.invoke()`。

**项目里怎么用**：`agent/tools/files.py` 6 个 + `agent/tools/notify.py` 2 个，统一登记到 `agent/tools/__init__.py` 的 `ALL_TOOLS`。

**坑**
- 📘 05 章：**没有 docstring 会直接报错** —— `ValueError: Function must have a docstring if description not provided.`
- 📘 05 章：`config` / `runtime` 是 LangChain 内部保留参数名，别用作工具参数名。
- 🧩 `agent/tools/__init__.py:12-14`：「用 `@tool` 装饰过的函数**如果不登记进 `ALL_TOOLS`，模型根本看不到它** —— 工具写好了不登记，等于没写。」

### 4.3 docstring 就是给模型看的说明书

📘 05 章第 9 页，查看真实解析结果：

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
rprint(convert_to_openai_tool(get_weather))
```

输出结构：

```python
{
 'type': 'function',
 'function': {
   'name': 'get_weather',
   'description': '天气查询工具',
   'parameters': {
     'properties': {'city': {'description': '城市名称', 'type': 'string'}},
     'required': ['city'],
     'type': 'object'
   }
 }
}
```

**🧩 项目自己的约定**（`agent/tools/files.py` 模块 docstring）：

> 被 `@tool` 装饰的函数，它的 docstring 会被当作「工具描述」发给模型，相当于提示词的一部分。
> 所以这些 docstring 只写模型需要知道的功能性说明，深度解释放在周围的 `#` 注释里 —— **改 docstring 等于改模型行为**。

**坑**
- 📘 05 章：**要用 Google 风格的参数说明，必须显式开 `parse_docstring=True`**。开了之后 `description` 只剩首行，参数说明落到 `properties.<arg>.description`。
- 🧩 **项目的 8 个工具全部开了 `parse_docstring=True`**（`files.py` 6 个 + `notify.py` 2 个），
  docstring 一律写成 Google 风格（首行概括 + `Args:` 逐参数说明）。
  效果实测：`read_file` 的 `offset` 参数在 schema 里带上了
  `'description': '起始行号，从 0 开始。用来接着上次读到的位置继续读。'` ——
  模型不用再靠参数名猜。不开的话整段 docstring 会原样进 `description`，参数仍是没说明的白板。
- 📘 05 章：开了 `parse_docstring=True` 但 docstring 不合法 → `ValueError: Found invalid Google-Style docstring.`；
  docstring 里写了签名里没有的参数 → `ValueError: Arg city in docstring not found in function signature.`。
- 📘 05 章：docstring 写了参数、但该参数**没有类型注解** → 报错。

### 4.4 参数类型 → 自动生成 JSON Schema

📘 05 章 13-14 页实测规律：

| 函数签名 | 生成的 schema |
|---|---|
| `city: str` | `'required': ['city']` |
| `city: str = "北京"` | `'city': {'default': '北京', ...}`，且 `required` 整个字段消失 |
| `dt: str, city: str = "北京"` | `'required': ['dt']` |

**项目里怎么用**：`list_files(pattern: str = "*")` → `pattern` 带默认值、不进 required；
`write_file(path: str, content: str)` → 两个都在 required。

**坑**：📘 05 章 —— 删掉类型注解，schema 里该参数就变成 `'city': {}`，模型完全不知道该传什么。

**🧩 宽泛类型要慎用**：`save_json` 原来标注 `data: dict`，生成的 schema 是
`'data': {'additionalProperties': true, 'type': 'object'}` —— 模型只知道是个对象，不知道该放什么键。
更糟的是**只收对象**：模型想存一个 JSON 数组（极常见的数据形状）会触发 Pydantic 校验失败，
被 `ToolRetryMiddleware` 重试两次同样的参数后彻底失败。
现在改成 `data: dict | list`，schema 变成 `anyOf[object, array]`，两种形状都能存。

想进一步收紧（比如枚举、范围限制）有两条标准做法（项目都没用）：
- Pydantic `args_schema`：`class WeatherInput(BaseModel): unit: Literal["celsius","fahrenheit"] = Field(default="celsius", description="气温单位")` → `@tool(args_schema=WeatherInput)`
- JSON Schema 字典：`@tool(args_schema={...})` —— 📘 05 章提醒，传的只能是 `parameters` 那一段，**不是整个 `{"type":"function",...}` 外壳**。

### 4.5 工具描述的三个组成部分

`bind_tools` 最终调 `convert_to_openai_tool` 生成三段式，模型据此决定「调不调、调哪个、参数怎么填」：

```
name        工具名
description 工具描述（来自 docstring）
args schema 参数 JSON Schema
```

**🧩 工具名可以自定义**（`@tool(name_or_callable="getWeather")` 或 `@tool("getWeather")`），
但 📘 05 章说「开发中习惯使用函数名作为工具名称，不推荐自定义工具名称」。项目全部用函数名。

**📘 强制使用工具 `tool_choice`**（项目未用）：`none` / `auto`（默认）/ `required` / 指定工具名。
项目对「必须汇报」的约束写在系统提示词里，不是靠 `tool_choice` 强逼。

### 4.6 ToolMessage，以及「拦截也必须补」

**协议要求**：每个 `tool_call` 都必须有对应的 `ToolMessage`，靠 `tool_call_id` 配对。

📘 05 章第 13 页：

```python
ToolMessage(
    content="<工具输出>",
    name="get_weather",
    tool_call_id="call_00_nUD2NC9QRN5Cg1GaoIkBJQ4s"  # 一定要和 AI 消息中的调用 ID 匹配
)
```

**🧩 项目踩过的大坑**（`custom.py:80-86`，`tests/test_e2e.py:186-190` 守着这条）：

```python
ToolMessage(
    content=(f"当前任务为只读模式，已阻止写操作 {names}。"
             "不要再调用这个工具，改用只读方式把结论写在回复里。"),
    name=tc["name"],
    tool_call_id=tc["id"],   # 必须与原始调用对上
    status="error",          # 告诉下游这是一次失败而不是成功结果
)
```

**两个讲究**：
1. **少了 ToolMessage，下一轮请求被 OpenAI / MiniMax 这类兼容接口直接 400**
   （`tool call result does not follow tool call`）—— 消息历史里留着没有结果的 `tool_call`。
2. **只补 ToolMessage、不补收尾消息**。agent 图的路由看的是最后一条 AIMessage 的 `tool_calls`：
   都已有 ToolMessage 回应 → 判定为「人工注入的工具消息」→ 跳回模型节点重新回答 →
   agent 改用文字给结论。顺手补一条 AIMessage 会改变路由，模型就不再重新回答。

### 4.7 沙箱与上下文保护（🧩 课件未讲，项目自加）

```python
SANDBOX = DATA_DIR.resolve()          # agent/tools/files.py

def _resolve(path: str) -> Path:
    target = (SANDBOX / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not target.is_relative_to(SANDBOX):
        raise ValueError(f"路径超出允许范围（仅允许 {SANDBOX}）：{path}")
    return target
```

**坑**：🧩 直接判断字符串里有没有 `".."` **是不够的** —— 绝对路径、软链接都能绕出去。
先 `resolve()` 规范化、再判包含关系才可靠。

**🧩 越界为什么不抛异常**：工具层走 `_resolve_safe()`，越界时**返回一句给模型看的中文说明**，
而不是让 `_resolve()` 的 `ValueError` 冒出去。原因是一个实测出来的连锁反应：

> `ToolRetryMiddleware` 的 `retry_on` 默认是 `(Exception,)` —— 所有异常都重试。
> 越界抛异常 → 同一个错参数被重试 **3 次**、空耗约 **3 秒** →
> 模型最后收到 `Tool 'write_file' failed after 3 attempts with ValueError: ... Please try again.`
> —— 英文、没告诉它该怎么改，它很可能再拿同一个错路径试一轮。

返回说明文字后，模型一次就拿到「请改用数据目录内的相对路径，例如 'out/report.md'」。
这跟 `_read_text` 对二进制、对超大文件的处理是同一套哲学：**给模型一句能照做的说明，而不是把整轮推理打断**。
（`_resolve()` 仍然抛异常 —— 那是留给 Python 调用方的。）

另外几层保护：
- **单次读取 200KB 上限 + 默认 200 行上限** —— 否则 agent 一句「看下这个文件」就能把几十 MB 塞进上下文，直接撑爆模型上下文窗口。
- **按行分段读取**（`read_file(path, offset, limit)`）—— 超限提示写「请分段读取」，就得真有分段读取的手段，
  而且返回里要报出总行数和下一步该传什么 `offset`。
- **文件列表带修改时间**（`out/report.md  (21 bytes, 2026-11-02 16:15)`）——
  模型要回答「哪些文件很久没动」时光有文件名和大小是没有依据的，只能靠猜。
- **记忆库禁读**（`memory.sqlite` 及其 WAL/SHM）—— 二进制读了也没意义，还触发上面的坑；列表工具也把它滤掉，免得模型去读。
- **原子写** —— 先写 `xxx.tmp` 再 `os.replace` 覆盖，中断时不会留下半截文件被当成正常内容读进去。

### 4.8 课件的五条实践经验（📘 05 章 6 节）

| # | 守则 | 项目现状 |
|---|---|---|
| 1 | **清晰的描述** —— docstring 要写清功能与参数 | ✅ 达标：8 个工具全开 `parse_docstring=True`，docstring 一律 Google 风格，参数说明进了 schema 的 `properties.<arg>.description` |
| 2 | **功能单一** —— 一个工具只做一件事 | ✅ 达标，8 个工具职责不重叠 |
| 3 | **如何处理工具失败？三层防护**：① 工具内部处理 → ② Agent 级重试（提示词）→ ③ 调用级重试 | ✅ 三层都有：① 工具内部返回说明文字而非抛异常；② `SYSTEM_PROMPT` 第 5 条；③ `ToolRetryMiddleware` |
| 4 | **返回字符串** —— 别返回 dict | ✅ 全部返回 `str`，`tests/test_e2e.py` 有参数化用例守着 |
| 5 | **选择同步 vs 异步** —— IO 密集型该用 async | ⚠️ `notify.py` 仍是同步调用，但已改成复用 `httpx.Client`（连接池复用，不再每次新建 TCP/TLS）。真正改 async 需要整条调用链异步化 —— `agent.invoke()` 是同步的，async 工具在同步路径下不会被执行 |

> ⚠️ **常见误引**：「工具数量 2-5 个最佳」这句话在 **07 章**（📘 07 章 P21「只给 Agent 需要的工具，工具太多会混淆。一般 2-5 个工具最佳」），**不在 05 章**。
> 按这条标准，本项目 8 个工具偏多，模型选错工具的概率会上升。

**🧩 项目自注的补充**（`graph.py:225-234`），课件没讲但很重要：
> 对发送消息、修改文件、写入数据库等**有副作用的工具，自动重试可能造成重复操作**。
> 第一次发送消息实际上成功，但网络超时导致客户端认为失败，重试后可能再发一次。

---

## 五、Agent（07 章）

### 5.1 `create_agent` 是「编译好的图」，不是类

📘 07 章 1.4.2：

> LangChain 在 1.0 版本后做了彻底重构：将所有 Agent 的创建方式统一为一个入口 `create_agent()`，
> 不再按场景分成多个分支函数。

📘 07 章 P4-5 的参数签名：

```python
agent = create_agent(
    model: str | BaseChatModel,             # 必需：聊天模型
    tools: List[BaseTool],                  # 必需：工具列表
    *,
    system_prompt: str = "",                # 系统提示词
    middleware: Sequence[AgentMiddleware] = (),
    interrupt_before: List[str] = None,     # 在某些工具前暂停（人机协作）
    interrupt_after: List[str] = None,
    debug: bool = False,
    name: str | None = None,
)
```

返回类型（📘 07 章 P4 实测）：

```python
agent = create_agent("deepseek-v4-flash")
print(type(agent))
# <class 'langgraph.graph.state.CompiledStateGraph'>
```

**🧩 项目实际用到 8 个参数**（`agent/graph.py:516-618`，全项目唯一的 `create_agent` 调用点）：

```python
return create_agent(
    model=model or MODEL.build(),                    # 508
    tools=tools if tools is not None else ALL_TOOLS,  # 521
    system_prompt=SYSTEM_PROMPT,                      # 528
    middleware=middleware,                            # 535
    checkpointer=(checkpointer if checkpointer is not None
                  else _build_checkpointer()),        # 547-551
    store=(store if store is not None
           else InMemoryStore()),                     # 565-569
    context_schema=AgentContext,                      # 590
    name="automation_agent",                          # 598
)
```

**关键认知**：「Agent」不是一个需要实例化的类，而是一次 `create_agent()` 调用返回的**编译好的图**。
模型、工具、中间件、记忆全是**注入参数** —— 所以换模型/换工具不用改调用方，这就是 `build_agent()` 存在的意义。

**坑**
- 🧩 `checkpointer=None` **不是「不要记忆」，是「用默认的」** —— 源码是
  `checkpointer if checkpointer is not None else _build_checkpointer()`，传 None 会建**真实的 sqlite checkpointer**。
  测试必须显式传 `InMemorySaver()`，否则测试会话会落进 `data/memory.sqlite` 跨次累积，最后把测试搞挂。
- 🧩 `tools=[]` 是合法用法（用 `is not None` 而不是 `or` 正是为此），`model or MODEL.build()` 用 `or` 则是因为模型对象永远为真。
- 📘 07 章：模型参数既可以传字符串也可以传模型对象。项目一律走**模型对象**路线（`init_chat_model`），
  因为裸字符串写法没有 `model_provider` 参数，课件也没讲怎么指定供应商。

### 5.2 Agent 循环与结束判据

📘 07 章 P21-23：

```
用户问题 → AI 思考 → 调用工具 → 观察结果 → 继续思考 → ... → 最终答案
```

📘 07 章 P26 的结束判据：**AIMessage 不包含 `tool_calls` 时即结束**。

🧩 项目直接取最后一条消息当最终答案（`runner.py:178`）：

```python
"output": result["messages"][-1].content,
```

前提是模型每轮的 tool_calls 都被 ToolMessage 回应过，否则最后一条不是 AIMessage。

### 5.3 系统提示词 `system_prompt`

📘 07 章 P28-30：可以传字符串，也可以传 `SystemMessage` 对象。三条建议：明确角色、定义输出格式、说明何时使用工具。

🧩 项目（`graph.py:653-662`）：

```python
SYSTEM_PROMPT = """你是一个自动化日常事务助手。

工作方式：
1. 先用 TodoList 把多步任务拆成清晰的子步骤，然后逐条推进。
2. 每一步做完就用 notify 或 notify_summary 把结果推给用户，不要攒到最后。
3. 通知里给结论，不要复述你的思考过程。
4. 需要读写本地文件时，路径都相对于数据目录。
5. 如果某一步失败，说明失败原因和已经尝试过的办法，不要静默跳过。
6. 缺少信息时按最合理的默认做法继续，并在结果里说明你用了什么默认值。
"""
```

**🧩 项目自己写明的局限**（`graph.py:664-680`）：
> 提示词属于模型行为约束，**不能代替程序级权限控制和错误处理**。
> 路径约束要在文件工具实现、`allow_write` 要在写工具里检查。

📘 07 章 P29：提示词分静态与动态，**动态必须借助中间件**。项目 = 静态 `SYSTEM_PROMPT` + `ContextInjectionMiddleware` 动态注入。

### 5.4 `name`：运行时身份 ID

📘 07 章 P27：设置后 Agent 产生的 AIMessage 会携带该 name，用于流式归因、消息身份标记、trace 可读性、前端展示、日志检索。

🧩 项目用 `name="automation_agent"`，注释定位为「标识、日志和调试」，并注明「不会自动改变 Agent 的推理能力」。

### 5.5 `stream_mode` 七种模式

📘 07 章 P68 / P75 对照表：

| 模式 | 输出内容 | 场景 | 项目用了吗 |
|---|---|---|---|
| `values` | 每步执行后输出**完整状态** | 每步要完整状态、状态持久化 | ✅ 取最终回答与消息数 |
| `updates`（默认） | 每步只输出**增量**变化 | 监控执行进度 | ❌ |
| `messages` | 流式返回 token 及元数据 | 打字机效果、实时交互 | ✅ 归一化为 `model.text` |
| `tasks` | task 开始/结束时间、结果、错误 | 监控任务生命周期 | ❌ |
| `debug` | 类似 tasks，多时间戳与 task 类型 | 调试、监控 | ❌ |
| `checkpoints` | 检查点创建时触发 | 状态持久化、工作流恢复 | ❌ |
| `custom` | `get_stream_writer()` 自定义数据 | 业务进度、自定义日志 | ✅ EventBridge 发出 `tool.*` 事件 |

**🧩 项目当前使用 `messages`、`values`、`custom` 三种模式**，其余模式暂未接入。

流式事件的**归一化处理**位于 `runtime/events.py`：
- `ThinkFilter`：滤掉模型混在正文里的 `<think>` 思考块；
- `is_hint_node`：按 `langgraph_node` 滤掉中间件插入的例行提示。

WebSocket 事件流由 `runtime/consumer.py` 消费 `messages`、`custom` 与 `values`；
多模式的迭代形式仍是 `(stream_mode, chunk)` 二元组。

**坑**
- 📘 07 章 P75-76：**多模式的迭代形式是 `(stream_mode, chunk)` 二元组**，不是裸 payload。
- 🧩 `messages` 模式的 payload 是 `(chunk, metadata)` 二元组；`chunk` 可能是 `AIMessageChunk`（增量）也可能是整条 `AIMessage`（中间件插入的消息）。当前事件流要同时处理这两种形态。
- 🧩 **工具调用只能从 `values` 里认** —— messages 模式下非流式模型不吐 `tool_call_chunks`。
- 🧩 `values` 每步推整个状态字典，`messages` 列表随轮次线性膨胀。多模式调用时要用 `mode` 分流，不能把两种 payload 混着解。
- 🧩 `meta` 里带 `langgraph_node`，用它过滤中间件噪音 —— 判定逻辑在 `runtime/events.py` 的 `is_hint_node`。

### 5.6 `invoke` 与 `stream` 的返回值差异

```python
# invoke：返回最终状态字典
response = agent.invoke({"messages": [...]})
final_answer = response['messages'][-1].content

# stream(values)：每步产出同样的状态字典
for chunk in agent.stream({"messages": [...]}, stream_mode="values"): ...

# stream(messages)：产出 (chunk, metadata) 二元组
for chunk in agent.stream({"messages": [...]}, stream_mode="messages"): ...
```

### 5.7 结构化输出 `response_format`：1.x 必须显式包策略类

📘 07 章 P36-39，四种策略：

```python
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy, AutoStrategy

# ① ProviderStrategy —— 用供应商的原生结构化输出能力
agent = create_agent(model=model, response_format=ProviderStrategy(ContactInfo))

# ② ToolStrategy —— 用「虚拟工具」实现，兼容绝大多数支持工具调用的模型
agent = create_agent(model=model, tools=[search_tool],
                     response_format=ToolStrategy(ContactInfo))
print(result["structured_response"])

# ③ AutoStrategy —— 自动选（源码可见，官方文档未列）
agent = create_agent(model=model, response_format=AutoStrategy(ContactInfo))
```

**📘 07 章 P39 原文警告**：
> 特别注意：在 LangChain 1.0 及以上版本中，直接传递模式（如 `response_format=ContactInfo`）不再支持，
> 必须显式使用 ToolStrategy 或 ProviderStrategy。

**🧩 项目未使用** —— `create_agent` 没有 `response_format` 参数，返回值按自然语言处理。
加的时候按 1.x 规范显式包 `ToolStrategy(Schema)`。

**`ToolStrategy` 的三个参数**（📘 07 章 P37、P40）：

```python
ToolStrategy(
    schema=Schema,                    # 必需，支持 Pydantic / TypedDict / JSON Schema / dataclass / Union
    tool_message_content="已成功抽取信息",  # 自定义伪 ToolMessage 内容，省 token
    handle_errors=True,               # 校验失败的重试策略
)
```

`handle_errors` 五种取值：`True`（默认，捕获所有异常让模型重试）/ `False`（直接抛）/ 字符串 / 异常类型 / callable。

**坑**
- 📘 07 章 P40：默认情况下 ToolMessage 里塞的是**完整数据**（`Returning structured response: name='小明' ...`），
  设了 `tool_message_content` 才变简短。无论怎么设，`result["structured_response"]` 里的数据都正确。
- 📘 07 章 P43：系统提示词里结构化输出的提示要放在**最后**，放前面可能导致后面的工具不再被调用。

### 5.8 限次：`recursion_limit` vs 中间件

📘 07 章 P26：

```python
config = {"recursion_limit": 5}   # 最多 5 步
```

🧩 项目**不用** `recursion_limit`，改用中间件限次（见 6.10）—— 因为中间件能给出更清楚的错误信息。

---

## 六、中间件（08 章）—— 本项目精华

### 6.1 中间件是什么

📘 08 章 1.1：中间件是挂在 `create_agent()` 执行流程关键节点上的扩展点，
在不改主流程源码的前提下插入、修改或替换行为。

课件把 Agent 的四个部件类比成：**模型 = 大脑、工具 = 手脚、系统提示词 = 角色、中间件 = 中枢**。

📘 08 章 1.2：它解决的是「横切关注点」分离 —— 日志、鉴权、重试、风控、审计这些
与业务无关但与执行过程强相关的逻辑，塞进主流程会导致臃肿、不可复用、粒度不够。

🧩 项目三个自定义中间件正好对应三类横切需求：

| 中间件 | 需求类别 | 钩子 |
|---|---|---|
| `TaskGateMiddleware` | 鉴权 / 风控 | `after_model` |
| `ContextInjectionMiddleware` | 转换（改提示词） | `before_model` + `wrap_model_call` |
| `TimingMiddleware` | 日志与分析 | `after_agent` |

### 6.2 六个钩子

📘 08 章 5.2：

```
类型1：Node-style hooks（节点风格）
    before_agent ：Agent 开始运行之前
    before_model ：模型调用之前
    after_model  ：模型调用之后
    after_agent  ：Agent 流程全部完成之后

类型2：Wrap-style hooks（包装风格）
    wrap_model_call （包裹模型调用）
    wrap_tool_call  （包裹工具调用）
```

**🧩 项目用到了其中四个**：

| 钩子 | 位置 | 用途 |
|---|---|---|
| `before_model` | `custom.py:164` | 只读模式下提前提醒一句 |
| `after_model` | `custom.py:98` | 检查本轮有没有写工具调用，有就拦下 |
| `after_agent` | `custom.py:226` | 统计 token 与耗时 |
| `wrap_model_call` | `custom.py:181` | 把运行时上下文追加进系统提示词 |

`before_agent` 与 `wrap_tool_call` 未使用。

**时机关系**（📘 08 章 5.3.1）：`before_agent` 先于 `before_model`；`after_agent` 晚于 `after_model`。
`after_agent` 是**整轮 Agent 跑完**才触发，不是「一轮模型调用后」。

### 6.3 执行顺序：三条规则（📘 08 章 5.6，重点）

```
before_* 钩子函数：从前到后执行
after_*  钩子函数：从后往前执行
wrap_*   钩子函数：洋葱架构，前面的包裹后面的
```

> **这里的顺序并非定义顺序，而是创建 Agent 时传递中间件的顺序。**

课件的实际输出（3 个 before / 3 个 after / 3 个 wrap 混排）：

```
Human Message: 你好啊...  -> before_model-1 <-  -> before_model-2 <-  -> before_model-3 <-
                 -> wrap_model-before-1 <-  -> wrap_model-before-2 <-  -> wrap_model-before-3 <-
Ai Message:    你好啊！   -> wrap_model-after-3 <-  -> wrap_model-after-2 <-  -> wrap_model-after-1 <-
                 -> after_model-3 <-  -> after_model-2 <-  -> after_model-1 <-
```

**🧩 项目的中间件栈**（`graph.py`）：

```python
middleware = [
    *LIMITS,          # 模型/工具调用次数限制
    *RETRY,           # 模型/工具失败重试
    TODO,             # TodoList 任务管理
    CONTEXT_EDITING,  # 上下文清理
    *CUSTOM_MIDDLEWARE,
]
```

**为什么 `LIMITS` 放最前**：它的限次检查是 `before_model` 型 —— `before_*` 从前到后执行，
放最前 = 最先检查 = 能在模型跑飞之前拦住。放最后就晚了。
**为什么 `CUSTOM_MIDDLEWARE` 放最后**：它能看到前面所有中间件处理后的状态
（比如只读闸门拦截后，统计中间件统计到的就是被拦之后的结果）。

**坑**
- 🧩 `graph.py:277-285` 自注：**中间件的排列顺序可能影响重试与调用次数统计之间的关系** —— 重试会额外产生模型/工具调用。
- 🧩 `middleware` 必须是**扁平列表**，项目用 `*LIMITS` / `*RETRY` 解包。写 `middleware=[LIMITS, RETRY]` 会塞进两个列表对象。

### 6.4 Node-style vs Wrap-style 怎么选

| | Node-style | Wrap-style |
|---|---|---|
| 拿到什么 | `state` + `runtime` | `request` + `handler` |
| 能改什么 | **只能改状态**（追加消息、更新字段） | **能改请求本身**，也能改响应 |
| 典型用途 | 记录日志、验证、裁剪消息 | 重试、回退、缓存、**改系统提示词** |

🧩 项目里 `ContextInjectionMiddleware` **同时实现了两种**，分工明确：

```python
def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """before_model 只能往消息列表里追加内容"""
    ctx = _ctx(runtime)
    if not ctx.allow_write:
        return {"messages": [AIMessage(content="（提示：本轮为只读模式，任何写操作都会被拒绝。）")]}
    return None

def wrap_model_call(self, request, handler):
    """wrap_model_call 能拿到 request，改造完再交给 handler 继续往下走"""
```

📘 08 章 5.5 的选型口诀：**单钩子用装饰器，多钩子用类**。

### 6.5 怎么写自定义中间件

📘 08 章 5.3.1 的三条硬规则：
1. **必须继承 `AgentMiddleware`**（固定）
2. **方法名固定**为 `before_model` / `after_model` / `before_agent` / `after_agent`（固定）
3. **类名随意**（不固定）

```python
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from typing import Any

class MyMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        state["messages"][-1].content += " -> before_model <- "
        return None
```

🧩 项目（`custom.py:73`）：

```python
class TaskGateMiddleware(AgentMiddleware):
    WRITE_TOOLS = {"write_file", "append_file", "save_json", "notify", "notify_summary"}

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        ...
```

📘 08 章 5.3.2：**装饰器写法与类写法底层等价** —— 装饰器会基于你重写的方法构造一个 `AgentMiddleware` 子类实例。
但 🧩 装饰器版**拿不到实例属性**（`__dict__` 为空），想做自省/调试/复杂配置就别用闭包。项目全走类写法。

### 6.6 Node-style 返回值语义

📘 08 章 5.3.4：

```
返回 None：不修改状态
返回字典：更新状态
返回 {"jump_to": "..."}：控制流程
    "__end__"  结束 Agent
    "tools"    跳到工具节点
```

🧩 项目只用 `None`（放行）和 dict（追加消息），**没用 `jump_to`**。

### 6.7 `can_jump_to` 与 `hook_config`

📘 08 章 5.3.5：

```python
# 装饰器写法
@before_model(can_jump_to=["tools"])
def force_tool_first(state, runtime): ...

# 类写法：必须用 @hook_config
from langchain.agents.middleware import hook_config

class MyMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["tools", "end"])
    def before_model(self, state, runtime): ...
```

`can_jump_to` 可取值：`end` / `tools` / `model`。

**坑**
- 📘 08 章 5.3.5：**跳转时写 `"__end__"`（双下划线），声明可跳目标时写 `"end"`（无下划线）** —— 拼写不一样。
- 📘 08 章：`after_model` 跳回 `"model"` 会造成**无限重跳**，必须自建防重入机制（课件案例用「已注入过就不再跳」判断）。
- 🧩 `hook_config` **只用于 `can_jump_to`**，无参装饰器写法是错的。项目未使用这两者。

### 6.8 `ModelRequest` 不可变，`override()` 是替换语义

📘 08 章 5.4：

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from typing import Callable

@wrap_model_call
def add_context(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    original_content = request.system_message.content if request.system_message else ""
    new_system_message = SystemMessage(content=f"""{original_content}
    当前时间：{current_time}
    用户位置：中国
    """)
    modified_request = request.override(system_message=new_system_message)
    return handler(modified_request)
```

🧩 项目（`custom.py:204-216`，正例）：

```python
# system_prompt 只是只读 property，直接赋值已废弃；改用 override 造新 request
base = request.system_message
content = base.text if base is not None else ""

# override 是「替换」语义，不是「追加」语义。
new_request = request.override(
    system_message=SystemMessage(content=(content or "") + extra)
)
return handler(new_request)
```

**坑**
- 🧩 **`override()` 是替换语义** —— 不把原文拼回去，原系统提示词会被冲掉，模型就不知道自己该怎么扮演了。
- 🧩 `system_prompt` 只是**只读 property**（返回 `system_message.text`），直接赋值会抛 `DeprecationWarning`。
- 📘 08 章：`ModelRequest` 的 `messages` **不含 system message**（系统消息单独在 `system_message` 字段），所以改提示词要走 `system_message` + `override()`。
- 📘 08 章：`override(system_prompt=...)` 虽仍可用但已标 deprecated，应改用 `system_message=`；两者同时传会抛 `ValueError`。

### 6.9 `runtime.context` 可能为 None（🧩 课件 08 章未讲）

```python
def _ctx(runtime: Runtime) -> AgentContext:      # agent/middleware/custom.py:48
    ctx = runtime.context
    return ctx if isinstance(ctx, AgentContext) else AgentContext()
```

**为什么必须有**：context 是 `invoke` 的可选参数，不传时 `runtime.context` 就是 `None`，
中间件里直接用会抛 `AttributeError`，把整个调用打挂。

**为什么判 `isinstance` 而不是只判 `None`**：调用方理论上可以传任意对象当 context，
不是 `AgentContext` 就说明类型不对，同样退化成默认值更安全。

**🧩 项目约定**：新增中间件钩子时**一律用 `_ctx(runtime)`**，不要直接访问 `runtime.context`。

### 6.10 项目用到的六个内置中间件

| 中间件 | 参数 | 作用 | 课件 |
|---|---|---|---|
| `ModelCallLimitMiddleware` | `thread_limit=30` | 限模型调用次数，防跑飞烧钱 | 📘 08 章 3.1 |
| `ToolCallLimitMiddleware` | `thread_limit=50` | 限工具调用次数 | 📘 08 章 3.2 |
| `ModelRetryMiddleware` | `max_retries=3, backoff_factor=2.0` | 模型失败指数退避重试 | 📘 08 章 3.6 |
| `ToolRetryMiddleware` | `max_retries=2, on_failure="continue"` | 工具失败重试 | 📘 08 章 3.5 |
| `TodoListMiddleware` | 无参 | 注入 `write_todos` 工具，让模型拆解任务 | 📘 08 章 2.4 |
| `ContextEditingMiddleware` + `ClearToolUsesEdit` | `trigger=120_000, keep=3, clear_tool_inputs=True` | 上下文超限时清理旧工具结果 | 📘 08 章 3.8 |

**坑**
- 📘 08 章 3.1/3.2：**`thread_limit` 需要 checkpointer**（课件示例注释写死 `# Required for thread limiting`）。
- 📘 08 章 3.1：`thread_limit` 是**同一 thread_id 内的累计**限制 —— 🧩 定时任务复用 `cron-{任务名}` 作 thread_id，
  调用次数会跨天累积。想限制单次运行得用 `run_limit`。
- 📘 08 章 3.2：`exit_behavior` 三种：`error` / `end` / `continue`（**默认**）。🧩 项目没显式设，靠 `thread_limit=50` 兜底 ——
  默认 `continue` 在模型能力不足时**可能死循环**。
- 📘 08 章 3.5：**`max_retries` 不含首次调用** —— 课件注释写明「一共最多调 1 + 6 = 7 次」。
- 🧩 `graph.py:217-221`：`on_failure="continue"` **不保证模型一定会向用户报告失败**。
- 📘 08 章 2.4：`TodoListMiddleware` **只是提供工具，不保证模型一定先拆解任务**（课件 2.4.2 给了决策树）。
- 📘 08 章 3.8：`ContextEditingMiddleware` **不会更改持久化的消息列表**，所以验证效果要看 `usage_metadata` 而不是看 messages。
  🧩 `clear_tool_inputs` 这个参数课件未列出，是项目自用的扩展项。

### 6.11 未启用但值得加的三个内置中间件

| 中间件 | 签名 | 为什么加 / 为什么不加 |
|---|---|---|
| `SummarizationMiddleware` | `SummarizationMiddleware(model=..., trigger=[("tokens",100)], keep=("messages",2))` | ⚠️ 与已有 `ContextEditingMiddleware` 功能部分重叠 |
| `HumanInTheLoopMiddleware` | `HumanInTheLoopMiddleware(interrupt_on={"send_email_tool": {"allowed_decisions": ["approve","reject"]}})` | ⚠️ 人工审批与「无人值守定时任务」定位冲突 |
| `ModelFallbackMiddleware` | `ModelFallbackMiddleware("openai:gpt-4o-mini")` | ⚠️ 需要先决定备用供应商和 key 从哪来，属于配置决策 |

📘 08 章 2.3 补充：`SummarizationMiddleware` 的摘要阈值经验值（4k 窗口设 3000、8k 设 6000、16k 设 12000），
要给工具调用和系统提示留余量。

### 6.12 PII 脱敏（已启用，`agent/middleware/pii.py`）

📘 08 章 2.3 提到的内置中间件，课件没展开。

**为什么需要**：提示词与工具结果都会发给模型供应商，还可能因 LangSmith 追踪上云。
用户指令、文件内容里的手机号 / 邮箱 / 身份证号不该原样出去。

**挂法** —— `PIIMiddleware` 的构造函数一次只接一种 `pii_type`，所以要脱敏几种就挂几个实例：

```python
PIIMiddleware("email",       strategy="redact", apply_to_input=True, apply_to_tool_results=True)
PIIMiddleware("credit_card", strategy="redact", apply_to_input=True, apply_to_tool_results=True)
PIIMiddleware("phone_cn",    strategy="redact", detector=PHONE_CN_PATTERN, ...)
PIIMiddleware("id_card_cn",  strategy="redact", detector=ID_CARD_CN_PATTERN, ...)
```

**内置的 5 种类型**：`email` / `credit_card` / `ip` / `mac_address` / `url`。
项目只用前两种 —— `url` 和 `ip` 脱敏会破坏正常信息（用户让 agent 查某个 URL 就废了）。

**自定义类型要自己给 detector**（正则字符串或 callable），否则构造时直接 `ValueError`。

| 类型 | 正则 | 占位符 |
|---|---|---|
| 手机号 | `(?<!\d)1[3-9]\d{9}(?!\d)` | `[REDACTED_PHONE_CN]` |
| 身份证 | `(?<!\d)\d{17}[\dXx](?!\d)` | `[REDACTED_ID_CARD_CN]` |

边界断言 `(?<!\d)` / `(?!\d)` 不能省 —— 否则更长数字串里的一段会被当手机号。
实测「订单 20261103 共 1234567890 元」不会被误伤。

**脱敏哪些内容**（`before_model` 钩子）：

| 内容 | 处理 | 说明 |
|---|---|---|
| 用户指令 | 只处理**最后一条** HumanMessage | 历史里的旧指令已经进过模型了 |
| 工具结果 | 最后一条 AIMessage 之后的**全部** ToolMessage | **文件工具读出来的内容走这条**，是主要泄露通道 |
| 模型输出 | **不脱敏**（`apply_to_output=False`） | 模型输出多半来自用户自己的输入，用户自己知道那是什么 |

**strategy 选 `redact` 不选 `mask`**：mask 只保留末尾几位（手机号是 `****5678`）——
那几位仍是真实个人信息，照样会进供应商和 LangSmith。redact 整个替换成 `[REDACTED_EMAIL]`。

**坑**
- 🧩 **脱敏改写的是 state 里的消息**，所以落进 checkpointer 的也是脱敏后的版本 ——
  原始 PII 不入库。这是好事，但也意味着**改脱敏策略不会回溯已存的历史**。
- 🧩 **没检测到 PII 时中间件返回 `None`**，不改 state —— 所以它不会干扰
  `TaskGateMiddleware` 等其他中间件对消息列表的判断。
- 🧩 脱敏是**全局**的。四种类型一起开关，没法只脱敏手机号不脱敏邮箱。
- 🧩 开关是 `.env` 的 `PII_MASKING`（默认 `1`）。
  想让 agent 处理一份本来就该被它看到的通讯录时改成 `0`。

### 6.13 事件桥接（`agent/middleware/events.py`）

📘 08 章 5.3 的 wrap-style 钩子。桌面版的事件源。

**它解决什么**：`messages` 流里只有模型的输出和工具结果，没有「工具什么时候开始、
跑了多久、成功还是失败」。桌面版的工具卡片需要这些，所以用 `wrap_tool_call`
包住工具执行，前后各发一条 `custom` 事件。

| 事件 | 时机 | 字段 |
|---|---|---|
| `tool.start` | 调用前 | `name` / `call_id` / `args` |
| `tool.end` | 成功返回后 | `name` / `call_id` / `ok` / `duration_ms` / `result_preview` |
| `tool.error` | 失败后 | `name` / `call_id` / `message` / `duration_ms` |

**8 个工具函数一行都不用改** —— 事件全部从中间件这一层发出去。

**装配在中间件栈的最外层**（`agent/graph.py` 的 `middleware` 列表第一项）。
wrap-style 钩子是洋葱结构，第一个定义的最外层，所以它包住的是
**「工具调用 + 重试」整体**：

- 一次工具调用只发一对事件，不会因为重试 3 次发 3 条 error；
- `duration_ms` 是用户真正等待的时间（含重试）。

**🧩 失败有两种形态，两种都要处理**：

| 形态 | 什么时候 | 怎么判 |
|---|---|---|
| handler 直接抛异常 | 没有重试中间件兜着时 | `except` 捕获 → 发 tool.error → **继续往外抛**（中间件不吞异常） |
| 正常返回但 `status="error"` | 项目常态：`ToolRetryMiddleware` 的 `on_failure="continue"` 把重试耗尽后的失败包成 ToolMessage | 检查 `result.status` |

只处理第一种会**漏报失败**（在项目里失败走的是第二条路）；
只处理第二种会**误报成功**。

**坑**
- 🧩 `get_stream_writer()` 只能在图的执行上下文里调用，外面调会抛
  `RuntimeError: Called get_config outside of a runnable context`。
  取 writer 的地方必须 try/except 兜底 —— 发不出去一个事件不该搞崩整个调用。
- 🧩 `run_id` 由归一化层统一盖上，中间件不用管。
- 🧩 只读闸门拦下写工具时工具**根本没执行**，所以不会有 tool.start / tool.end
  —— 没跑的工具不该有执行事件。
- 🧩 `result_preview` 截断到 500 字符：工具结果可能几百行，完整塞进事件会撑爆界面。

---

## 七、上下文与记忆（09 章）

### 7.1 四个概念的分工（📘 09 章 1.2.2，本章骨架）

| 上下文类型 | 描述 | 可变性 | 生命周期 | 访问方法 | 项目载体 |
|---|---|---|---|---|---|
| **动态运行时上下文** | 单次运行中演变的可变数据 | 动态 | 单次运行 | LangGraph **state** 对象 | `AgentState`（messages / jump_to） |
| **动态跨会话上下文** | 对话间共享的持久数据（用户偏好、历史洞察、知识条目） | 动态 | 跨对话 | LangGraph **store** 对象 | `InMemoryStore`（当前几乎没用） |
| **静态运行时上下文** | 启动时传入的用户元数据、工具、数据库连接 | 静态 | 单次运行 | LangGraph **context** 对象 | `AgentContext` |

**checkpointer 不在这张表里** —— 它是 state 的**持久化机制**（把 state 存成检查点快照）。

### 7.2 短期记忆 = State + Checkpointer + Thread ID

📘 09 章 PAGE 6 原文：

```
State       ：默认存储历史消息列表 messages，通过 State 管理历史消息
Checkpointer：负责将 State 作为检查点持久化保存，检查点是某个时刻的 State 快照
Thread ID  ：用于唯一标识 State，LangChain 运行时会按照 thread_id 读写 State 快照
```

课件类比：「就像玩 RPG 游戏时的『自动存档』」。

📘 09 章 1.3.2 明确：**1.x 通过 state 和 store 构建记忆系统**，没有专门的记忆类 ——
记忆由 checkpointer 持久化 state 来实现。

### 7.3 checkpointer 自动管理历史（📘 09 章 12-13 页）

```
① 读取之前的历史
② 追加新消息
③ 调用模型（传入完整历史）
④ 保存新的历史
```

**🧩 项目只传一条新消息**（`runner.py:165`），历史全靠 checkpointer。

📘 09 章 10 页强调：「所有消息历史都会自动追加到 AgentState 的 messages 字段中，**无需手动维护**」—— 别自己拼接历史消息，那会重复。

### 7.4 `InMemorySaver` vs `SqliteSaver`

📘 09 章 PAGE 11：

> 注意：InMemorySaver 内存中保存，进程结束就丢失数据，适合测试。
> 生产环境可换成数据库持久化的 SqliteSaver、PostgresSaver 等

📘 09 章 PAGE 16 的 `setup()` docstring：

> This method creates the necessary tables ... and runs database migrations.
> **It MUST be called directly by the user the first time checkpointer is used.**

🧩 项目（`graph.py:292-417`）：

```python
conn = sqlite3.connect(path, check_same_thread=False)
saver = SqliteSaver(conn)
saver.setup()          # 建表，重复执行不会重建（Create If Not Exists）
```

**坑**
- 📘 09 章 24 页：`InMemorySaver()` 进程结束**或重建 Saver()** 则历史状态丢失。
- 📘 09 章 14 页：**`InMemorySaver` 没有提供删除 API** —— 想清空某会话历史只能换 thread_id。
- 🧩 `graph.py:404-405` 自注：「当前实现没有显式关闭 `conn` 的逻辑，长期运行时应由应用管理连接生命周期」。

### 7.5 `SqliteSaver.from_conn_string()` 是上下文管理器（🧩 项目踩过）

实测 `langgraph-checkpoint-sqlite 3.1.2`：

```python
@classmethod
@contextmanager
def from_conn_string(cls, conn_string: str) -> Iterator[SqliteSaver]:
    with closing(sqlite3.connect(conn_string, check_same_thread=False)) as conn:
        yield cls(conn)
```

**坑**：`with` 退出时连接会被 `closing()` 关掉。直接把它传给 `create_agent` 会报 **`Invalid checkpointer`**。
要么像项目这样自持一条 `sqlite3` 连接，要么保证 Agent 的生命周期始终在 `with` 块内。

（📘 09 章只给了 Postgres 版本：
`with PostgresSaver.from_conn_string(DB_URL) as checkpointer: checkpointer.setup()`）

### 7.6 `thread_id` 如何隔离记忆

📘 09 章 13 页的正误对照：

```python
# ❌ 没有 checkpointer → 不会记住
# ❌ 没有 config → 不会记住
# ❌ thread_id 不同 → 不同会话
# ✅ 同一个 thread_id → 记得
agent = create_agent(model=model, tools=[], checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "1"}}
agent.invoke({...}, config)
agent.invoke({...}, config)  # 记得！
```

📘 09 章 11 页两条结论：「同一个 thread_id 共享记忆，不同 thread_id 完全隔离」「thread_id 是记忆管理的核心开关」。

🧩 项目的 thread_id 命名规则（`runner.py:119-122`）：

| 入口 | thread_id | 效果 |
|---|---|---|
| 定时任务 | `cron-{任务名}` | 跨天保留记忆 |
| `runner.run_task()` 不传 | `task-{随机8位}` | 一次性，不留痕 |
| 显式传 thread_id | 调用方自己给 | 同值即共享记忆 |

### 7.7 静态运行时上下文：`context_schema=` 定义，`context=` 传入

📘 09 章 PAGE 60-62：

```python
@dataclass
class UserContext:
    username: str

agent = create_agent(
    model="deepseek-chat",
    middleware=[CheckCredit()],
    store=store,
    context_schema=UserContext
)
response1 = agent.invoke(
    {"messages": ["你好啊，你知道 Ada Lovelace 的贡献吗？"]},
    context=UserContext(username="Ada Lovelace")
)
```

📘 09 章 PAGE 60 正文：
> 静态运行时上下文表示不可变的数据……通常在运行开始时通过 **invoke / stream 的 context 参数**传递，
> 此类数据在运行期间不会更改。

🧩 项目三处对应：

```python
# 1. 定义结构：agent/context.py:57
@dataclass
class AgentContext:
    task_name: str = "manual"
    trigger: Trigger = "cli"        # Literal["schedule","api","cli","lab"]
    allow_write: bool = True
    notify_channel: str = "terminal"

# 2. 声明：agent/graph.py:609
context_schema=AgentContext,

# 3. 传入：runtime/runner.py:165
result = agent.invoke(
    {"messages": [HumanMessage(content=instruction)]}, config, context=ctx
)
```

**🧩 项目最大的坑**（`runner.py:154-164` 原文）：

```python
agent.invoke(msgs, {"configurable": {...}}, context=ctx)            # ✅
agent.invoke(msgs, {"configurable": {..., "context": ctx}})          # ❌
```

写错时 `invoke` **不报错**，但 `runtime.context` 会是 `None`，
中间件兜底后表现为「任务名和写权限都丢了」这种极难查的问题。

**📘 09 章的笔误**：PAGE 60 正文写「用户自定义 ContextSchema 用 `@dataclass` 修饰，在 Agent 创建时通过
**`state_schema`** 参数传递」，但同页代码用的是 `context_schema=UserContext`。
`state_schema` 是扩展 AgentState 用的，此处应为 `context_schema`。项目用的是正确的写法。

**🧩 `thread_id` 不放在 context 里** —— 它是 checkpointer 使用的会话标识，走 `config["configurable"]["thread_id"]`。

### 7.8 在钩子里读 context 的三种方式

📘 09 章 PAGE 57-58 / PAGE 72：

```python
# Node-style：从 runtime.context 读
def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...

# Wrap-style：wrap_model_call
def wrap_model_call(self, request: ModelRequest[ContextT],
                    handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]]): ...

# Wrap-style：wrap_tool_call
def wrap_tool_call(self, request: ToolCallRequest,
                   handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]): ...

# dynamic_prompt：官方动态改写系统提示词的方式
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    username = request.runtime.context.username
    store = request.runtime.store
    preferences = store.get(namespace, username).value["chat_preferences"]
    return f"# 用户偏好\n{'\n'.join(preferences)}"
```

🧩 项目实现了前两种，**第三种没实现** —— `dynamic_prompt` 是动态改写系统提示词的官方方式，值得练一次。

**坑**
- 📘 09 章 61 页的示例 `context = runtime.context; username = context.username` **没有 None 保护** ——
  不传 context 直接 AttributeError。项目补的 `isinstance` 判断比只判 `None` 更严。
- 🧩 `dynamic_prompt` 与项目的 `wrap_model_call` 语义不同：`dynamic_prompt` 是**整段替换**系统提示词，
  项目是**追加**到末尾（保留原有 `SYSTEM_PROMPT`）。
- 📘 09 章 74 页：`dynamic_prompt` **不会更改消息列表**，所以消息列表里看不到相关提示词。

### 7.9 记忆治理三路

📘 09 章 2.4 —— state 会无限增长，必须治理：

| 路径 | 时机 | 手段 | 特点 |
|---|---|---|---|
| **消息裁剪** | 调用模型**前** | `RemoveMessage(id=REMOVE_ALL_MESSAGES)` + 保留近期 | 临时视图，不改状态 |
| **消息删除** | 调用模型**后** | `RemoveMessage(id=m.id)` 按 id 删 | **永久更改状态** |
| **摘要** | 触发阈值 | `SummarizationMiddleware` | 保语义，丢细节 |

📘 09 章 30 页的**墓碑机制**：
> 框架收到 `RemoveMessage(id="1")` 后，并不会去内存的数组里把 id="1" 的对象删掉，
> 而是把这个 RemoveMessage 作为一条新记录追加到当前线程的状态历史中 —— 下一次 invoke 时由 Reducer 把被标记的消息过滤掉。

**🧩 项目走的是另一条路** —— 工具结果清理（`ContextEditingMiddleware` + `ClearToolUsesEdit`），
属于 08 章 3.8，不是裁剪/删除/摘要。这条路还没走。

### 7.10 长期记忆 store（🧩 项目未启用）

📘 09 章 PAGE 40-44 —— 四层架构 `store → namespace → key → value`：

```python
namespace = ("users", "user_123", "preferences")   # 元组
key = "profile"                                     # 字符串
value = {"language": "zh-CN", "style": "short_direct"}  # 字典
store.put(namespace, key, value)
```

三个基础 API：`store.put(namespace, key, value)` / `store.get(namespace, key)` / `store.search(namespace_prefix, query=..., filter=..., limit=10)`。

🧩 项目注入了 `InMemoryStore()`（`graph.py:584-588`）但**没有任何工具或中间件读写它**。

**坑**
- 📘 09 章 43 页：`InMemoryStore` 每次 `put` 都创建**新 Item**（`created_at` 与 `updated_at` 始终一致）；
  `PostgresStore` 是 update 语义。
- 📘 09 章 48-49 页：语义检索需要初始化时配 `index`（`{"embed":..., "dims":..., "fields":[...]}`），
  不配 `index` 时 `search(query=...)` 不可用。
- 📘 09 章 52 页：工具里用 `ToolRuntime` 访问 store：`runtime.store.put(...)`。
  `ToolRuntime` 默认泛型是 `[None, Any]`，不显式指定上下文类型时底层认为 context 是 None，
  `invoke` 传了 context 会抛警告 —— 正确写法是 `ToolRuntime[UserContext, Any]`。

---

## 八、LangGraph 图运行时（`courseware/langGraph/`）

前七章讲「LangChain 的 API 怎么用」，这一章讲**底下那层图运行时** ——
项目每个 Agent 都是一张 LangGraph 状态图，只是图是 `create_agent` 替你画的。

> 📘 标注约定：本章的 `LG XX 章` 指 `courseware/langGraph/` 里的专项课件
> （6 篇 Markdown，章节号 1-13 连续），与前面 LangChain PDF 课件的 01-10 章**不是同一套编号**。

### 8.1 项目跑的那张图长什么样（实测）

`create_agent()` 返回的 `CompiledStateGraph` 底层就是 `StateGraph` 编译出的 Pregel 图。
实测 `agent.get_graph()`：

| 节点 | 是什么 |
|---|---|
| `__start__` | 入口 |
| `model` | 模型节点 |
| `tools` | `ToolNode` —— 框架自带的工具执行节点 |
| `__end__` | 出口 |

| 边 | 条件边？ | 含义 |
|---|---|---|
| `__start__ → model` | 否 | 固定 |
| `model → tools` | **是** | 模型这轮有 `tool_calls` 就走工具 |
| `model → __end__` | **是** | 模型给了最终答案就结束 |
| `tools → model` | **是** | 工具结果回灌给模型 |

```python
print(agent.get_graph().draw_mermaid())   # 直接把这张图画出来
```

**🧩 这张图解释了本项目最难的那个坑**（见 4.6）：
「只补 ToolMessage、不补收尾消息」—— agent 图的路由看的是**最后一条 AIMessage 的 `tool_calls`
是否都已被回应**。补了 ToolMessage → 全部已回应 → 判定为「人工注入的工具消息」→ 从 `model` 节点重新跑一轮；
顺手补一条 AIMessage → 路由变成走 `tools` → 模型不再重新回答。

### 8.2 State 与 Reducer

📘 LG 01 章 3.1-3.5（状态定义、Reducer、节点访问 State、预定义状态）

`AgentState` 实际只有三个键（实测注解）：

| 键 | 类型 | reducer | 说明 |
|---|---|---|---|
| `messages` | `list[BaseMessage]` | `add_messages`（编译后是 `BinaryOperatorAggregate` 通道） | 消息历史 |
| `jump_to` | `Literal["tools","model","end"] \| None` | `EphemeralValue` | 控制流跳转，**不持久化** |
| `structured_response` | `ResponseT` | `LastValue` | 结构化输出 |

**Reducer 是状态图的核心概念**（📘 LG 01 章 3.2）：签名 `(Value, Value) -> Value`，
通过 `Annotated[Type, reducer_function]` 给状态键指定；**没指定 reducer 的键默认覆盖（Last-Write-Wins）**。

- `messages` 的 schema 声明的是 `add_messages`（`AgentState.messages: Annotated[list[AnyMessage], add_messages]`），
  编译后通道类型显示为 `BinaryOperatorAggregate`。实测语义 = **追加 + 按 id 替换** ——
  两条 `id` 相同的消息只留后一条（`add_messages` 的 docstring 原话：append-only，unless the new message has the same ID）。
- `jump_to` 是 `EphemeralValue` —— **只在本轮有效，不写进检查点**
- `structured_response` 是 `LastValue` —— 后写覆盖先写

📘 LG 01 章 3.3：`Overwrite` 可以绕过 reducer 直接覆盖某个键。

**🧩 项目怎么改 state**：不直接写节点，全靠中间件的返回值 ——
Node-style 钩子返回 `dict` 就是一次 state 更新（见 6.6）。
`TaskGateMiddleware` 返回的 `{"messages": [ToolMessage(...)]}` 走的就是 `messages` 这个 channel。

### 8.3 可恢复执行：检查点存的是「每一步的快照」

📘 LG 03 章 6.1-6.5

启用 checkpointer 后，LangGraph 在每个**超步（super-step）**结束时存一份检查点。实测 API：

```python
agent.get_state(config)                # → StateSnapshot：当前快照（values / next / config）
list(agent.get_state_history(config)) # → 历史快照列表
```

实测一次**不带工具**的简单调用（`InMemorySaver` + 一次 `invoke`）产生了 **3 个检查点**，
每个的 `metadata` 里有 `step` / `source` / `parents`。

**持久化模式**（📘 LG 03 章 6.3）—— 三档取舍：

| 模式 | 写入时机 | 容灾能力 |
|---|---|---|
| `exit` | 运行退出时 | 最弱（扛不住进程崩溃） |
| `async`（默认） | 每个超步末尾写主检查点，后台异步写 | 较强 |
| `sync` | 进下一个超步之前等写完 | 最强，响应延迟最大 |

**🧩 项目没显式设模式**，用默认的 `async` + `SqliteSaver`（`graph.py:292`）。

### 8.4 节点容错：LangGraph 的 RetryPolicy vs 项目的中间件重试

📘 LG 02 章 5.1-5.7

LangGraph 在**节点层**提供三种异常处理机制，执行顺序是：
超时/异常 → `RetryPolicy` 判断是否重试 → 重试直到次数耗尽 → `error_handler` → 都没配就向外抛。

**🧩 但当前版本有天花板**：实测 `langgraph 1.1.2` 的 `RetryPolicy` 只有重试字段
（`initial_interval` / `backoff_factor` / `max_interval` / `max_attempts` / `jitter` / `retry_on`），
**没有 `timeout`、没有 `error_handler`** —— 这两个要 langgraph ≥ 1.2 才有
（📘 LG 02 章 5.1 原注：「本课程当前环境为 langgraph==1.1.2，因此重点讲解重试机制和节点缓存」）。

**🧩 项目压根没用节点级 `RetryPolicy`**，走的是中间件层：

| | 节点级 `RetryPolicy` | 项目的中间件重试 |
|---|---|---|
| 粒度 | 单个节点 | 整个模型调用 / 整个工具调用 |
| 生效时机 | 节点抛异常时 | 同样，但由中间件包住 |
| 错误消息 | 框架生成 | `ModelRetryMiddleware` / `ToolRetryMiddleware` 可定制 |
| 项目用了吗 | ❌ | ✅ `graph.py:194` 的 `RETRY` |

选中间件而不是节点策略，是因为**错误消息可控** —— 工具失败时给模型一句能照做的中文说明，
而不是 `failed after 3 attempts ... Please try again.`（见 4.7）。

### 8.5 图里的运行时上下文

📘 LG 03 章 7.3

`context_schema=` 声明的 dataclass 会在运行时通过 `runtime.context` 暴露给每个节点与钩子。
它是**静态**的 —— 一次运行内不变、不持久化（详见 7.7）。

**🧩 项目对应**：`agent/context.py` 的 `AgentContext`，
`agent/graph.py:609` 声明、`runtime/runner.py:165` 传入。
`runtime.context` 可能为 `None` 是因为 context 是可选参数（见 6.9）。

### 8.6 中断与部署（项目未接）

📘 LG 04 章 8-10

- **动态中断** `interrupt()`：节点里调 `interrupt(payload)` 抛起，图暂停，payload 暴露给调用方；
  恢复时用 `graph.invoke(Command(resume=...))`，**`resume` 的值会成为 `interrupt()` 的返回值**，
  参与后续计算。`langgraph.types.interrupt` 在 1.1.2 里可用。
- **静态断点**：`compile(interrupt_before=["tools"])`，编译时就指定在哪停。
- **工具节点** `ToolNode`：项目用的就是它（见 8.1），课件 10.1 讲的是手写版。
- **部署**：`langgraph dev` 起本地 Agent Server + Studio，AgentChatUI 接前端。

**🧩 项目为什么套不上 `langgraph dev`**：那个命令要模块导出一个**图对象**，
而项目 `runtime/server.py` 导出的是 FastAPI `app`。要用得再导一个 `graph = build_agent()`。

**🧩 中断 ≠ 桌面版的「取消」**：桌面版第一步做的是**协作式取消**
（消费循环每轮检查标志位后 `break`），不是 `interrupt()`。
后者要设计暂停点并处理检查点恢复，是另一件事（见 `DESKTOP_AGENT_PLAN.md`）。

### 8.7 子图与设计模式（项目未接）

📘 LG 05 章 12-13

- **子图**：把编译好的图当节点塞进父图。两种嵌入方式 —— `compile()`（每次调用独立状态）
  与 `compile(checkpointer=True)`（按 thread 共享状态）。
- **设计模式**：路由（routing）、编排者-工人（orchestrator-workers）、评估者-优化者（evaluator-optimizer）。

**🧩 项目现状**：单张 ReAct 循环图（`model ⇄ tools`）。路由模式天然具备
（模型决定调不调工具），编排者-工人要手写 `StateGraph`。

### 8.8 一张表看清用到的 / 没用到

| LangGraph 能力 | 项目 |
|---|---|
| `StateGraph` 与编译 | 间接 —— `create_agent` 编译出来的 |
| 状态与 Reducer | ✅ `AgentState.messages` |
| 条件边 | ✅ `model ⇄ tools` |
| checkpointer | ✅ `SqliteSaver` / `InMemorySaver` |
| store | ⚠️ 挂了 `InMemoryStore`，没实际用 |
| 运行时上下文 | ✅ `AgentContext` |
| 节点级 `RetryPolicy` | ❌ 走中间件重试 |
| 显式设置持久化模式 | ❌ 用默认 `async` |
| `interrupt()` | ❌ 桌面版用协作式取消 |
| 子图 | ❌ |
| `langgraph dev` 部署 | ❌ 自建 FastAPI |
| `astream_events` | ❌ 当前用同步 `agent.stream()` + WebSocket 事件协议 |

---

## 九、项目自研的运行时外壳（课件未覆盖）

这一层没有课件对应，是项目自己实现的部分。

### 9.1 共享 Agent 单例与调用路径

**设计要点**：`get_agent()` 提供进程内共享的 Agent 单例。定时任务和 Python 调用方通过
`run_task()` 调用 `agent.invoke()`；WebSocket 端点复用同一实例，由 `StreamConsumer` 调用
`agent.stream()`。两条执行路径都显式构造 `AgentContext`，并共享 checkpointer 的会话记忆。

```python
_AGENT = None          # 模块级惰性单例

def get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT
```

**为什么必须复用**：每次 `build_agent()` 会多开一条 SQLite 连接、多建一套中间件，
而且不同 Agent 的 thread_id 记忆是隔离的，换 Agent 等于把之前的会话记忆丢掉。

### 9.2 定时必须自己接 APScheduler

**LangGraph 1.x 本地没有内置 cron**（`langgraph_sdk` 也未导出 cron，那是云端 LangGraph Platform 的能力）。

```python
CronTrigger.from_crontab(cron_expr, timezone=RUNTIME.timezone)   # 标准 5 段格式
```

### 9.3 FastAPI 用 `lifespan` 而不是 `@app.on_event`

📘 课件未覆盖。FastAPI 0.135 已废弃 `@app.on_event`（🧩 记在 `CLAUDE.md` 陷阱 11）：

```python
@asynccontextmanager
async def lifespan(app):
    # 启动逻辑放 yield 前
    yield
    # 清理逻辑放 yield 后
```

### 9.4 配置层：`.env` 覆盖环境变量，所以命令行开关要走独立通道

🧩 `agent/config.py` 的 `set_cli_override()`：

```python
_CLI_PREFIX = "__CLI_"

def set_cli_override(name: str, value: str) -> None:
    os.environ[f"{_CLI_PREFIX}{name}__"] = str(value)
```

**为什么**：`config.py` 用 `load_dotenv(override=True)`（.env 覆盖环境变量），
直接设进 `os.environ` 的值会在 reload 时被 `.env` 里的同名项顶掉 ——
`.env` 默认带 `SCHEDULER_ENABLED=1`，`--no-schedule` 形同虚设。
`set_cli_override` 写的是 `__CLI_SCHEDULER_ENABLED__` 这种独立前缀的键，`.env` 里不会有它。

**附带一条**：`SCHEDULER_ENABLED` 必须在 import `runtime.server` 之前生效 ——
`server.py` 在模块导入时就读了 `RUNTIME.scheduler_enabled` 做启动钩子，
所以 `__main__.py` 里 `--no-schedule` 要先写覆盖再 `importlib.reload(agent.config)` 和 `reload(server)`。

### 9.5 通知出口只有 `notify` / `notify_summary`

🧩 项目约定：通知一律走工具，不直接 `print`。
`print` 只在 `notify.py` 内部做降级输出使用 —— 直接 print 会绕过渠道配置（webhook 地址、投递格式），消息就散落在日志里。

### 9.6 事件模型与归一化层（`runtime/events.py`）

🧩 项目自研，课件未覆盖。是桌面版的事件协议。

**事件模型**：`BaseEvent` 加 9 种事件。公共字段四个 —— `version`（协议版本）、
`run_id`（归属哪次执行）、`timestamp`（秒级浮点）、`type`（判别字段）。

| 事件 | 什么时候发 | 关键字段 |
|---|---|---|
| `run.start` | 收到启动命令、分配好 run_id 后 | `task_name` / `thread_id` / `instruction` |
| `model.text` | `"messages"` 模式里的增量文本 | `text` |
| `tool.start` | 工具开始执行 | `name` / `call_id` / `args` |
| `tool.end` | 工具执行完成 | `name` / `call_id` / `ok` / `duration_ms` / `result_preview` |
| `tool.error` | 工具执行失败 | `name` / `call_id` / `message` |
| `run.progress` | **只定义不发出**（留给 TodoList 进度） | `note` |
| `run.end` | 消费循环跑完 | `output` / `messages` / `duration_ms` |
| `run.cancelled` | 收到取消或连接断开 | `reason` |
| `run.error` | 消费循环抛异常 | `message` / `recoverable` |

`run.cancelled` 与 `run.end` 是**互斥**的 —— 取消时发前者，客户端据此把界面从「运行中」改到「已取消」。

**归一化层** `StreamNormalizer`：把 `agent.stream()` 的 `(模式名, 载荷)` 二元组变成事件列表。

| stream_mode | 载荷 | 归一化成 |
|---|---|---|
| `"messages"` | `(chunk, metadata)` | `model.text` |
| `"custom"` | 中间件发的 dict | 按 `type` 分发成 `tool.*` |

**🧩 `messages` 模式里有三种消息，只留一种**（实测 `agent.stream(stream_mode=["messages"])` 的产出）：

| chunk 类型 | 来自 | 处理 |
|---|---|---|
| `AIMessageChunk` | 模型增量 | → `model.text`（过一遍 `ThinkFilter` 滤掉 `<think>` 块） |
| `AIMessage`（整条） | 中间件插入的提示 | 命中例行提示节点就丢掉，其余文本 → `model.text` |
| `ToolMessage` | 工具结果 | **丢掉** —— 它不是模型输出 |

**坑**
- 🧩 `ToolMessage` 漏掉过滤，模型的一句话会被当成两段发出去。
- 🧩 中间件提示的 `langgraph_node` 实测是 `"ContextInjectionMiddleware.before_model"` ——
  带后缀，所以判定要用**前缀匹配**（`is_hint_node` 用 `startswith`），精确相等会漏。
- 🧩 模型最终答案只以 `AIMessageChunk` 形式出现，**不会**再以整条 `AIMessage` 重复一次；
  整条 `AIMessage` 只来自中间件。
- 🧩 `AIMessageChunk` 的 `content` 可能是内容块列表（多模态），非字符串形态要直接跳过。
- 🧩 一次 run 一个 `StreamNormalizer` 实例 —— `ThinkFilter` 的缓冲跨 chunk 累积，换实例等于清零。

### 9.7 同步流消费器（`runtime/consumer.py`）

🧩 项目自研，桌面版改造计划的阶段 3。

**它解决什么**：`agent.stream()` 是同步的，要跑在**工作线程**里（不能占住事件循环）；
而事件队列是 asyncio 的，跨线程投递要显式处理。消费器把这两件事接起来。

```python
consumer = StreamConsumer(
    agent=agent, run_id=run_id, queue=queue, loop=loop, cancel_event=cancel_event,
)
consumer.run(instruction, ctx, thread_id)   # 阻塞，在工作线程里跑
```

**用的流模式是三种**：`["messages", "values", "custom"]`
- `messages` —— 模型的增量输出（阶段 1 的归一化层处理）
- `custom` —— 中间件发的工具事件（阶段 2 的 EventBridgeMiddleware）
- `values` —— 每步的完整状态，**消费循环结束时用来取最终答案与消息条数**

**🧩 背压是两层的，缺一个内存就会被吃光**：

| 层 | 管什么 | 在哪 |
|---|---|---|
| 队列容量 `maxsize` | 已入队的事件 | `asyncio.Queue` |
| 在途上限 `max_pending` | 已投递但还没入队的事件 | 消费器自己计数 |

在途达到上限时生产者等，等事件循环把队列消化掉、计数降下来。

**🧩 千万别用 `run_coroutine_threadsafe(...).result()` 来等投递完成** ——
实测会**永久阻塞**：那个 future 靠事件循环的回调来 resolve，
事件循环在 resolve 之前关闭（进程退出、连接关闭）时 future 永远不 resolve，
事件其实已经 put 进队列了，生产者却再也回不来。
消费线程卡死的表现是「事件 0.03 秒就发完了，线程 8 秒还不退出」。

**🧩 `run.end` 的 output 用「累积的 model.text」，不是 state 里最后一条 AIMessage 的 content** ——
后者是**原始**内容，推理模型（MiniMax 等）会把 `<think>` 思考块混在里面，又脏又长。
model.text 是归一化层滤过思考块的正文，那才是用户该看到的。

**🧩 累积的正文为空时会回退到 state 的最后一条消息** ——
多数是**模型调用失败**：`on_failure="continue"` 把失败包成一条 AIMessage 写进 state，
但那条消息不经过模型节点，**`messages` 模式看不到它**，
所以既没有 model.text，累积的正文也是空的。回退一次，用户至少能看到「模型调用失败了」。

**🧩 推理模型在工具调用轮次后可能返回空 content** ——
实测 MiniMax-M3 多轮调工具时，那些轮的 `content` 是空串
（`additional_kwargs` 只有 `refusal`，`response_metadata` 里也没有推理内容）。
所以 `run.end` 的 `output` **允许为空**，客户端要能处理「这一轮模型没给出正文」。

**🧩 模型调用失败不会逃出消费循环**：
`ModelRetryMiddleware` 的默认 `on_failure="continue"` 会把重试耗尽后的失败
包成一条 AIMessage 继续跑 —— 图本身是「成功」跑完的。
所以现状是：模型失败 → 消费循环正常结束 → 发 `run.end`，错误信息出现在 `output` 里。
`run.error` 只在**不被任何中间件兜住**的异常时出现。

**异常路径必须闭合**（三种退出各有一条结束事件，客户端才不会一直转圈）：

| 情况 | 发什么 |
|---|---|
| 正常跑完 | `run.end` |
| 收到取消（`cancel_event` 置位） | `run.cancelled`（不是 `run.end`） |
| 不被兜住的异常 | `run.error` |

**取消是协作式的**：置位 `cancel_event` 后，消费循环在**下一次迭代**时检查到并 break ——
正在跑的那一步（模型调用或工具执行）会做完。

### 9.8 Run Manager（`runtime/runs.py`）

🧩 项目自研，桌面版改造计划的阶段 4。职责单一：**run 的生命周期**，不碰 agent、不碰事件格式。

**🧩 run_id 与 thread_id 是两回事，混了记忆隔离就出错**：

| | 含义 | 生命周期 | 生成规则 |
|---|---|---|---|
| `run_id` | 一次执行的标识 | 只在本次执行内有效，结束即注销 | `run-{uuid8}` |
| `thread_id` | 会话标识，记忆的隔离键 | 跨多次执行复用 | 客户端指定；不指定则一次性（`task-{uuid8}`） |

一个会话里可以连续跑多个 run —— 各自独立的 run_id 和取消标志，共享同一份记忆。
这是桌面 Agent 的常态：用户在同一个对话里连问三件事。

**单连接单任务**：每个连接（owner）同时只允许一个活跃 run。
第二次 `start` 抛 `RunBusyError`，由调用方回一条 `run.error` 说明「上一轮还在运行」——
**不静默取消旧的**，因为静默取消会让用户以为自己看到的是新一轮的结果。

**接口**：

| 方法 | 做什么 |
|---|---|
| `start(...)` | 分配 run_id、建 cancel_event、登记、起后台任务 |
| `get(run_id)` | 查活跃的 run |
| `cancel(run_id)` | 置位 cancel_event，返回是否命中 |
| `finish(run_id)` | 注销（幂等） |
| `active()` / `active_for(owner)` | 列活跃的 run |

**🧩 取消是协作式的**：
置位 `cancel_event` 后，消费循环在**下一次迭代**时检查到并 break ——
正在跑的那一步（模型调用或工具执行）**会做完**。
取消信号最终体现为一条 `run.cancelled` 事件（不是 `run.end`），
客户端据此把界面从「运行中」改到「已取消」。

**这跟 `interrupt()`（暂停等输入再恢复）是两件事** —— 后者要设计暂停点并处理检查点恢复。
桌面版的「取消」不需要它。

**坑**
- 🧩 `start()` 里的后台任务是 `loop.create_task(asyncio.to_thread(consumer.run, ...))` ——
  消费器是同步的，要放在工作线程里，不能占住事件循环。
- 🧩 任务跑完会**自动注销**（done callback 调 `finish`），不用手动调。
- 🧩 `RunBusyError` 要由调用方捕获并转成 `run.error` 事件 ——
  Run Manager 自己不发事件，它不碰事件格式。

### 9.9 WebSocket 事件流端点（`runtime/server.py` 的 `/ws`）

🧩 项目自研，桌面版改造计划的阶段 5。

**协议**（客户端 → 服务端，两条命令）：

```json
{"type": "run.start", "instruction": "...", "task_name": "...",
 "thread_id": "可选", "allow_write": false}
{"type": "cancel"}
```

**协议**（服务端 → 客户端）：一条一个 JSON 事件。
`run.start` / `model.text` / `tool.start` / `tool.end` / `tool.error` /
`run.progress` / `run.end` / `run.cancelled` / `run.error`

**🧩 单一发送点**：Starlette 的 `WebSocket.send` **没有锁**，并发发帧会交错。
所以所有要发给客户端的消息（事件的 + 端点自己产生的 `run.start` / `run.error`）
都进同一个队列，由唯一的推送任务发出去 —— 端点自己不直接 `send`。

**🧩 单连接单任务**：同一连接已有活跃 run 时，新的 `run.start` 被拒绝，
回一条 `run.error` 说明「上一轮还在运行」—— 不静默取消旧的。

**🧩 连接断开要清理**：断开时取消这个连接上的所有 run，
否则消费线程会一直跑到 agent 结束。

**错误也走事件**：命令非法、JSON 不合法、缺 `instruction`、取消时没有活跃 run ——
全都回一条 `run.error`，客户端不会一直转圈等。

**坑**
- 🧩 **写权限默认关闭**：`allow_write` 不传就是 False（只读是默认态）。
  `AgentContext` 这个 dataclass 自己的默认值是 `True`，所以端点必须**显式传**，不能省。
- 🧩 **端点内部调 `get_agent()` 拿进程内的 Agent 单例** ——
  测试必须替换 `runtime.server.get_agent`（而不是 `runtime.runner.get_agent`，
  那是名字绑定）。不替换的话测试会**真的调模型 API**：
  慢、不确定、模型可能一直说话，测试还会挂。
- 🧩 测试时还要把 `start_scheduler` / `stop_scheduler` 换成空操作 ——
  调度器会到点触发真实的定时任务。
- 🧩 `RunManager` 挂在 `app.state` 上，每个应用实例一个，测试之间不共享状态。

---

## 十、课件里有、项目还没用的

| 课件章 | 内容 | 项目现状 | 建议 |
|---|---|---|---|
| 06 结构化输出 | `with_structured_output()` 的四种模式（Pydantic / TypedDict / JSON Schema / dataclass） | 只有手写的 `save_json` | 用 `model.with_structured_output(PydanticModel)` 让模型直接返回结构化对象，比「输出 JSON 字符串再解析」可靠得多 |
| 07 章 7.3 | `ToolStrategy` / `ProviderStrategy` / `AutoStrategy` | 没用 `response_format` | 让 Agent 直接返回结构化对象。注意 1.x **必须显式** `ToolStrategy(Schema)` |
| 08 章 2.x | `SummarizationMiddleware` / `HumanInTheLoopMiddleware` / `ModelFallbackMiddleware` | 均未启用 | 见 6.11 |
| 09 章 3 | 长期记忆 store | 只挂了 `InMemoryStore` 没实际用 | `store.put(namespace, key, value)` / `store.search()` |
| 09 章 2.4 | 消息裁剪 / 删除 / 摘要 | 只做了工具结果清理 | 见 7.9 |
| 09 章 4.1 | `dynamic_prompt` | 未实现 | 值得练一次 |
| 10 RAG | 检索增强生成 | requirements 里被注释掉了 | 需要向量库 + 切分，最后再碰 |
| LG 02 章 5.3-5.4 节点超时 / 错误处理 | 当前 `langgraph 1.1.2` 不支持 | 升级到 langgraph ≥ 1.2 后，`RetryPolicy` 才有 `timeout` 与 `error_handler` |
| LG 04 章 8 中断 | 没用 `interrupt()` | 当前 WebSocket 取消是协作式取消；需要暂停并等待用户输入时再引入中断与恢复 |
| LG 04 章 9.1 `langgraph dev` 部署 | 自建 FastAPI | 要用得让模块导出一个图对象（现在导出的是 FastAPI `app`） |
| LG 05 章 11.3 `astream_events` | 没用 | 当前以同步 `agent.stream()` 归一化为项目自己的 WebSocket 事件协议 |
| LG 05 章 12 子图 | 单张图 | 等任务表长出多步骤工作流再拆 |
| LG 05 章 13 设计模式 | 单一 ReAct 循环 | 编排者-工人 / 评估者-优化者要手写 `StateGraph` |

### 10.1 LangSmith 追踪（已接入，默认关）

📘 LangChain 03 章 / LangGraph 04 章 9.1。看每次 invoke 的完整 trace：每一步的输入输出、耗时、token 用量。

**开启** —— 只改 `.env` 三行，**代码零改动**（langchain 自动读环境变量）：

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2-...
LANGSMITH_PROJECT=automation-agent
```

**坑**
- 🧩 **值必须是精确的小写 `true`**。框架的判定是
  `get_env_var("TRACING_V2", default=get_env_var("TRACING", default="")) == "true"`
  —— 精确匹配，写成 `1` 或 `True` 都不生效。
- 🧩 **只看开关、不看 API key**。没填 key 就开着，每次调用刷一条 `401 Unauthorized`，
  但**不影响运行** —— 追踪失败被 langchain 吞掉，只打一条日志。
- 🧩 **`LANGSMITH_TRACING_V2` 优先于 `LANGSMITH_TRACING`**（旧变量仍被认）。
  项目只暴露后者。
- 🧩 **追踪失败 ≠ 调用失败**。「代码跑通」不等于「追踪通了」——
  验证看日志里有没有 `Failed to multipart ingest runs`，再去 LangSmith 网站按项目名找最新 run。
- 🧩 **追踪开着会改变模型的调用路径**（流式 vs 非流式），
  **这会打破用假模型的测试** —— 假消息被加速消耗，不够就 `StopIteration`。
  `tests/conftest.py` 的 autouse fixture 在测试期间关掉追踪：
  实测同一组用例开着 6.5 秒、关掉 0.18 秒。
  判定入口是 `langsmith.utils.tracing_is_enabled()`，
  `langchain_core.tracers.context` 通过模块属性调用它，所以 monkeypatch 属性即可
  —— 不能用环境变量，因为项目的 `.env` 是 `load_dotenv(override=True)`，会盖掉它。
- 🧩 **只能改 `.env`，`export` 不管用**。项目用 `load_dotenv(override=True)`，
  `.env` 里的值盖掉 shell 里的同名环境变量。`set_cli_override()` 也不行 ——
  它写的是 `__CLI_LANGSMITH_TRACING__` 这种独立前缀的键，框架只认 `LANGSMITH_TRACING`。
- 🧩 **隐私**：提示词与模型输出会发到 LangSmith 云端。

---

## 十一、踩坑速查（按症状索引）

| 症状 | 原因 | 修法 | 出处 |
|---|---|---|---|
| `runtime.context` 是 `None`，任务名和写权限丢了 | 把 `context` 写进了 `config` | `agent.invoke(msgs, config, context=ctx)` | 🧩 `runner.py:154` |
| `AttributeError: 'NoneType' object has no attribute ...` | 钩子里直接访问 `runtime.context` | 用 `_ctx(runtime)` | 🧩 `custom.py:48` |
| HTTP 400 `tool call result does not follow tool call` | 拦下工具后没补 `ToolMessage` | 逐个补 `ToolMessage(tool_call_id=tc["id"])` | 🧩 `custom.py:80` |
| `Invalid checkpointer` | 用了 `SqliteSaver.from_conn_string()`（上下文管理器，连接会被关） | 自持 `sqlite3` 连接 | 🧩 `graph.py:354` |
| 测试会话落进 `data/memory.sqlite` 跨次累积 | `build_agent(checkpointer=None)` 不是「不要记忆」是「用默认的」 | 测试显式传 `InMemorySaver()` | 🧩 `graph.py:566` |
| `DeprecationWarning` / 系统提示词改不动 | `system_prompt` 是只读 property | `request.override(system_message=...)` | 🧩 `custom.py:190` |
| 改了提示词但模型还是原样 | `override()` 是**替换**语义 | 把原提示词拼回去 | 🧩 `custom.py:208` |
| 中间件钩子静默不触发 | `middleware` 里传了类不是实例 | 传 `TaskGateMiddleware()` | 🧩 `custom.py:17` |
| `hook_config` 报错 | 无参装饰器写法 | `hook_config` 只用于 `can_jump_to` | 🧩 `CLAUDE.md` 陷阱 6 |
| `--no-schedule` 不生效，`/health` 仍返回 `scheduler: true` | `.env` 的 `SCHEDULER_ENABLED=1` 顶掉了 `os.environ` | 用 `set_cli_override()` | 🧩 `config.py:49` |
| `create_agent(response_format=Schema)` 报错 | 1.x 不再支持直接传 schema | `ToolStrategy(Schema)` / `ProviderStrategy(Schema)` | 📘 07 章 P39 |
| `ValueError: Function must have a docstring if description not provided.` | 工具没写 docstring | 补 docstring | 📘 05 章 |
| 模型选错工具 / 不知道参数是干什么的 | docstring 模糊 / 参数没说明 | 写 Google 风格 docstring 并开 `parse_docstring=True`；工具数收敛到 2-5 个 | 📘 07 章 P21 |
| `Tool 'xxx' failed after 3 attempts with ValueError: ... Please try again.` | 工具抛的异常被 `ToolRetryMiddleware` 重试了 3 次（默认 `retry_on=(Exception,)`） | 模型能自己纠正的错误（路径越界、格式不对）改成**返回说明文字**，别抛异常 | 🧩 `files.py` `_resolve_safe` |
| 单次任务把上下文撑爆 | 工具返回了超大文件 | 200KB 字节上限 + 200 行上限 + `offset`/`limit` 分段读 | 🧩 `files.py` |
| 工具超时/失败 | 没配重试 | `ToolRetryMiddleware`（注意副作用工具的幂等性） | 📘 08 章 3.5 |
| `NotImplementedError`（测试里） | 假模型没实现 `bind_tools` | 用 `GenericFakeChatModel` | 🧩 `CLAUDE.md` 陷阱 7 |
| `GenericFakeChatModel` 报错 | `_generate` 返回了 `AIMessage` | 必须返回 `ChatResult` | 🧩 `CLAUDE.md` 陷阱 8 |
| `RetryPolicy(timeout=...)` 报 TypeError | 当前 `langgraph 1.1.2` 的 `RetryPolicy` 只有重试字段 | 升级到 langgraph ≥ 1.2 | 📘 LG 02 章 5.1 |
| agent 改用文字回答了，不再调工具 | 闸门拦截后补了收尾 AIMessage，改变了条件边判定 | 只补 `ToolMessage`，别补收尾消息 | 🧩 `custom.py:80-86` |
| 不知道 agent 的图长什么样 | —— | `agent.get_graph().draw_mermaid()` | 🧩 实测 |

---

