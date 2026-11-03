# 课件

本目录存放本项目对照学习用的两套讲义：

| 目录 | 内容 | 格式 |
|---|---|---|
| `langChain/` | **尚硅谷 LangChain 1.x 课程讲义**（10 章） | PDF 原件 |
| `langGraph/` | **LangGraph 专项课件**（6 篇，章节号 1-13 连续） | Markdown + 图片 |

源码注释里的「对应课件 XX 章 X.X 节」标注指向的就是这里 —— 改代码前先翻对应章节，
保持实现与课件 API 一致。只有当课件 API 与 langchain 1.2 实际行为冲突时，才按实际行为写，
并在注释里说明差异。

## 来源与使用说明

| 项 | 说明 |
|---|---|
| 来源 | 尚硅谷 LangChain 1.x 系列课程讲义 |
| 原始位置 | `../langchain学习/01-课件/`（本目录是它的副本） |
| 版权 | 课件内容版权归原作者所有，此处仅作个人学习用途的参考留存 |
| 格式 | PDF 原件，未做任何修改或转换 |

课件是幻灯片版式的讲义，含大量示意图与逐步标注的代码，**保持 PDF 原样比转成 Markdown 更适合对照阅读**，
所以这里不做格式转换。

## 章节索引

文件名里的数字就是章号 —— `langChain/尚硅谷-05-Tools.pdf` 即课件 05 章。

| 章 | 文件 | 内容 | 在本项目里的位置 |
|---|---|---|---|
| 01 | `langChain/尚硅谷-01-LangChain概述.pdf` | 课程总览、1.x 的架构与生态定位 | 背景概念，代码里没有直接对应 |
| 02 | `langChain/尚硅谷-02-模型的创建与调用.pdf` | 模型接入、`init_chat_model` | 主线 1 —— `agent/config.py` 的 `ModelConfig.build()` |
| 03 | `langChain/尚硅谷-03-LangSmith的使用.pdf` | LangSmith 追踪 | 本项目未使用 |
| 04 | `langChain/尚硅谷-04-Message与提示词模板.pdf` | 消息类型、提示词模板 | 主线 1 —— `HumanMessage` 输入构造 |
| 05 | `langChain/尚硅谷-05-Tools.pdf` | `@tool`、工具描述、沙箱、function calling | 主线 2 —— `agent/tools/` |
| 06 | `langChain/尚硅谷-06-结构化输出.pdf` | `with_structured_output` 的四种模式 | `agent/tools/files.py` 的 `save_json` 是手写版，可对照改造 |
| 07 | `langChain/尚硅谷-07-智能体.pdf` | `create_agent`、Agent 循环、`stream_mode` | 主线 1 —— `agent/graph.py` 的 `build_agent()` |
| 08 | `langChain/尚硅谷-08-中间件.pdf` | 中间件机制、钩子、执行顺序 | 主线 3 —— `agent/middleware/custom.py`（本项目精华） |
| 09 | `langChain/尚硅谷-09-上下文与记忆.pdf` | 静态运行时上下文、checkpointer、记忆治理 | 主线 4 —— `agent/context.py` |
| 10 | `langChain/尚硅谷-10-RAG.pdf` | RAG | 本项目不覆盖 RAG |

第 10 章（RAG）在 `langChain/requirements.txt` 里也是注释掉的 —— 这个骨架停在「Agent 怎么做」，
没往「Agent 查什么」走。

## LangGraph 专项课件（`langGraph/`）

LangChain 课件讲「怎么用 `create_agent`」，这套讲「`create_agent` 底下跑的是什么」。
本项目每个 Agent 都是一张 **LangGraph 状态图**，只是图是 `create_agent` 替你画的。

| 文件 | 章节 | 内容 | 在本项目里的位置 |
|---|---|---|---|
| `00-环境配置.md` | — | conda 环境、Jupyter、依赖安装 | **跳过** —— 项目已有 `langchain1.2` 环境，模型走 `.env` 的 `LLM_API_KEY` |
| `01-LangGraph基础入门.md` | 1-3 | StateGraph、State 与 reducer、`AgentState`、Graph vs Functional API | 概念层：项目没手写 `StateGraph`，但 `AgentState`/reducer 在中间件里天天用 |
| `02-LangGraph控制流与节点执行.md` | 4-5 | 顺序/分支/循环、条件边、节点容错与重试、超时、缓存 | 对应 `create_agent` 生成的 `model → tools` 条件边循环 |
| `03-LangGraph持久化与记忆管理.md` | 6-7 | checkpointer、持久化模式、`thread_id`、store、运行时上下文 | **重合度最高的一篇** —— 6.2/7.1/7.3 正好是项目的 `_build_checkpointer()` / store / `AgentContext` |
| `04-LangGraph中断与工具与部署.md` | 8-10 | `interrupt()` 中断、静态断点、`langgraph dev` 部署、`ToolNode` | 项目未用；04 的中断正是 `DESKTOP_AGENT_PLAN.md` 第一步「取消」要的东西 |
| `05-LangGraph高级特性.md` | 11-13 | `stream`/`astream`、`astream_events`、子图、运行图设计模式 | 流式部分项目在用（`stream_mode`）；子图与设计模式未碰 |

配套的 `langgraph-runtime-viz/index.html` 是「LangGraph Runtime 运行图」的离线可视化页面。

**两处不要照着课件做**：

- **00 章让你建 `langgraph` conda 环境 + 配 DeepSeek key** —— 别建。项目用 `langchain1.2` 环境，
  模型走 `.env` 的 `LLM_API_KEY`（默认 `deepseek`）。
- **03 章附录 A 教部署 PostgreSQL** —— 项目用 SQLite（`data/memory.sqlite`），换过去是给自己挖坑。

**抄代码注意**：课件手写 `StateGraph`、项目用 `create_agent` 封装，概念相通但 API 不同。
别抄 `create_react_agent` —— 那会绕开项目现有的中间件栈。

## 配套资料

课件同目录的 `../langchain学习/02-资料/` 下还有几份配套文档（conda 用法、Milvus 指南等），
与本项目当前的实现没有直接关系，未收录进仓库。

## 入库策略

课件是学习资料，要能离线读，所以**原件一律进版本库**，不做外链：

| 内容 | 处理 |
|---|---|
| 10 章 PDF 原件 | 进版本库（约 78MB，仓库已如此） |
| `langGraph/` 6 篇 Markdown | 进版本库 |
| `langGraph/images/`（128 张，约 9.5MB） | 进版本库 —— Markdown 里是相对路径引用，不入库就全成死链 |
| `langGraph/langgraph-runtime-viz/` | 进版本库（单个 HTML，约 47KB） |

`images/` 里有 24 张**没有被任何 Markdown 引用**的图（约 1.5MB）。仍然入库：
日后改课件时可能引用到，逐个列进 `.gitignore` 是噪音，不如统一收着。

## 怎么配合源码看

1. 在 `LEARNING.md` 里找到对应主线 —— 那里列了「该读哪几个文件（按顺序）」和这个知识点的关键认知。
2. 打开本目录对应的课件（LangChain 是 PDF、LangGraph 是 Markdown），按小节号对照源码。
3. 源码注释里的标注形如「对应课件 05 章 2.2」，直接对应课件里的章节标题。

`LEARNING.md` 是索引，本目录是原文，两者配合使用。
