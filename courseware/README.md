# 课件

本目录存放本项目对照学习用的**尚硅谷 LangChain 1.x 课程讲义**（PDF 原件，共 10 章）。

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

文件名里的数字就是章号 —— `尚硅谷-05-Tools.pdf` 即课件 05 章。

| 章 | 文件 | 内容 | 在本项目里的位置 |
|---|---|---|---|
| 01 | `尚硅谷-01-LangChain概述.pdf` | 课程总览、1.x 相比 0.x 的变化 | 背景概念，代码里没有直接对应 |
| 02 | `尚硅谷-02-模型的创建与调用.pdf` | 模型接入、`init_chat_model` | 主线 1 —— `agent/config.py` 的 `ModelConfig.build()` |
| 03 | `尚硅谷-03-LangSmith的使用.pdf` | LangSmith 追踪 | 本项目未使用 |
| 04 | `尚硅谷-04-Message与提示词模板.pdf` | 消息类型、提示词模板 | 主线 1 —— `HumanMessage` 输入构造 |
| 05 | `尚硅谷-05-Tools.pdf` | `@tool`、工具描述、沙箱、function calling | 主线 2 —— `agent/tools/` |
| 06 | `尚硅谷-06-结构化输出.pdf` | `with_structured_output` 的四种模式 | `agent/tools/files.py` 的 `save_json` 是手写版，可对照改造 |
| 07 | `尚硅谷-07-智能体.pdf` | `create_agent`、Agent 循环、`stream_mode` | 主线 1 —— `agent/graph.py` 的 `build_agent()` |
| 08 | `尚硅谷-08-中间件.pdf` | 中间件机制、钩子、执行顺序 | 主线 3 —— `agent/middleware/custom.py`（本项目精华） |
| 09 | `尚硅谷-09-上下文与记忆.pdf` | 静态运行时上下文、checkpointer、记忆治理 | 主线 4 —— `agent/context.py` |
| 10 | `尚硅谷-10-RAG.pdf` | RAG | 本项目不覆盖 RAG |

第 10 章（RAG）在 `requirements.txt` 里也是注释掉的 —— 这个骨架停在「Agent 怎么做」，
没往「Agent 查什么」走。

## 配套资料

课件同目录的 `../langchain学习/02-资料/` 下还有几份配套文档（conda 用法、Milvus 指南等），
与本项目当前的实现没有直接关系，未收录进仓库。

## 怎么配合源码看

1. 在 `LEARNING.md` 里找到对应主线 —— 那里列了「该读哪几个文件（按顺序）」和「做哪个练习」。
2. 打开本目录对应的 PDF，按小节号对照源码。
3. 源码注释里的标注形如「对应课件 05 章 2.2」，直接对应 PDF 里的章节标题。

`LEARNING.md` 是索引，本目录是原文，两者配合使用。
