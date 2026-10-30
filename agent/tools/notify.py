"""通知出口工具。

默认组合：webhook 可配置（留空则只输出终端）。发到飞书/钉钉/企业微信
都是同一个 webhook 接口，只是 payload 格式不同，这里按飞书格式组装，
其他平台改 `_format_payload` 即可。

对应课件 05 章（courseware/尚硅谷-05-Tools.pdf）：@tool 装饰器定义工具，docstring 即给模型的工具说明。

主要职责：
1. 提供 notify / notify_summary 两个通知出口。
2. 把消息投递到 webhook，未配置时降级为终端输出。
3. 把 payload 组装收在一处，换平台只改一个函数。

两个出口的区别：
- notify：单条消息，Agent 汇报某一步的结果时用。
- notify_summary：结构化汇总，一次推一整份多行清单，
  不做额外加工，适合日报、清单类内容。

项目约定：
通知一律走 notify / notify_summary，不要直接 print。
print 只在本模块内部做降级输出使用。
直接 print 会绕过渠道配置（webhook 地址、投递格式），
消息就散落在日志里，使用方收不到。

注意：
notify / notify_summary 都属于**写工具**，
名字写在 middleware/custom.py 的 TaskGateMiddleware.WRITE_TOOLS 里。
只读模式（AgentContext.allow_write=False）下会被闸门拦下，
所以只读任务要汇报结论，只能写在回复文本里。
"""

from __future__ import annotations

# httpx：HTTP 客户端。
# 用于向 webhook 地址 POST 消息体。
#
# 说明：
# 这里没有 import json —— httpx 的 json= 参数会自己把 dict
# 序列化成请求体，用不到标准库的 json 模块。
import httpx

# @tool：LangChain 的工具装饰器。
# 装饰后函数会被包装成模型可调用的工具，
# docstring 就是模型看到的工具说明。
from langchain_core.tools import tool

# RUNTIME：应用运行配置。
#
# notify_webhook：webhook 地址，默认为空字符串。
# 配置形如 NOTIFY_WEBHOOK=https://<你的群机器人地址>
#
# notify_channel：默认通知渠道，
# terminal 表示只输出终端，webhook 表示走 HTTP 投递。
from ..config import RUNTIME


# ============================================================
# 1. payload 组装
# ============================================================

def _format_payload(text: str, title: str) -> dict:
    """按飞书群机器人格式组装。其他平台在这里换。"""

    # 飞书群机器人的文本消息格式：
    # msg_type 固定为 text，
    # content.text 里放正文。
    #
    # 标题和正文拼在一起发，
    # 是因为各平台的文本消息都只有这一个可写字段。
    return {"msg_type": "text", "content": {"text": f"{title}\n{text}"}}


# ============================================================
# 2. 实际投递
# ============================================================

def _deliver(channel: str, text: str, title: str) -> str:
    """实际投递。webhook 未配置时降级为终端输出。"""

    # 降级条件有两个，缺一不可：
    #
    # 1. channel 不是 webhook
    #    —— 配置成了 terminal 就是只想看终端输出。
    #
    # 2. notify_webhook 为空
    #    —— 环境变量没填 NOTIFY_WEBHOOK。
    #
    # 两个条件都满足时走终端分支，
    # 好处是没配 webhook 也能把整条流程跑通，
    # 不用为了调试专门去申请一个机器人地址。
    if channel != "webhook" or not RUNTIME.notify_webhook:

        # 用等号框住内容，是为了在终端日志里一眼分辨出通知块，
        # 免得和模型输出的日志混在一起。
        print(f"\n{'=' * 60}\n[{title}]\n{text}\n{'=' * 60}")
        return "已输出到终端（未配置 webhook）"

    # 组装请求体。
    payload = _format_payload(text, title)

    # 向 webhook POST。
    #
    # timeout=15：
    # 通知是收尾动作，不值得为它长时间挂起。
    #
    # 注意：
    # 投递失败会抛异常，不会静默吞掉。
    # 上层（Agent 工具调用）能看到失败，
    # 配合 graph.py 里的工具重试中间件可以重试。
    resp = httpx.post(RUNTIME.notify_webhook, json=payload, timeout=15)

    # 非 2xx 直接抛异常，
    # 否则 webhook 返回 4xx/5xx 时会被当成投递成功。
    resp.raise_for_status()
    return f"已推送到 webhook（HTTP {resp.status_code}）"


# ============================================================
# 3. 通知工具
# ============================================================

@tool
def notify(text: str, title: str = "Agent 通知") -> str:
    """把一条消息推送给用户。这是 Agent 汇报结果的主要出口。

    Args:
        text: 消息正文，Agent 应该在正文里给出结论而不是过程。
        title: 消息标题，通常是任务名。
    """

    # 渠道由 RUNTIME.notify_channel 决定，
    # 不从参数传入：
    # 同一套工具在不同任务里应该走同一条投递路径。
    #
    # 提示词（graph.py 的 SYSTEM_PROMPT）要求
    # 每一步做完就推一条，不要攒到最后。
    return _deliver(RUNTIME.notify_channel, text, title)


@tool
def notify_summary(items: str, title: str = "每日汇总") -> str:
    """推送一条结构化汇总。items 是已经整理好的多行文本。

    与 notify 的区别是本工具不做额外处理，适合批量清单类内容。
    """

    # 同样的投递逻辑，同样的渠道。
    #
    # 单独分一个工具，是为了让模型在提示词里有一个明确的
    # 「汇总」入口：调它等于声明"这是一份完整结果"，
    # 而不是过程中的又一条进展汇报。
    return _deliver(RUNTIME.notify_channel, items, title)


# ============================================================
# 4. 导出通知工具
# ============================================================

# NOTIFY_TOOLS 会被 tools/__init__.py 汇总进 ALL_TOOLS，
# 最终交给 create_agent 作为模型可调用的工具。
#
# 注意：
# 往这里加新工具时，要同步更新
# middleware/custom.py 里 TaskGateMiddleware.WRITE_TOOLS。
# 通知类工具都有外部副作用，
# 只读模式下必须被拦住，否则会往用户那里推不该推的消息。
NOTIFY_TOOLS = [notify, notify_summary]
