"""通知出口工具。

默认组合：webhook 可配置（留空则只输出终端）。发到飞书/钉钉/企业微信
都是同一个 webhook 接口，只是 payload 格式不同，这里按飞书格式组装，
其他平台改 `_format_payload` 即可。
"""

from __future__ import annotations

import json

import httpx
from langchain_core.tools import tool

from ..config import RUNTIME


def _format_payload(text: str, title: str) -> dict:
    """按飞书群机器人格式组装。其他平台在这里换。"""
    return {"msg_type": "text", "content": {"text": f"{title}\n{text}"}}


def _deliver(channel: str, text: str, title: str) -> str:
    """实际投递。webhook 未配置时降级为终端输出。"""
    if channel != "webhook" or not RUNTIME.notify_webhook:
        print(f"\n{'=' * 60}\n[{title}]\n{text}\n{'=' * 60}")
        return "已输出到终端（未配置 webhook）"

    payload = _format_payload(text, title)
    resp = httpx.post(RUNTIME.notify_webhook, json=payload, timeout=15)
    resp.raise_for_status()
    return f"已推送到 webhook（HTTP {resp.status_code}）"


@tool
def notify(text: str, title: str = "Agent 通知") -> str:
    """把一条消息推送给用户。这是 Agent 汇报结果的主要出口。

    Args:
        text: 消息正文，Agent 应该在正文里给出结论而不是过程。
        title: 消息标题，通常是任务名。
    """
    return _deliver(RUNTIME.notify_channel, text, title)


@tool
def notify_summary(items: str, title: str = "每日汇总") -> str:
    """推送一条结构化汇总。items 是已经整理好的多行文本。

    与 notify 的区别是本工具不做额外处理，适合批量清单类内容。
    """
    return _deliver(RUNTIME.notify_channel, items, title)


NOTIFY_TOOLS = [notify, notify_summary]
