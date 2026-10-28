"""运行时上下文。Agent 每次调用的静态配置载体。

对应课件 09 章 4.1 节：context 通过 runtime.context 在中间件和工具中访问。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# 触发来源。schedule=定时调度，api=HTTP 接口，cli=命令行/直接调用，lab=实验脚本
Trigger = Literal["schedule", "api", "cli", "lab"]


@dataclass
class AgentContext:
    """每次 invoke 传入的静态运行时配置。

    thread_id 放在 config 而非这里 —— 它属于会话标识而非运行时配置。
    """

    # 本次任务的人类可读名称，出现在日志与通知里
    task_name: str = "manual"
    # 触发来源：schedule / api / cli / lab
    trigger: Trigger = "cli"
    # 是否允许写操作（发消息、改文件、删数据）。默认关闭，写权限需显式开启
    allow_write: bool = False
    # 通知出口标识，用于日志区分
    notify_channel: str = "terminal"
