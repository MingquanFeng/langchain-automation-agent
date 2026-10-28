
"""
Agent 运行时上下文。

对应课件第 09 章 4.1 节。

主要职责：
1. 描述某一次 Agent 调用的执行环境。
2. 保存任务名称、触发来源、权限等信息。
3. 通过 runtime.context 提供给工具和中间件。

注意：
    Context 不等于 Agent 的消息状态。
    Context 通常不会由 Checkpointer 自动持久化。
"""

from __future__ import annotations

# dataclass：标准库的数据类装饰器。
# 自动生成 __init__、__repr__、__eq__ 等方法。
from dataclasses import dataclass

# Literal：类型字面量。
# 用来把某个字段的取值限制在固定的几个字符串里。
from typing import Literal


# ============================================================
# 1. 定义触发来源类型
# ============================================================

# Trigger 是触发来源的取值集合。
#
# 用 Literal 而不是普通 str 的原因：
# 普通 str 写错了编译期发现不了，
# 比如 trigger="schdule"（拼错）也能跑，
# 只是日志里看不出异常。
#
# 用了 Literal 之后，编辑器和类型检查工具
# 会直接标出不在集合里的取值。
#
# 新增触发方式时，要同步改这里。
Trigger = Literal["schedule", "api", "cli", "lab"]

# 各取值的含义：
#
# schedule：定时调度器触发（runtime/scheduler.py）。
# api：HTTP 接口触发（runtime/server.py）。
# cli：命令行或直接调用触发（runtime/runner.py、runtime/repl.py）。
# lab：学习实验脚本触发（自己写的实验脚本）。


# ============================================================
# 2. Agent 运行时上下文
# ============================================================

@dataclass
class AgentContext:
    """
    单次 Agent 调用的上下文数据结构。

    在 graph.py 中通过：
        context_schema=AgentContext

    告诉 create_agent 使用该上下文结构。

    执行时通过：
        agent.invoke(..., context=AgentContext(...))

    将具体上下文传入 Agent。

    工具中可以通过：
        runtime.context

    访问当前上下文。

    注意：
        thread_id 不放在这里。

        thread_id 是 Checkpointer 使用的会话标识，
        应通过 config["configurable"]["thread_id"] 传入。
    """

    # --------------------------------------------------------
    # 1. 任务名称
    # --------------------------------------------------------

    # 当前任务的人类可读名称。
    # 主要用于日志、通知和任务追踪。
    #
    # 示例：
    # task_name="每日新闻汇总"
    # task_name="文件异常检查"
    #
    # 默认 manual，表示手动任务。
    task_name: str = "manual"

    # --------------------------------------------------------
    # 2. 任务触发来源
    # --------------------------------------------------------

    # 当前任务是从哪儿触发的。
    #
    # 可选值见上面的 Trigger 定义：
    # cli：命令行触发。
    # api：HTTP API 触发。
    # schedule：定时调度器触发。
    # lab：实验脚本触发。
    #
    # 这里用 Trigger 类型做了取值约束，
    # 写错会在编辑时就提示。
    trigger: Trigger = "cli"

    # --------------------------------------------------------
    # 3. 写操作权限
    # --------------------------------------------------------

    # 是否允许当前任务执行写操作。
    #
    # True：允许写操作。
    # False：禁止写操作。
    #
    # 写操作例如：
    # - 修改或删除文件
    # - 发送消息
    # - 修改数据库
    #
    # 重要：
    # 此字段本身不会自动阻止写操作。
    # 必须由相关工具或中间件主动检查。
    #
    # 保留原代码默认值 True。
    # 无人值守场景建议考虑默认 False。
    allow_write: bool = True

    # --------------------------------------------------------
    # 4. 通知渠道
    # --------------------------------------------------------

    # 当前任务的通知渠道标识。
    #
    # 示例：
    # terminal：终端。
    # webhook：HTTP Webhook。
    #
    # 具体支持哪些渠道取决于通知模块。
    #
    # 与 RUNTIME.notify_channel 不同：
    # 这里表示单次任务的配置。
    #
    # 当前代码没有自动继承 RUNTIME 的值。
    notify_channel: str = "terminal"
