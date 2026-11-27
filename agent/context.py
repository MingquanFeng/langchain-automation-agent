
"""
Agent 运行时上下文。

对应课件第 09 章 4.1 节（courseware/langChain/尚硅谷-09-上下文与记忆.pdf）。

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
from dataclasses import dataclass, field
import threading

# Literal：类型字面量。
# 用来把某个字段的取值限制在固定的几个字符串里。
from typing import Any, Callable, Literal


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


class TaskCancelledError(RuntimeError):
    """协作式取消信号；不得作为普通模型或工具失败重试。"""


def check_cancelled(ctx) -> None:
    """在模型和工具的调用边界检查取消；无上下文的旧调用保持兼容。"""
    event = getattr(ctx, "cancel_event", None)
    if event is not None and event.is_set():
        raise TaskCancelledError("任务已请求取消")


@dataclass
class RepairSession:
    """一次失败检查修复闭环的短生命周期限制。"""

    argv: list[str]
    cwd: str
    timeout_seconds: int
    output_summary: str
    source_run_id: str = ""
    source_call_id: str = ""
    attempts: int = 0
    edited_since_check: bool = False
    edit_pending: bool = False
    # 空字符串表示修复仍可继续。终态必须来自真实工具回执，不能由模型文本推断。
    terminal_status: str = ""
    terminal_message: str = ""

    def close(self, status: str, message: str) -> None:
        """写入第一个终态，避免后续模型调用覆盖真实的修复结论。"""
        if not self.terminal_status:
            self.terminal_status = status
            self.terminal_message = message

# 各取值的含义：
#
# schedule：定时调度器触发（runtime/scheduler.py）。
# api：HTTP 层触发（runtime/server.py）。目前没有调用方，
#      桌面版的 WebSocket 端点会用这个值。
# cli：命令行或直接调用触发（runtime/runner.py）。
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

    # WebSocket 交互任务提供的单次工具授权回调。
    #
    # TaskGateMiddleware 用它区分「需要向用户请求确认」与「没有授权入口」。
    # EventBridgeMiddleware 在工具真正执行前调用它；回调会阻塞当前工作线程，
    # 直到用户响应、任务取消或请求超时。普通 Python / 定时任务不传此字段。
    approval_handler: Callable[[str, str, dict[str, Any]], bool] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

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

    # 由执行入口注入，与审批等待、流消费器共用；不写入会话检查点。
    cancel_event: threading.Event | None = field(default=None, repr=False, compare=False)

    # 本次执行的短生命周期标识。命令工具用它把审计记录关联到运行历史；
    # 不属于会话状态，不能写入 Checkpointer。
    run_id: str = field(default="", repr=False, compare=False)

    # 仅由桌面端“修复此检查”入口注入，不进入会话检查点。
    repair_session: RepairSession | None = field(default=None, repr=False, compare=False)
    command_results: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False, compare=False)
    # 明确要求单次命令时，校验失败、审批拒绝也消耗尝试次数。
    # 锁覆盖并行工具节点，避免两个调用同时通过检查；每轮由中间件重置。
    single_command: bool = field(default=False, repr=False, compare=False)
    command_attempted: bool = field(default=False, repr=False, compare=False)
    command_attempt_lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)
    # edit_file 返回人类可读文本；这个映射保存实际文件操作终态，防止中间件把
    # “文件已变化，未写入”误判为一次成功编辑。
    edit_results: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    # 审批通道回填批准、拒绝、超时或取消，受控修复据此决定是否关闭。
    approval_results: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    # 审批等待中的工具调用。取消可在图提交 ToolMessage 前中断流消费器；
    # 此字段让消费器在关闭流后补偿持久化对应结果，避免污染会话协议。
    pending_approval_tool_call: dict[str, str] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
