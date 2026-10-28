
"""
定时任务定义。

每个任务是一个 (cron 表达式, 指令, 名称) 三元组。
改这个文件就能调整 Agent 每天自动做什么。

对应课件：
课件未覆盖定时任务，需要自行实现。

主要职责：
1. 定义定时任务的数据结构 ScheduledTask。
2. 保存默认任务表 TASKS。
3. 提供运行时增删查接口。

注意：
TASKS 是模块级内存字典，不落盘。
进程重启后通过 POST /tasks 动态加的任务全部消失，
只有改代码里的 TASKS 列表才是持久定义。
"""

from __future__ import annotations

# dataclass：标准库的数据类装饰器。
from dataclasses import dataclass


# ============================================================
# 1. 定时任务数据结构
# ============================================================

@dataclass(frozen=True)
class ScheduledTask:
    """
    一个定时任务的完整定义。

    @dataclass：
        自动生成 __init__、__repr__、__eq__。

    frozen=True：
        禁止创建后再改字段。
        任务一旦注册进调度器就是不可变的，
        要改就换一个同名任务重新注册。
    """

    # --------------------------------------------------------
    # 1.1 任务名称
    # --------------------------------------------------------

    # 任务的人类可读名称。
    #
    # 同时是调度器里 job 的 id，
    # 也是 thread_id 的一部分（cron-{name}），
    # 所以改名等于换一个全新的会话，之前的记忆不会带过来。
    name: str

    # --------------------------------------------------------
    # 1.2 cron 表达式
    # --------------------------------------------------------

    # 标准 5 段 cron 表达式：分 时 日 月 周。
    #
    # 示例：
    # "0 9 * * *"    每天 9:00
    # "0 18 * * *"   每天 18:00
    # "*/30 * * * *" 每 30 分钟
    #
    # 注意：
    # 按 RUNTIME.timezone（SCHEDULER_TIMEZONE）解释，
    # 不是服务器本地时区。
    cron: str

    # --------------------------------------------------------
    # 1.3 任务指令
    # --------------------------------------------------------

    # 给 Agent 的自然语言指令。
    # 和 HTTP /run 的 instruction 是同一个东西。
    instruction: str

    # --------------------------------------------------------
    # 1.4 写权限
    # --------------------------------------------------------

    # 是否允许这个任务执行写操作。
    #
    # 默认 False，即默认只读。
    # 需要 notify 推送通知时必须显式设成 True，
    # 因为 notify 属于写工具。
    allow_write: bool = False

    # --------------------------------------------------------
    # 1.5 启用开关
    # --------------------------------------------------------

    # 是否启用这个任务。
    #
    # False 时任务定义仍然保留在表里，
    # 但 start_scheduler() 不会把它注册进调度器。
    # 适合临时停掉某个任务而不删掉它。
    enabled: bool = True


# 默认任务表。cron 表达式按 agent/config.py 的 SCHEDULER_TIMEZONE 解释。
#
# 注意：
# 这份列表是「持久定义」的唯一来源，
# 改这里 = 改 Agent 每天自动做什么。
TASKS: list[ScheduledTask] = [
    ScheduledTask(
        name="每日待办汇总",
        cron="0 9 * * *",
        instruction=(
            "读取数据目录下的 todo.md，整理出今天要做的 3 件最优先的事，"
            "用 notify 推送给我。每件事一行，不要展开。"
        ),
        # notify 属于写工具，要推送就必须显式开写权限
        allow_write=True,
    ),
    ScheduledTask(
        name="数据目录巡检",
        cron="0 18 * * *",
        instruction=(
            "列出数据目录下的所有文件，把超过 7 天没修改的文件列出来，"
            "用 notify 汇总推送。"
        ),
        # 注意：
        # 这里 allow_write=False，但指令里又要 notify 推送。
        # notify 属于写工具，会被 TaskGateMiddleware 拦掉，
        # 这是个已知的不一致，需要推送时得改成 True。
        allow_write=False,
    ),
]

# 运行时可动态添加
#
# 这是一个普通的模块级内存字典，
# 键是任务名，值是任务对象。
#
# 注意：
# 不落盘。进程重启后这里的动态任务就没了，
# 剩下的只有上面 TASKS 列表里的内容
# （下面这行初始化会把 TASKS 灌进 _registry）。
_registry: dict[str, ScheduledTask] = {t.name: t for t in TASKS}


# ============================================================
# 3. 任务表操作接口
# ============================================================

def all_tasks() -> list[ScheduledTask]:
    """
    列出任务表里的全部任务。

    返回：
        ScheduledTask 列表。

    注意：
    返回的是新列表，
    但里面的 ScheduledTask 对象是 frozen 的，
    改不动，外部只能读。
    """
    return list(_registry.values())


def add_task(task: ScheduledTask) -> None:
    """
    向任务表添加（或覆盖）一个任务。

    参数：
        task：要添加的任务对象。

    注意：
    同名任务会被直接覆盖，不会报错。
    覆盖后需要重新 start_scheduler() 才会真正生效
    （POST /tasks 接口已经帮你做了这一步）。

    注意：
    加进内存字典不等于注册进调度器。
    只调 add_task() 的话，调度器里并没有这个 job。
    """
    _registry[task.name] = task


def remove_task(name: str) -> bool:
    """
    从任务表删除一个任务。

    参数：
        name：任务名。

    返回：
        True 表示确实删掉了；
        False 表示表里本来就没这个名字。

    注意：
    同样只改内存字典，
    需要重新 start_scheduler() 才会在调度器里消失。
    """
    # pop 的第二个参数是默认值。
    # 任务不存在时返回 None，
    # 因此 `is not None` 正好等价于「删成功了」。
    return _registry.pop(name, None) is not None
