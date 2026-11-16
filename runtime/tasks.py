
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

默认任务由 TASKS 提供；服务首次启动时写入运行时数据库。之后动态增删改会持久化，
数据库中的任务注册表在后续启动时作为实际配置来源。
"""

from __future__ import annotations

# dataclass：标准库的数据类装饰器。
from dataclasses import dataclass
from typing import Any


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


# 默认任务表。首次创建运行时数据库时会将这些默认项写入数据库。
# cron 表达式按 agent/config.py 的 SCHEDULER_TIMEZONE 解释。
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
        # notify 属于写工具，要推送就必须显式开写权限。
        #
        # 这里原来写的是 False —— 指令要 notify 推送，
        # 但只读模式下 notify 会被 TaskGateMiddleware 拦掉，
        # 定时任务跑完什么也推不出去（结果只在 agent 的回复文本里，
        # 而定时任务没有接收回复文本的地方）。
        # 指令的意图明确是要推送，所以这里开成 True。
        allow_write=True,
    ),
]

# 热路径使用内存注册表；服务启动时从 SQLite 恢复，增删改时同步持久化。
_registry: dict[str, ScheduledTask] = {t.name: t for t in TASKS}
_store: Any | None = None


def configure_persistence(store: Any) -> None:
    """从运行时数据库恢复任务；数据库首次创建时种入代码默认值。"""
    global _registry, _store
    defaults = [
        {
            "name": task.name,
            "cron": task.cron,
            "instruction": task.instruction,
            "allow_write": task.allow_write,
            "enabled": task.enabled,
        }
        for task in TASKS
    ]
    rows = store.load_or_seed_tasks(defaults)
    _registry = {
        row["name"]: ScheduledTask(
            name=row["name"],
            cron=row["cron"],
            instruction=row["instruction"],
            allow_write=row["allow_write"],
            enabled=row["enabled"],
        )
        for row in rows
    }
    _store = store


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

    持久化完成后更新内存注册表；HTTP 接口会重建调度器使任务立即生效。
    """
    if _store is not None:
        _store.save_task({
            "name": task.name,
            "cron": task.cron,
            "instruction": task.instruction,
            "allow_write": task.allow_write,
            "enabled": task.enabled,
        })
    _registry[task.name] = task


def remove_task(name: str) -> bool:
    """
    从任务表删除一个任务。

    参数：
        name：任务名。

    返回：
        True 表示确实删掉了；
        False 表示表里本来就没这个名字。

    数据库和内存注册表都会删除该任务；HTTP 接口会重建调度器。
    """
    # pop 的第二个参数是默认值。
    # 任务不存在时返回 None，
    # 因此 `is not None` 正好等价于「删成功了」。
    existed = name in _registry
    if _store is not None:
        _store.delete_task(name)
    _registry.pop(name, None)
    return existed
