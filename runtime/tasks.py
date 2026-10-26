"""定时任务定义。

每个任务是一个 (cron 表达式, 指令, 名称) 三元组。
改这个文件就能调整 Agent 每天自动做什么。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduledTask:
    name: str
    cron: str
    instruction: str
    allow_write: bool = True
    enabled: bool = True


# 默认任务表。cron 表达式按 agent/config.py 的 SCHEDULER_TIMEZONE 解释。
TASKS: list[ScheduledTask] = [
    ScheduledTask(
        name="每日待办汇总",
        cron="0 9 * * *",
        instruction=(
            "读取数据目录下的 todo.md，整理出今天要做的 3 件最优先的事，"
            "用 notify 推送给我。每件事一行，不要展开。"
        ),
    ),
    ScheduledTask(
        name="数据目录巡检",
        cron="0 18 * * *",
        instruction=(
            "列出数据目录下的所有文件，把超过 7 天没修改的文件列出来，"
            "用 notify 汇总推送。"
        ),
        allow_write=False,
    ),
]

# 运行时可动态添加
_registry: dict[str, ScheduledTask] = {t.name: t for t in TASKS}


def all_tasks() -> list[ScheduledTask]:
    return list(_registry.values())


def add_task(task: ScheduledTask) -> None:
    _registry[task.name] = task


def remove_task(name: str) -> bool:
    return _registry.pop(name, None) is not None
