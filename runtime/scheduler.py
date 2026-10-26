"""APScheduler 调度器。

LangGraph 1.x 本地没有内置 cron（实测 langgraph_sdk 也未导出 cron），
所以定时触发由 APScheduler 负责。
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from agent.config import RUNTIME
from .runner import run_task
from .tasks import ScheduledTask, all_tasks

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _execute(task: ScheduledTask) -> None:
    """定时任务执行入口。异常不能让它把调度器带崩。"""
    logger.info("定时任务开始：%s", task.name)
    try:
        result = run_task(
            instruction=task.instruction,
            task_name=task.name,
            trigger="schedule",
            allow_write=task.allow_write,
            # 同一任务共用一个 thread，保留跨天记忆
            thread_id=f"cron-{task.name}",
        )
        logger.info("定时任务完成：%s -> %s", task.name, result["output"][:200])
    except Exception:
        logger.exception("定时任务失败：%s", task.name)


def start_scheduler(tasks: list[ScheduledTask] | None = None) -> BackgroundScheduler:
    """启动调度器并注册全部任务。重复调用会先停掉旧的。"""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)

    scheduler = BackgroundScheduler(timezone=RUNTIME.timezone)
    for task in tasks if tasks is not None else all_tasks():
        if not task.enabled:
            continue
        scheduler.add_job(
            _execute,
            trigger=CronTrigger.from_crontab(task.cron, timezone=RUNTIME.timezone),
            args=[task],
            id=task.name,
            replace_existing=True,
            misfire_grace_time=3600,
            max_instances=1,
            coalesce=True,
        )
        logger.info("已注册定时任务：%s (%s)", task.name, task.cron)

    scheduler.start()
    _scheduler = scheduler
    logger.info("调度器已启动，时区 %s", RUNTIME.timezone)
    return scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("调度器已停止")


def list_jobs() -> list[dict]:
    if _scheduler is None:
        return []
    return [
        {
            "id": job.id,
            "next_run": str(job.next_run_time) if job.next_run_time else None,
        }
        for job in _scheduler.get_jobs()
    ]
