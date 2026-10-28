
"""
APScheduler 调度器。

LangGraph 1.x 本地没有内置 cron（实测 langgraph_sdk 也未导出 cron），
所以定时触发由 APScheduler 负责。

对应课件：
课件未覆盖定时任务，需要自行实现。

主要职责：
1. 把 cron 表达式注册成后台定时任务。
2. 到点后调用 runner.run_task() 执行 Agent 任务。
3. 提供 job 列表查询和调度器启停接口。

注意：
LangGraph 1.x 本地没有内置 cron（实测 langgraph_sdk 也未导出 cron），
cron 是 LangGraph Platform（云端）才有的能力。
本地想定时触发只能自己接调度器，这里选的是 APScheduler。
"""

from __future__ import annotations

# logging：标准库的日志模块。
import logging

# BackgroundScheduler：APScheduler 的后台调度器。
# 线程池在后台跑任务，不阻塞主线程（HTTP 服务还能正常响应）。
from apscheduler.schedulers.background import BackgroundScheduler

# CronTrigger：按 cron 表达式触发。
from apscheduler.triggers.cron import CronTrigger

# RUNTIME：应用运行配置，这里取调度器时区。
from agent.config import RUNTIME

# run_task：统一的 Agent 调用入口，见 runner.py。
from .runner import run_task

# ScheduledTask / all_tasks：定时任务的数据结构和任务表。
from .tasks import ScheduledTask, all_tasks

# 模块级 logger，名字会自动带上 "runtime.scheduler"。
logger = logging.getLogger(__name__)

# 进程内唯一的调度器实例。
# None 表示当前没有启动。
_scheduler: BackgroundScheduler | None = None


# ============================================================
# 1. 定时任务执行入口
# ============================================================

def _execute(task: ScheduledTask) -> None:
    """定时任务执行入口。异常不能让它把调度器带崩。

    为什么这里必须吞掉异常：
    APScheduler 里 job 抛异常只会被记进日志，
    不会自动停掉调度器，但会把这次执行标记为失败。
    更糟的是没有兜底时，异常会沿着调度线程往上抛，
    表现为「某天之后所有定时任务都不再执行」。
    所以整个执行过程包在 try 里，
    失败只记日志，不影响后续任务。

    注意：
    这里用的是 run_task() 而不是流式接口，
    所以定时任务拿不到增量输出，
    只有跑完之后的一条最终结果。
    """
    logger.info("定时任务开始：%s", task.name)
    try:
        result = run_task(
            instruction=task.instruction,
            task_name=task.name,
            trigger="schedule",
            allow_write=task.allow_write,
            # 同一任务共用一个 thread，保留跨天记忆
            #
            # 固定用 cron-{任务名}，
            # 这样「每日待办汇总」每天跑都知道自己昨天推过什么。
            thread_id=f"cron-{task.name}",
        )
        logger.info("定时任务完成：%s -> %s", task.name, result["output"][:200])
    except Exception:
        # logger.exception 会连同堆栈一起打出来，方便定位。
        logger.exception("定时任务失败：%s", task.name)


# ============================================================
# 2. 启动调度器
# ============================================================

def start_scheduler(tasks: list[ScheduledTask] | None = None) -> BackgroundScheduler:
    """启动调度器并注册全部任务。重复调用会先停掉旧的。

    参数：
        tasks：
            要注册的任务列表。
            不传则取 tasks.py 里的默认任务表。

    返回：
        启动后的 BackgroundScheduler 实例。

    注意：
    重复调用不会叠加注册，
    因为每次都先 shutdown 掉旧的实例再新建一个。
    所以 POST /tasks、DELETE /tasks/{name} 之后
    可以直接重新 start_scheduler() 来让改动生效。

    注意：
    CronTrigger.from_crontab() 用的是标准 5 段 cron
    （分 时 日 月 周），不是 Quartz 的 6 段格式。
    """
    global _scheduler

    # 已有一个就先停掉，避免同一个任务被注册两遍。
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)

    # 时区来自 RUNTIME.timezone（环境变量 SCHEDULER_TIMEZONE）。
    #
    # 这里必须显式指定，
    # 否则 APScheduler 会用机器本地时区，
    # 服务器时区一变，cron 的触发时刻就跟着漂。
    scheduler = BackgroundScheduler(timezone=RUNTIME.timezone)

    # tasks 为 None 时取默认任务表，
    # 否则用调用方传进来的列表。
    for task in tasks if tasks is not None else all_tasks():

        # enabled=False 的任务跳过，不注册。
        # 这样可以保留任务定义但临时停掉它。
        if not task.enabled:
            continue

        scheduler.add_job(
            # 所有任务共用 _execute 这一个执行函数，
            # 靠 args 区分具体是哪个任务。
            _execute,
            trigger=CronTrigger.from_crontab(task.cron, timezone=RUNTIME.timezone),
            args=[task],

            # job 的 id 就是任务名，方便查日志和 /jobs 接口。
            id=task.name,

            # 已存在同名 id 时替换掉旧的，
            # 避免重复注册同一个任务。
            replace_existing=True,

            # 错过触发时间后，1 小时内仍然补跑一次。
            #
            # 服务停机重启、机器休眠之后，
            # APScheduler 不会把错过的任务补齐，
            # 这个参数让「错过一点」的任务还有机会执行。
            misfire_grace_time=3600,

            # 同一个任务同时最多跑一个实例。
            #
            # 防止上一次的执行还没结束，
            # 下一次又起一个 Agent 跑同一个任务。
            max_instances=1,

            # 错过的多次触发合并成一次。
            # 配合上面的 misfire_grace_time 使用。
            coalesce=True,
        )
        logger.info("已注册定时任务：%s (%s)", task.name, task.cron)

    # start() 之后调度器才真正在后台线程里跑。
    scheduler.start()

    # 保存实例，供 stop_scheduler() 和 list_jobs() 使用。
    _scheduler = scheduler
    logger.info("调度器已启动，时区 %s", RUNTIME.timezone)
    return scheduler


# ============================================================
# 3. 停止调度器
# ============================================================

def stop_scheduler() -> None:
    """
    停止调度器并清空实例引用。

    注意：
    shutdown(wait=False) 表示不等待正在跑的任务结束。
    服务关闭时这样调用可以尽快退出，
    代价是正在执行的那个任务会被中途掐断。
    """
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("调度器已停止")


# ============================================================
# 4. 查询已注册的 job
# ============================================================

def list_jobs() -> list[dict]:
    """
    列出调度器已注册的 job 和下次运行时间。

    返回：
        每个元素含 id 和 next_run 两个字段。
        调度器没启动时返回空列表。

    注意：
    这里查的是「调度器实际注册了什么」，
    和 /tasks 接口查的「任务表里定义了什么」是两个不同的视角。
    两者对不上，说明调度器还没重新启动。
    """
    if _scheduler is None:
        return []
    return [
        {
            "id": job.id,
            # next_run_time 为 None 表示这个 job 没有下次运行计划
            # （比如已经被移除或调度器刚启动还没算出来）。
            "next_run": str(job.next_run_time) if job.next_run_time else None,
        }
        for job in _scheduler.get_jobs()
    ]
