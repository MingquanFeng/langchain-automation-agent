
"""
运行时外壳：调度器、HTTP 服务、交互式终端、任务定义。

对应课件：
课件未覆盖 runtime 层，需要自行实现。

主要职责：
1. 对外暴露 Agent 调用入口（get_agent、run_task）。
2. 对外暴露定时任务表的增删查接口。
3. 汇总 runtime 层各子模块的公开接口。

注意：
本文件只做导入转发，不含任何业务逻辑。
具体实现分散在 runner.py、tasks.py、scheduler.py、server.py、repl.py。
"""

# ============================================================
# 1. 导入 Agent 调用入口
# ============================================================

# . 表示当前 Python 包。
#
# 从 runner.py 中导入：
# get_agent：获取进程内唯一的 Agent 单例。
# run_task：执行一次任务的统一入口。
from .runner import get_agent, run_task

# ============================================================
# 2. 导入定时任务表接口
# ============================================================

# ScheduledTask：定时任务的数据结构。
# add_task / remove_task / all_tasks：任务表的增删查。
#
# 注意：
# 这里导出的任务表是内存字典，不是持久化存储，
# 详见 tasks.py 的说明。
from .tasks import ScheduledTask, add_task, all_tasks, remove_task

# ============================================================
# 3. 定义通配符导出列表
# ============================================================

# __all__ 主要控制：
#
# from runtime import *
#
# 这种写法会导入哪些名称。
#
# 它不会禁止其他代码显式导入未列出的公开对象。
#
# 注意：
# 调度器、HTTP 服务、REPL 的入口故意不放进来。
# 避免 `import runtime` 就把 FastAPI、APScheduler 一并拉起来，
# 只想要任务表或调用入口时不必承担这些依赖的初始化开销。
__all__ = [
    "ScheduledTask",
    "add_task",
    "all_tasks",
    "get_agent",
    "remove_task",
    "run_task",
]
