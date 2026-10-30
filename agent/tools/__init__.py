"""
Agent 工具集合的统一导出入口。

对应课件 05 章（courseware/尚硅谷-05-Tools.pdf）：@tool 定义工具，docstring 就是给模型看的说明。

主要职责：
1. 汇总文件工具和通知工具。
2. 提供 ALL_TOOLS 列表，供 graph.py 注入 create_agent。
3. 作为「Agent 能调用哪些工具」的唯一白名单。

注意：
ALL_TOOLS 是 Agent 能拿到的**全部**工具。
用 @tool 装饰过的函数如果不登记进这里，
模型根本看不到它 —— 工具写好了不登记，等于没写。
"""

# ============================================================
# 1. 导入各工具分组
# ============================================================

# FILE_TOOLS：本地文件读写工具。
# 来自 files.py，全部被沙箱限制在 data/ 目录内。
from .files import FILE_TOOLS

# NOTIFY_TOOLS：通知出口工具。
# 来自 notify.py，webhook 未配置时降级为终端输出。
from .notify import NOTIFY_TOOLS


# ============================================================
# 2. 汇总全部工具
# ============================================================

# ALL_TOOLS 是传给 create_agent(tools=...) 的列表，
# 也是模型能看到的工具全集。
#
# * 是列表解包：
# 把 FILE_TOOLS 和 NOTIFY_TOOLS 里的元素摊平到同一个列表。
#
# 用 *FILE_TOOLS, *NOTIFY_TOOLS 而不是直接拼接 +，
# 是因为 + 只能连接两个列表，解包的方式可以继续加第三组。
ALL_TOOLS = [*FILE_TOOLS, *NOTIFY_TOOLS]


# ============================================================
# 3. 定义通配符导出列表
# ============================================================

# __all__ 主要控制：
#
# from agent.tools import *
#
# 这种写法会导入哪些名称。
#
# 它不会禁止其他代码显式导入未列出的公开对象。
#
# FILE_TOOLS 和 NOTIFY_TOOLS 一并导出，
# 是为了让调用方能只装配某一组工具
# （比如只给只读任务挂文件工具，不给通知工具）。
__all__ = ["ALL_TOOLS", "FILE_TOOLS", "NOTIFY_TOOLS"]


# ============================================================
# 4. 使用示例
# ============================================================

# 默认装配（graph.py 的做法）：
#
# from .tools import ALL_TOOLS
#
# agent = create_agent(
#     model=model,
#     tools=ALL_TOOLS,
# )
#
# 只装配文件工具：
#
# from agent.tools import FILE_TOOLS
#
# agent = create_agent(model=model, tools=FILE_TOOLS)
#
# 注意：
# 新增写类工具时，除了登记进 ALL_TOOLS，
# 还要同步更新 middleware/custom.py 里
# TaskGateMiddleware.WRITE_TOOLS，
# 否则只读模式下这个工具不会被拦住。
