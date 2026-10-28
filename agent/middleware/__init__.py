"""
自定义中间件的统一导出入口。

对应课件 08 章第 5 节（自定义中间件）与 09 章 4.1 节（中间件中访问上下文）。

主要职责：
1. 把 custom.py 里实现的中间件集中导出。
2. 提供 CUSTOM_MIDDLEWARE 列表，供 graph.py 展开注入。
3. 对外隐藏中间件的具体实现文件。

注意：
create_agent 收的是中间件**实例**，不是类。
CUSTOM_MIDDLEWARE 里放的必须是已经构造好的对象，
写成 [TaskGateMiddleware, ContextInjectionMiddleware] 会直接报错。
"""

# ============================================================
# 1. 导入自定义中间件
# ============================================================

# 从 custom.py 导入：
#
# ContextInjectionMiddleware：
#   把运行时上下文（任务名、触发方式、写权限）
#   注入系统提示词。
#
# CUSTOM_MIDDLEWARE：
#   上面这些中间件组成的**实例列表**，
#   graph.py 用 *CUSTOM_MIDDLEWARE 展开后传给 create_agent。
#
# TaskGateMiddleware：
#   只读模式下的写操作闸门，
#   拦截 write_file / notify 这类写工具。
#
# TimingMiddleware：
#   统计本轮模型调用次数与 token 用量，写进日志。
from .custom import (
    ContextInjectionMiddleware,
    CUSTOM_MIDDLEWARE,
    TaskGateMiddleware,
    TimingMiddleware,
)


# ============================================================
# 2. 定义通配符导出列表
# ============================================================

# __all__ 主要控制：
#
# from agent.middleware import *
#
# 这种写法会导入哪些名称。
#
# 它不会禁止其他代码显式导入未列出的公开对象。
#
# 注意：
# 类的名字也一并导出，是为了让调用方能单独构造某个中间件
# （比如测试里只挂 TaskGateMiddleware）。
# 正常装配走 CUSTOM_MIDDLEWARE 就够了。
__all__ = [
    "CUSTOM_MIDDLEWARE",
    "ContextInjectionMiddleware",
    "TaskGateMiddleware",
    "TimingMiddleware",
]


# ============================================================
# 3. 使用示例
# ============================================================

# 装配 Agent 时（graph.py 的做法）：
#
# from .middleware.custom import CUSTOM_MIDDLEWARE
#
# middleware = [
#     *CUSTOM_MIDDLEWARE,
# ]
#
# 单独使用某个中间件：
#
# from agent.middleware import TaskGateMiddleware
#
# agent = create_agent(
#     model=model,
#     tools=ALL_TOOLS,
#     middleware=[TaskGateMiddleware()],
# )
#
# 注意：这里传的是 TaskGateMiddleware()，
# 带括号是构造实例，不带括号传的是类本身。
