
"""
Agent 装配模块（graph.py）。

对应课件：
- 第 07 章（courseware/尚硅谷-07-智能体.pdf）：create_agent，创建 Agent。
- 第 08 章（courseware/尚硅谷-08-中间件.pdf）：Middleware，中间件机制。
- 第 09 章（courseware/尚硅谷-09-上下文与记忆.pdf）：Memory，记忆与状态持久化。

主要职责：
1. 配置上下文清理、失败重试和调用次数限制。
2. 根据配置创建 Checkpointer，保存 Agent 执行状态。
3. 将模型、工具、中间件、记忆和上下文组装成 Agent。
4. 提供统一的 build_agent() 创建入口。

注意：
本模块负责创建 Agent，不负责直接执行用户任务。
真正执行任务通常需要调用 agent.invoke()。
"""

from __future__ import annotations

# SQLite：Python 内置的轻量级数据库模块。
# 用于创建数据库连接，持久化 Agent 的执行状态。
import sqlite3

# Path：用于处理文件路径、获取父目录和创建目录。
from pathlib import Path


# ============================================================
# 1. 导入 LangChain Agent 创建函数
# ============================================================

# create_agent 是 LangChain 提供的 Agent 构建函数。
#
# 它负责组织模型与工具的调用循环：
# 用户输入 → 模型判断 → 工具调用 → 模型继续推理 → 最终回答。
from langchain.agents import create_agent


# ============================================================
# 2. 导入 LangChain 内置中间件
# ============================================================

from langchain.agents.middleware import (
    # 限制模型调用次数，避免无限循环和费用失控。
    ModelCallLimitMiddleware,

    # 模型调用失败时自动重试。
    ModelRetryMiddleware,

    # 为 Agent 提供待办事项管理能力。
    # 模型可以创建、更新任务列表并追踪任务进度。
    TodoListMiddleware,

    # 限制工具调用次数。
    ToolCallLimitMiddleware,

    # 工具执行失败时自动重试。
    ToolRetryMiddleware,
)


# ============================================================
# 3. 导入上下文编辑中间件
# ============================================================

from langchain.agents.middleware.context_editing import (
    # 定义旧工具调用内容的清理策略。
    ClearToolUsesEdit,

    # 在满足触发条件时执行上下文清理。
    ContextEditingMiddleware,
)


# ============================================================
# 4. 导入 LangGraph 记忆组件
# ============================================================

# InMemorySaver：
# 将 Agent 的检查点状态保存在当前进程内存中。
# 程序重启后，内存中的状态会丢失。
from langgraph.checkpoint.memory import InMemorySaver

# SqliteSaver：
# 将 Agent 的检查点状态保存到 SQLite 数据库。
# 支持程序重启后恢复已保存的对话和执行状态。
from langgraph.checkpoint.sqlite import SqliteSaver

# InMemoryStore：
# 提供跨会话访问的内存键值存储。
#
# 注意：
# Store 与 Checkpointer 的职责不同。
# Checkpointer 主要保存线程的执行状态。
# Store 用于存储可跨线程访问的数据。
#
# InMemoryStore 不会自动持久化到磁盘，
# 也不会自动提取或保存用户偏好。
from langgraph.store.memory import InMemoryStore


# ============================================================
# 5. 导入项目内部组件
# ============================================================

# MODEL：模型配置对象，提供 build() 方法创建模型。
# RUNTIME：应用运行配置，包含数据库路径、调用次数限制等。
from .config import MODEL, RUNTIME

# AgentContext：
# 定义单次 Agent 调用的运行时上下文结构。
#
# 例如：
# task_name：任务名称。
# trigger：任务触发来源。
# allow_write：是否允许写操作。
# notify_channel：通知渠道。
from .context import AgentContext

# 项目自定义中间件列表。
# 用于扩展日志、权限检查、通知等功能。
# 具体功能取决于 custom.py 的实现。
from .middleware.custom import CUSTOM_MIDDLEWARE

# Agent 可以调用的工具集合。
# 例如文件操作、查询、通知等工具。
# 实际有哪些工具取决于 tools 模块的实现。
from .tools import ALL_TOOLS


# ============================================================
# 6. 上下文清理中间件
# ============================================================

# Agent 在执行任务时，模型和工具的消息会持续累积。
#
# 例如：
# 第一次工具调用：返回 10000 token。
# 第二次工具调用：返回 20000 token。
# 第三次工具调用：返回 30000 token。
#
# 随着上下文增长：
# 1. 模型调用成本可能增加。
# 2. 响应速度可能下降。
# 3. 可能超过模型上下文窗口。
#
# ContextEditingMiddleware 用于在达到指定条件时，
# 清理较旧的工具调用内容，减少上下文占用。

CONTEXT_EDITING = ContextEditingMiddleware(
    # edits 接收上下文编辑策略列表。
    edits=[
        ClearToolUsesEdit(
            # 上下文达到指定 token 阈值时触发清理。
            #
            # 120_000 是 Python 的数字分隔写法，
            # 等价于整数 120000。
            #
            # 具体 token 计算方式由中间件实现决定。
            trigger=120_000,

            # 保留最近 3 条符合策略的工具结果。
            # 较旧的工具调用内容可以被清理。
            keep=3,

            # 清理旧工具内容时，
            # 同时清理对应的工具输入参数。
            #
            # 对于输入参数较大的工具，
            # 可以进一步减少上下文占用。
            clear_tool_inputs=True,
        )
    ]
)


# ============================================================
# 7. 失败重试中间件
# ============================================================

# RETRY 是一个 Python 列表。
# 列表中保存两个中间件实例。
#
# 注意：
# 创建中间件对象并不会立即执行重试。
# 只有 Agent 运行并发生对应调用时，
# 中间件才会参与执行。

RETRY = [
    # ------------------------
    # 7.1 模型调用重试
    # ------------------------

    ModelRetryMiddleware(
        # 模型调用失败时，最多重试 3 次。
        max_retries=3,

        # 重试等待时间的退避系数。
        # 用于降低连续请求对服务端造成的压力。
        backoff_factor=2.0,
    ),

    # ------------------------
    # 7.2 工具调用重试
    # ------------------------

    ToolRetryMiddleware(
        # 工具执行失败时，最多重试 2 次。
        max_retries=2,

        # 如果重试后仍然失败，
        # 将失败作为可继续处理的结果交给 Agent。
        #
        # 这样模型有机会根据错误信息决定下一步。
        # 但不保证模型一定会向用户报告失败。
        on_failure="continue",
    ),
]

# 注意：
# 对于发送消息、修改文件、写入数据库等工具，
# 自动重试可能造成重复操作。
#
# 例如：
# 第一次发送消息实际上成功，
# 但网络超时导致客户端认为失败。
# 重试后可能再次发送同一条消息。
#
# 因此有副作用的工具需要考虑幂等性。


# ============================================================
# 8. 调用次数限制中间件
# ============================================================

# LIMITS 用于防止 Agent 在无人值守时反复调用模型或工具。
#
# 模型调用和工具调用是两种不同的计数：
#
# 模型调用：
#   向大语言模型发送一次请求。
#
# 工具调用：
#   执行一次 Agent 工具。

LIMITS = [
    # ------------------------
    # 8.1 模型调用限制
    # ------------------------

    ModelCallLimitMiddleware(
        # 从 RUNTIME 读取最大模型调用次数。
        #
        # 例如：
        # RUNTIME.max_model_calls = 30
        #
        # thread_limit 表示同一线程的累计限制，
        # 不等于单次 invoke() 的调用次数限制。
        thread_limit=RUNTIME.max_model_calls,
    ),

    # ------------------------
    # 8.2 工具调用限制
    # ------------------------

    ToolCallLimitMiddleware(
        # 同一线程累计最多调用 50 次工具。
        thread_limit=50,
    ),
]

# 注意：
# 如果多个任务重复使用同一个 thread_id，
# 调用次数可能在该线程内持续累计。
#
# 如果希望限制单次 Agent 运行，
# 应考虑使用 run_limit。
#
# 另外，中间件的排列顺序可能影响
# 重试与调用次数统计之间的关系。


# ============================================================
# 9. Checkpointer 构建函数
# ============================================================

def _build_checkpointer():
    """
    根据运行配置创建 Agent Checkpointer。

    Checkpointer 的作用：
    1. 保存 Agent 的消息和执行状态。
    2. 支持同一 thread_id 的多轮对话。
    3. 支持从已保存的检查点恢复状态。

    存储后端：
        sqlite：使用 SQLite 持久化。
        其他值：使用内存存储。

    返回：
        SqliteSaver 或 InMemorySaver。

    函数名以 _ 开头：
        表示模块内部使用的辅助函数，
        这是 Python 命名约定，不是访问限制。
    """

    # --------------------------------------------------------
    # 9.1 判断是否使用 SQLite
    # --------------------------------------------------------

    # RUNTIME.memory_backend 来自 config.py。
    #
    # 例如：
    # MEMORY_BACKEND=sqlite
    #
    # 条件成立时使用 SQLite。
    if RUNTIME.memory_backend == "sqlite":

        # 将数据库路径转换为 Path 对象。
        #
        # 例如：
        # data/memory.sqlite
        path = Path(RUNTIME.sqlite_path)

        # 确保数据库文件所在的父目录存在。
        #
        # parents=True：
        # 递归创建缺失的父目录。
        #
        # exist_ok=True：
        # 目录已经存在时不报错。
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # 9.2 建立 SQLite 数据库连接
        # ----------------------------------------------------

        # sqlite3.connect() 用于建立数据库连接。
        #
        # check_same_thread=False：
        # 允许连接被创建线程之外的线程使用。
        #
        # 注意：
        # 这不代表 SQLite 连接可以无约束地并发读写。
        conn = sqlite3.connect(
            path,
            check_same_thread=False,
        )

        # ----------------------------------------------------
        # 9.3 创建 SQLite Checkpointer
        # ----------------------------------------------------

        # 将数据库连接传给 SqliteSaver。
        #
        # SqliteSaver 通过该连接读写检查点数据。
        saver = SqliteSaver(conn)

        # 初始化 Checkpointer 所需的数据库结构。
        #
        # 包括必要的数据表等。
        #
        # 注意：
        # 不应假定 setup() 一定会启用 WAL 模式。
        # 是否启用 WAL 需要根据实际实现或 PRAGMA 查询确认。
        saver.setup()

        # ----------------------------------------------------
        # 9.4 设置 SQLite WAL 日志大小限制
        # ----------------------------------------------------

        # PRAGMA 是 SQLite 的数据库配置命令。
        #
        # journal_size_limit 用于设置回滚日志或
        # WAL 文件在符合条件的重置、截断等操作后
        # 允许保留的大小上限。
        #
        # 67108864 字节 = 64 MiB。
        #
        # 注意：
        # 该设置不会立即把现有 WAL 文件缩小到 64 MiB，
        # 也不保证 WAL 文件运行期间绝不超过这个大小。
        #
        # 如果数据库确实启用了 WAL，
        # 还需要关注 checkpoint 是否正常执行，
        # 以及是否存在阻碍 checkpoint 的长事务。
        conn.execute(
            "PRAGMA journal_size_limit=67108864"
        )

        # 返回 SQLite Checkpointer。
        #
        # return 执行后，函数立即结束。
        #
        # 当前实现没有显式关闭 conn 的逻辑，
        # 长期运行时应由应用管理连接生命周期。
        return saver

    # --------------------------------------------------------
    # 9.5 使用内存 Checkpointer
    # --------------------------------------------------------

    # 如果 memory_backend 不是 sqlite，
    # 使用 InMemorySaver。
    #
    # 它将检查点保存在进程内存中，
    # 程序退出后状态会丢失。
    return InMemorySaver()


# ============================================================
# 10. Agent 构建函数
# ============================================================

def build_agent(
    model=None,
    tools=None,
    checkpointer=None,
    store=None,
):
    """
    创建并返回一个完整的 Agent。

    这是整个 graph.py 最核心的函数。

    主要职责：
    1. 组合所有中间件。
    2. 创建或接收模型实例。
    3. 注入工具列表。
    4. 注入 Checkpointer 和 Store。
    5. 设置系统提示词和运行时上下文。
    6. 调用 create_agent() 完成装配。

    参数：
        model：
            自定义模型。
            不传时使用 MODEL.build()。

        tools：
            自定义工具列表。
            不传时使用 ALL_TOOLS。

        checkpointer：
            自定义 Checkpointer。
            不传时调用 _build_checkpointer()。

        store：
            自定义 Store。
            不传时使用 InMemoryStore()。

    这种设计称为依赖注入：
        调用方可以替换组件，
        不需要修改 Agent 的装配逻辑。

    返回：
        LangChain create_agent() 创建的 Agent。
    """

    # --------------------------------------------------------
    # 10.1 组装中间件列表
    # --------------------------------------------------------

    middleware = [
        # *LIMITS：
        # 使用 * 解包列表中的元素。
        #
        # 假设 LIMITS = [A, B]，
        # *LIMITS 相当于依次放入 A、B。
        #
        # 这里加入模型和工具调用次数限制。
        *LIMITS,

        # 加入模型和工具失败重试机制。
        *RETRY,

        # 加入 TodoList 任务管理中间件。
        TODO,

        # 加入上下文清理中间件。
        CONTEXT_EDITING,

        # 加入项目自定义中间件。
        *CUSTOM_MIDDLEWARE,
    ]

    # --------------------------------------------------------
    # 10.2 创建并返回 Agent
    # --------------------------------------------------------

    return create_agent(

        # ------------------------
        # 模型
        # ------------------------

        # 如果调用方传入 model，则使用该模型。
        # 否则调用 MODEL.build() 创建默认模型。
        #
        # or 是 Python 的逻辑或运算符。
        # 它根据对象的真值决定使用哪一项。
        model=model or MODEL.build(),

        # ------------------------
        # 工具
        # ------------------------

        # 如果传入 tools，则使用自定义工具列表。
        # 否则使用 ALL_TOOLS。
        #
        # 使用 is not None 而不是 or，
        # 是为了允许调用方传入空列表 []。
        #
        # tools=[] 表示不使用默认工具。
        tools=tools if tools is not None else ALL_TOOLS,

        # ------------------------
        # 系统提示词
        # ------------------------

        # 设置 Agent 的角色、工作方式和行为规则。
        system_prompt=SYSTEM_PROMPT,

        # ------------------------
        # 中间件
        # ------------------------

        # 注入前面组装的中间件列表。
        middleware=middleware,

        # ------------------------
        # Checkpointer
        # ------------------------

        # 如果传入 checkpointer，则使用传入的实例。
        #
        # 否则调用 _build_checkpointer()，
        # 根据 RUNTIME.memory_backend 创建默认实例。
        #
        # Checkpointer 负责保存 Agent 的执行状态。
        checkpointer=(
            checkpointer
            if checkpointer is not None
            else _build_checkpointer()
        ),

        # ------------------------
        # Store
        # ------------------------

        # 如果传入 store，则使用自定义 Store。
        #
        # 否则创建 InMemoryStore。
        #
        # Store 可以保存跨线程访问的数据，
        # 但需要工具或中间件实现具体读写逻辑。
        #
        # InMemoryStore 不会在重启后保留数据。
        store=(
            store
            if store is not None
            else InMemoryStore()
        ),

        # ------------------------
        # 运行时上下文
        # ------------------------

        # 指定 AgentContext 为运行时上下文结构。
        #
        # 执行时可以传入：
        #
        # agent.invoke(
        #     {"messages": [...]},
        #     context=AgentContext(
        #         task_name="每日任务",
        #         trigger="schedule",
        #         allow_write=False,
        #     ),
        # )
        #
        # 工具和中间件可以通过 runtime.context
        # 读取这些信息。
        context_schema=AgentContext,

        # ------------------------
        # Agent 名称
        # ------------------------

        # 用于标识、日志和调试。
        # 不会自动改变 Agent 的推理能力。
        name="automation_agent",
    )


# ============================================================
# 11. TodoList 任务拆解中间件
# ============================================================

# TodoListMiddleware 为 Agent 提供待办事项管理能力。
#
# 例如：
# 用户：整理文件、检查重复项、生成报告。
#
# 模型可以创建任务列表：
# 1. 扫描文件。
# 2. 查找重复项。
# 3. 生成报告。
#
# 注意：
# 它提供任务管理工具，
# 不代表模型一定会先拆解所有任务。
TODO = TodoListMiddleware()


# ============================================================
# 12. Agent 系统提示词
# ============================================================

# SYSTEM_PROMPT 是传递给模型的系统级指令。
#
# 它用于规定 Agent 的角色和工作方式。
#
# 注意：
# 提示词属于模型行为约束，
# 不能代替程序级权限控制和错误处理。

SYSTEM_PROMPT = """你是一个自动化日常事务助手。

工作方式：
1. 先用 TodoList 把多步任务拆成清晰的子步骤，然后逐条推进。
2. 每一步做完就用 notify 或 notify_summary 把结果推给用户，不要攒到最后。
3. 通知里给结论，不要复述你的思考过程。
4. 需要读写本地文件时，路径都相对于数据目录。
5. 如果某一步失败，说明失败原因和已经尝试过的办法，不要静默跳过。
6. 缺少信息时按最合理的默认做法继续，并在结果里说明你用了什么默认值。
"""

# 提示词注意事项：
#
# 1. notify 和 notify_summary 必须是实际可用的工具，
#    或由项目提供对应的通知机制。
#
# 2. TodoListMiddleware 不能保证模型一定遵循
#    "先拆解任务" 的要求。
#
# 3. 文件路径约束需要在文件工具中实现。
#
# 4. AgentContext.allow_write 需要在写工具中检查。
#
# 5. 如果工具最终失败，
#    应通过程序化机制确保错误被正确记录和报告。
#
# 6. 对删除文件、发送消息等高风险操作，
#    不应仅依靠模型自行选择默认值。
