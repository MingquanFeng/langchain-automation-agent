
"""
Agent 装配模块（graph.py）。

对应课件：
- 第 07 章（courseware/langChain/尚硅谷-07-智能体.pdf）：create_agent，创建 Agent。
- 第 08 章（courseware/langChain/尚硅谷-08-中间件.pdf）：Middleware，中间件机制。
- 第 09 章（courseware/langChain/尚硅谷-09-上下文与记忆.pdf）：Memory，记忆与状态持久化。

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
# 用于创建数据库连接，持久化 Agent 的执行状态和长期记忆。
import sqlite3

# os：用于收紧长期记忆数据库的本机文件权限。
import os

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

# InMemoryStore / SqliteStore：
# 提供跨会话访问的键值存储。
#
# 注意：
# Store 与 Checkpointer 的职责不同。
# Checkpointer 主要保存线程的执行状态。
# Store 用于存储可跨线程访问的数据。
#
# InMemoryStore 用于显式选择 memory 后端和测试隔离；
# SqliteStore 是默认的持久化长期记忆后端。
# 两者都不会自动提取或保存用户偏好。
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore


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
from .middleware.events import EventBridgeMiddleware
from .middleware.todos import ObservableTodoListMiddleware
from .middleware.cancellation import (
    CancellationMiddleware, retry_unless_cancelled, tool_failure_message,
)
from .middleware.pii import build_pii_middleware

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
        retry_on=retry_unless_cancelled,
        # 耗尽重试必须成为失败状态，不能把错误说明当成成功回答保存。
        on_failure="error",

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
        retry_on=retry_unless_cancelled,

        # 普通工具失败交给 Agent 处理；取消信号由回调直接抛出，不再继续。
        #
        # 这样模型有机会根据错误信息决定下一步。
        # 但不保证模型一定会向用户报告失败。
        on_failure=tool_failure_message,
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

        # 会话检查点可能包含完整对话、工具参数和中间状态；与长期记忆、
        # 运行记录一样，不能沿用常见的 0644 默认权限。setup() 创建完
        # 主数据库文件后再收紧权限，保证首次启动和已有数据库都生效。
        os.chmod(path, 0o600)

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
# 10. Store 构建函数
# ============================================================

def _build_store():
    """
    根据运行配置创建跨会话长期记忆 Store。

    MEMORY_BACKEND=sqlite 时，Store 使用独立的 store.sqlite 数据库，
    与保存会话状态的 memory.sqlite 分离；其他值使用进程内存。

    本函数只提供持久化容器。哪些内容允许成为长期记忆、如何查看
    和删除，将由后续的显式记忆入口负责，当前不会自动复制对话。
    """
    # --------------------------------------------------------
    # 10.1 判断是否使用 SQLite
    # --------------------------------------------------------

    # MEMORY_BACKEND 与 Checkpointer 共用同一个总开关：
    # - sqlite：会话状态和长期记忆都持久化，但使用不同数据库；
    # - 其他值：两者都只保存在当前进程内存中。
    if RUNTIME.memory_backend != "sqlite":
        return InMemoryStore()

    # --------------------------------------------------------
    # 10.2 准备数据库路径
    # --------------------------------------------------------

    # expanduser() 支持配置中使用 ~；resolve() 将相对路径固定为绝对路径，
    # 避免后续工作目录变化时同一个进程连接到不同文件。
    path = Path(RUNTIME.store_path).expanduser().resolve()

    # 首次启动时 data/ 可能不存在。parents=True 递归创建父目录，
    # exist_ok=True 允许目录已经存在。
    path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 10.3 建立适合 SqliteStore 的连接
    # --------------------------------------------------------

    # timeout 与 busy_timeout 让短暂写锁等待一段时间，避免桌面运行时
    # 因一次并发写入立刻失败；SqliteStore 自身还会用锁串行化连接访问。
    conn = sqlite3.connect(
        path,
        check_same_thread=False,
        timeout=10,
        # SqliteStore 会显式执行 BEGIN / COMMIT；连接必须使用 autocommit，
        # 这也与它官方 from_conn_string() 的连接方式一致。
        isolation_level=None,
    )

    # sqlite3.connect(timeout=10) 设置连接层的锁等待时间；
    # busy_timeout 再明确设置 SQLite 连接当前使用的毫秒值。
    conn.execute("PRAGMA busy_timeout=10000")

    # WAL 允许读操作与单个写操作更好地并行，适合桌面应用持续运行场景。
    # SQLite 仍然只有一个写者，因此它不是跨进程并发协调方案。
    conn.execute("PRAGMA journal_mode=WAL")

    # --------------------------------------------------------
    # 10.4 初始化 Store 表结构
    # --------------------------------------------------------

    # 将已经配置好的连接交给 LangGraph Store。
    store = SqliteStore(conn)

    # setup() 创建和迁移 Store 所需表结构，是幂等操作，可以每次启动调用。
    store.setup()

    # 限制 checkpoint 或连接关闭后可保留的 WAL 大小。
    # 64 MiB 是上限目标，不代表运行期间 WAL 绝不会临时超过该值。
    conn.execute("PRAGMA journal_size_limit=67108864")

    # --------------------------------------------------------
    # 10.5 收紧文件权限并返回 Store
    # --------------------------------------------------------

    # 长期记忆可能包含用户偏好或事实，不应沿用常见的 0644 默认权限。
    # setup() 已确保主数据库文件存在，再把它收紧为仅当前用户可读写。
    os.chmod(path, 0o600)

    # 连接不能在这里关闭：SqliteStore 会在整个 Agent 生命周期中复用它。
    # 当前 Agent 是运行时单例，连接随 sidecar 进程结束而释放。
    return store


# ============================================================
# 11. Agent 构建函数
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
            不传时调用 _build_store()；默认使用 SQLite 持久化 Store。

    这种设计称为依赖注入：
        调用方可以替换组件，
        不需要修改 Agent 的装配逻辑。

    返回：
        LangChain create_agent() 创建的 Agent。
    """

    # --------------------------------------------------------
    # 11.1 组装中间件列表
    # --------------------------------------------------------

    middleware = [
        # 事件桥接放最外层 ——
        # 它要包住整个「工具调用 + 重试」过程，
        # 这样客户端看到的是「一次工具调用」而不是「一次重试」。
        #
        # wrap-style 钩子是洋葱结构：第一个定义的最外层。
        # 放在 ToolRetryMiddleware 里面的话，
        # 一个失败的工具会重试 3 次、发 3 条 tool.error —— 客户端看到的是噪音。
        # 放在外面则只有一对事件，duration_ms 也是用户真正等待的时间。
        EventBridgeMiddleware(),

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

        # 位于重试层内侧，确保每次重试前也检查取消。
        CancellationMiddleware(),

        # 加入 TodoList 任务管理中间件。
        TODO,

        # 加入上下文清理中间件。
        CONTEXT_EDITING,

        # 加入 PII 脱敏中间件。
        #
        # 放在自定义中间件之前 ——
        # 先把 PII 去掉，后面几个中间件看到的就是干净内容。
        # 脱敏开关关闭时这是空列表，展开后什么都不加。
        *build_pii_middleware(),

        # 加入项目自定义中间件。
        *CUSTOM_MIDDLEWARE,
    ]

    # --------------------------------------------------------
    # 11.2 创建并返回 Agent
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
        # 否则调用 _build_store()，根据 MEMORY_BACKEND
        # 创建 SQLite 持久化 Store 或 InMemoryStore。
        #
        # Store 可以保存跨线程访问的数据，
        # 但需要工具或中间件实现具体读写逻辑。
        #
        # SQLite Store 与 Checkpointer 使用不同数据库：
        # 一个保存跨会话长期记忆，一个保存 thread 状态。
        store=(
            store
            if store is not None
            else _build_store()
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
# 12. TodoList 任务拆解中间件
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
TODO = ObservableTodoListMiddleware()


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

SYSTEM_PROMPT = """你是阿序（Axu），用户的个人桌面 AI Agent。

工作方式：
1. 先用 TodoList 把多步任务拆成清晰的子步骤，然后逐条推进。
2. 普通进度和最终结果直接回复当前聊天，不要调用通知工具。
3. 只有用户明确要求“通知、推送、发送到 webhook 或通知渠道”时，才调用 notify 或 notify_summary；不能因为任务完成、需要汇总或通知渠道可用就自行调用。
4. 用户明确要求通知时，通知正文只给结论，不要复述你的思考过程；同一份结果不要同时调用 notify 和 notify_summary。
5. 需要读写本地文件时，路径都相对于当前 Agent 工作目录。
6. 如果某一步失败，说明失败原因和已经尝试过的办法，不要静默跳过。
7. 缺少信息时按最合理的默认做法继续，并在结果里说明你用了什么默认值。
8. 运行开发命令只能使用 run_command 的参数数组。它只适合检查、测试、构建和格式检查；不要尝试安装依赖、启动服务或执行任意 Shell。用户指定单条命令时，只执行该命令一次；成功后立刻汇报并停止调用工具。失败后只解释该命令的错误，不要自行改用其他解释器、路径、命令或探测命令；只有用户明确要求诊断或替代执行时才能继续。
9. 若任务提供失败检查的修复上下文，先定位根因并做最小源码或测试编辑；每次编辑和复跑都等待单独批准，只复跑指定原命令，达到三次验证仍失败就停止并汇总。
10. 只有用户明确说“记住、保存为长期记忆、以后都按这个偏好”等要求时，才调用 save_memory；不要从普通聊天中自行推断或保存。
11. 长期记忆只能保存一条简短、独立的偏好或事实，不能保存整段对话、长文档、工具输出、密码、密钥或令牌。
12. 用户询问已保存内容或要求按既有偏好处理时，可调用 list_memories；只有用户明确要求遗忘或删除时，才调用 delete_memory。
13. 修改或删除已有记忆时，先用 list_memories 查到准确 ID 和 revision，并把 revision 原样传给写工具；发生冲突时重新查询，不能用猜测的 ID 或旧修订号覆盖数据。
14. 新建长期记忆默认永不过期；只有用户明确要求临时保存时才选择 30、90 或 365 天过期策略。相同内容已存在时不要重复创建。
15. 当前没有删除本地文件的工具。收到删除文件请求时，明确说明“当前不支持删除文件，文件未删除”；不要用写文件、测试脚本或其他命令绕过限制。无法执行的任务必须说明原因，不能只返回权限提示或声称完成。
16. 解释命令结果时，以工具返回的实际执行参数、退出码和输出为准。Python -I 会忽略当前目录、PYTHONPATH 和用户 site-packages；不要把隔离模式下的模块缺失归因于工作目录的同名文件，也不要猜测未经验证的环境情况。
17. 最终答复先给结论，再列必要的结果和后续步骤；能简短说清时不要重复工具调用过程或逐字复述工具输出。多项内容用 Markdown 标题、列表和加粗组织；审批拒绝时明确说明工具未执行、文件未改动，不要扩写未验证的状态。
"""

# 提示词注意事项：
#
# 1. notify 和 notify_summary 是可选的外部通知出口，
#    只有用户明确要求通知时才应调用。
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
#
# 7. 长期记忆写入与删除仍由风险中间件逐次审批，
#    “只有明确要求才调用”的提示词不是唯一安全边界。
