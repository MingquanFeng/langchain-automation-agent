"""运行时 SQLite 存储：定时任务注册表与 Agent 执行历史。

职责：
1. 首次启动时从 `runtime/tasks.py` 默认列表初始化任务表；后续以 SQLite 注册表为准。
2. 为同步入口和 WebSocket 流式入口记录 Agent run 的开始、终态和结果。
3. 服务重启时将仍标记为 `running` 的旧记录归档为 `interrupted`。

设计边界：
- 运行数据使用独立的 `runtime.sqlite3`，不复用 LangGraph Checkpointer 的连接对象。
- 每次读写短暂创建一个 SQLite 连接；连接设置 busy timeout，事务结束后立即关闭。
- 运行记录包含指令与最终输出，默认保存在本地运行目录；POSIX 系统将数据库文件权限限制为当前用户读写。
"""

from __future__ import annotations

# os：区分 POSIX 与 Windows，决定是否设置 Unix 文件权限。
import os

# sqlite3：保存任务注册表与 Agent 执行记录。
import sqlite3

# threading：保护惰性初始化与进程级存储单例。
import threading

# time：生成运行记录的开始、结束和中断时间戳。
import time

# contextmanager：统一管理每次数据库操作使用的短生命周期连接。
from contextlib import contextmanager

# datetime / timezone：把 Unix 时间戳序列化为带 UTC 时区的 ISO 字符串。
from datetime import datetime, timezone

# Path：保存数据库文件路径并确保其父目录存在。
from pathlib import Path

# Iterator：标注数据库连接上下文管理器产生的连接类型。
from typing import Iterator

# RUNTIME：运行配置提供 Checkpointer 路径；运行数据数据库放在同目录。
from agent.config import RUNTIME


# ============================================================
# 1. RuntimeStore：运行时数据访问层
# ============================================================

class RuntimeStore:
    """使用独立 SQLite 文件保存运行时数据。

    一个实例对应一个数据库路径。公开方法会先确保 schema 已初始化，调用方不需要
    自己管理连接；任务数据和运行记录分别由独立表保存。
    """

    def __init__(self, path: Path | str) -> None:
        """创建存储对象，但延迟建库直到第一次实际访问。

        参数：
            path：SQLite 文件路径。路径可以是字符串或 Path。
        """
        # expanduser 处理用户目录简写；resolve 固化为绝对路径，避免工作目录变化影响位置。
        self.path = Path(path).expanduser().resolve()

        # `_initialized` 避免同一个实例重复建表；新进程会重新做幂等 schema 检查。
        self._initialized = False

        # 初始化可能由服务启动线程和第一次存储调用同时触发，用锁保证只执行一次。
        self._initialize_lock = threading.Lock()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """打开短生命周期数据库连接，并统一提交、回滚与关闭逻辑。"""
        # timeout 同时设置 SQLite 锁等待上限，避免并发写入时立即报 database is locked。
        connection = sqlite3.connect(self.path, timeout=10)

        # Row 支持按列名读取，避免序列化时依赖 SELECT 列的数字索引。
        connection.row_factory = sqlite3.Row
        try:
            # 正常退出时提交事务；调用方无需在每个方法中重复写 commit。
            yield connection
            connection.commit()
        except Exception:
            # 操作中途失败时回滚本次事务，防止写入半条任务或半条运行记录。
            connection.rollback()
            raise
        finally:
            # 每次操作结束都释放连接与文件句柄。
            connection.close()

    def initialize(self) -> None:
        """幂等创建数据库 schema。

        建表与崩溃恢复分开：本方法只负责结构初始化；
        `mark_interrupted_runs()` 由服务启动生命周期显式调用。
        """
        # 快速路径：大多数请求在服务启动时已完成初始化。
        if self._initialized:
            return

        with self._initialize_lock:
            # 获取锁后再检查，避免两个首发线程都进入 schema 创建流程。
            if self._initialized:
                return

            # 首次运行时创建数据库所在目录；桌面版通常是应用支持目录下的 data/。
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                # WAL 让读操作可与写操作并行；busy_timeout 给短暂写锁留出等待时间。
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA busy_timeout=10000")

                # 元数据表保存一次性初始化标记；即使用户删除全部任务，也不重新注入默认项。
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runtime_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )

                # 任务表保存可由 API 动态增删改的 cron 定义。
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scheduled_tasks (
                        name TEXT PRIMARY KEY,
                        cron TEXT NOT NULL,
                        instruction TEXT NOT NULL,
                        allow_write INTEGER NOT NULL,
                        enabled INTEGER NOT NULL
                    )
                    """
                )

                # 执行记录表保留输入、输出、状态与耗时，供 `/runs` 查询。
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_runs (
                        run_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        task_name TEXT NOT NULL,
                        trigger_source TEXT NOT NULL,
                        instruction TEXT NOT NULL,
                        status TEXT NOT NULL,
                        output TEXT NOT NULL DEFAULT '',
                        error TEXT,
                        started_at REAL NOT NULL,
                        finished_at REAL,
                        duration_ms INTEGER,
                        message_count INTEGER
                    )
                    """
                )

                # 兼容已创建但还没有 message_count 字段的运行数据库。
                run_columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(agent_runs)")
                }
                if "message_count" not in run_columns:
                    connection.execute("ALTER TABLE agent_runs ADD COLUMN message_count INTEGER")

                # 历史列表按开始时间倒序查询；索引避免记录增长后扫描整张表。
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_runs_started_at "
                    "ON agent_runs(started_at DESC)"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_runs_thread_id "
                    "ON agent_runs(thread_id)"
                )

            # 指令和输出可能包含用户数据；POSIX 下数据库文件仅允许当前用户读写。
            if os.name != "nt":
                self.path.chmod(0o600)

            # 只有 schema 与权限设置全部成功后才开放快速路径。
            self._initialized = True

    def load_or_seed_tasks(self, defaults: list[dict]) -> list[dict]:
        """读取任务表；数据库首次初始化时将代码默认任务写入表中。

        参数：
            defaults：由 `ScheduledTask` 转成的普通字典列表。

        返回：
            按首次插入顺序排列的任务字典列表。
        """
        self.initialize()
        with self._connect() as connection:
            # 用写事务保护「检查初始化标记 → 插入默认任务」这一对操作。
            connection.execute("BEGIN IMMEDIATE")
            initialized = connection.execute(
                "SELECT value FROM runtime_metadata WHERE key = 'scheduled_tasks_initialized'"
            ).fetchone()
            if initialized is None:
                # 只在新数据库第一次读任务时种入默认列表，不覆盖用户后续编辑。
                connection.executemany(
                    """
                    INSERT INTO scheduled_tasks(name, cron, instruction, allow_write, enabled)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            task["name"],
                            task["cron"],
                            task["instruction"],
                            int(task["allow_write"]),
                            int(task["enabled"]),
                        )
                        for task in defaults
                    ],
                )
                # 即使默认列表为空也记录初始化完成，避免未来启动时误认为是新数据库。
                connection.execute(
                    "INSERT INTO runtime_metadata(key, value) VALUES (?, ?)",
                    ("scheduled_tasks_initialized", "1"),
                )

            # rowid 保留任务首次进入注册表的顺序，方便 UI 保持稳定排序。
            rows = connection.execute(
                "SELECT name, cron, instruction, allow_write, enabled "
                "FROM scheduled_tasks ORDER BY rowid"
            ).fetchall()
        # SQLite 的 0/1 整数转换回 Python bool，避免向业务层泄露存储细节。
        return [
            {
                "name": row["name"],
                "cron": row["cron"],
                "instruction": row["instruction"],
                "allow_write": bool(row["allow_write"]),
                "enabled": bool(row["enabled"]),
            }
            for row in rows
        ]

    def save_task(self, task: dict) -> None:
        """插入或覆盖一个任务定义。

        参数：
            task：含 `name`、`cron`、`instruction`、`allow_write`、`enabled` 的字典。
        """
        self.initialize()
        with self._connect() as connection:
            # 以任务名为唯一键；同名 POST 更新已有定义，不生成重复任务。
            connection.execute(
                """
                INSERT INTO scheduled_tasks(name, cron, instruction, allow_write, enabled)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    cron = excluded.cron,
                    instruction = excluded.instruction,
                    allow_write = excluded.allow_write,
                    enabled = excluded.enabled
                """,
                (
                    task["name"], task["cron"], task["instruction"],
                    int(task["allow_write"]), int(task["enabled"]),
                ),
            )

    def delete_task(self, name: str) -> None:
        """按任务名删除持久定义；不存在时保持幂等。"""
        self.initialize()
        with self._connect() as connection:
            connection.execute("DELETE FROM scheduled_tasks WHERE name = ?", (name,))

    def start_run(
        self,
        *,
        run_id: str,
        thread_id: str,
        task_name: str,
        trigger: str,
        instruction: str,
        started_at: float | None = None,
    ) -> None:
        """登记一次 Agent 运行。

        该行先以 `running` 状态写入，执行终态由 `finish_run()` 更新。
        进程在两步之间退出时，下一次服务启动会将其标记为 `interrupted`。
        """
        self.initialize()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs(
                    run_id, thread_id, task_name, trigger_source, instruction,
                    status, started_at
                ) VALUES (?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    run_id, thread_id, task_name, trigger, instruction,
                    started_at if started_at is not None else time.time(),
                ),
            )

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        output: str = "",
        error: str | None = None,
        duration_ms: int | None = None,
        message_count: int | None = None,
        finished_at: float | None = None,
    ) -> bool:
        """更新运行终态与结果。

        参数：
            status：内部状态之一，如 `succeeded`、`failed`、`cancelled`。
            output：最终回答；没有成功输出时为空字符串。
            error：失败或取消说明；正常完成时为 None。
            duration_ms：运行耗时；未知时为 None。
            message_count：Agent 最终消息数；未知时为 None。

        返回：
            True 表示本次调用完成了从 running 到终态的转换；False 表示
            记录不存在或已经是终态。终态一经写入不能被迟到的错误覆盖。
        """
        self.initialize()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE agent_runs
                SET status = ?, output = ?, error = ?, duration_ms = ?,
                    message_count = ?, finished_at = ?
                WHERE run_id = ?
                  AND status = 'running'
                """,
                (
                    status, output, error, duration_ms, message_count,
                    finished_at if finished_at is not None else time.time(), run_id,
                ),
            )
            return cursor.rowcount > 0

    def mark_interrupted_runs(self) -> int:
        """将服务上次退出时未结束的运行改为 `interrupted`。

        返回：
            被更新的记录数。
        """
        self.initialize()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE agent_runs
                SET status = 'interrupted',
                    error = '服务重启时任务仍在运行',
                    finished_at = ?
                WHERE status = 'running'
                """,
                (time.time(),),
            )
            return cursor.rowcount

    def list_runs(self, *, limit: int = 50, offset: int = 0) -> list[dict]:
        """按开始时间倒序分页读取运行历史。

        参数：
            limit：最多读取多少条；HTTP 层限制在 1 到 100。
            offset：跳过的最新记录数。

        返回：
            JSON 可序列化的运行记录列表，时间字段使用 UTC ISO 格式。
        """
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, thread_id, task_name, trigger_source, instruction,
                       status, output, error, started_at, finished_at,
                       duration_ms, message_count
                FROM agent_runs
                ORDER BY started_at DESC, rowid DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._serialize_run(row) for row in rows]

    def delete_run(self, run_id: str) -> bool:
        """删除一条运行记录，不修改它所属会话的 Checkpointer。"""
        self.initialize()
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM agent_runs WHERE run_id = ?", (run_id,)
            )
            return cursor.rowcount > 0

    def is_run_running(self, run_id: str) -> bool:
        """判断运行记录是否仍处于执行态。"""
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM agent_runs WHERE run_id = ? AND status = 'running'",
                (run_id,),
            ).fetchone()
        return row is not None

    def has_running_run_for_thread(self, thread_id: str) -> bool:
        """判断会话是否有 WebSocket、CLI、API 或定时任务仍在执行。"""
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM agent_runs WHERE thread_id = ? AND status = 'running' LIMIT 1",
                (thread_id,),
            ).fetchone()
        return row is not None

    def has_running_runs(self) -> bool:
        """判断是否存在运行中的持久化任务。"""
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM agent_runs WHERE status = 'running' LIMIT 1"
            ).fetchone()
        return row is not None

    def delete_runs_for_thread(self, thread_id: str) -> int:
        """删除会话关联的全部运行记录；Checkpointer 由服务层单独清理。"""
        self.initialize()
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM agent_runs WHERE thread_id = ?", (thread_id,)
            )
            return cursor.rowcount

    def clear_runs(self) -> int:
        """清空运行记录，保留任务定义与会话 Checkpointer。"""
        self.initialize()
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM agent_runs")
            return cursor.rowcount

    @staticmethod
    def _serialize_run(row: sqlite3.Row) -> dict:
        """把 SQLite 行转成 API 可直接返回的字典。"""
        result = dict(row)

        # 避免 API 字段名暴露数据库内部命名。
        result["trigger"] = result.pop("trigger_source")
        for key in ("started_at", "finished_at"):
            timestamp = result[key]

            # 未结束的 run 没有 finished_at；其他时间戳统一返回带时区字符串。
            result[key] = (
                datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
                if timestamp is not None
                else None
            )
        return result


# ============================================================
# 2. 进程级共享存储实例
# ============================================================

# 服务、同步 runner 与定时任务共用同一个轻量存储对象；每次 SQL 仍各自开关连接。
_store: RuntimeStore | None = None

# 双重检查锁只保护惰性创建，不把数据库连接本身跨线程共享。
_store_lock = threading.Lock()


def get_runtime_store() -> RuntimeStore:
    """获取进程级运行数据存储。

    数据库放在 LangGraph Checkpointer 同目录，并使用独立文件名与连接。
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                path = RUNTIME.sqlite_path.expanduser().resolve().with_name("runtime.sqlite3")
                _store = RuntimeStore(path)
    return _store
