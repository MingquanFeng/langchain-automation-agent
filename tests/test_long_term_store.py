"""
长期记忆 Store 与命名空间的回归测试。

本文件守住阶段 9 第一项的关键边界：
1. SQLite Store 关闭并重开后数据仍然存在。
2. 长期记忆数据库不能被同机其他用户直接读取。
3. MEMORY_BACKEND=memory 时不应创建持久化数据库。
4. build_agent() 未显式注入 Store 时必须走默认工厂。
5. 同一用户和项目的命名空间在重启后保持稳定。
6. 不同用户或不同项目不能共享项目级记忆。
7. namespace 不应直接暴露用户身份或本机绝对路径。

所有 SQLite 用例都使用 pytest 提供的 tmp_path，测试结束后由 pytest
清理，不会写入项目 data/store.sqlite 或桌面应用支持目录。
"""

from __future__ import annotations

# Path：给测试辅助函数标注临时数据库路径类型。
from pathlib import Path

# datetime：用固定 UTC 时间验证过期边界，不依赖测试运行当天。
from datetime import datetime, timedelta, timezone

# SimpleNamespace：构造只包含测试所需字段的最小 RUNTIME 替身。
from types import SimpleNamespace

# pytest：使用 fixture、monkeypatch、tmp_path 和异常断言。
import pytest

# 两种 Store 实现分别用于验证持久化分支和显式内存分支。
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore

# 导入模块对象是为了替换 graph.py 中运行时读取的 RUNTIME、
# _build_store 和 create_agent，而不是只替换本测试文件里的名字。
import agent.graph as graph_module
import agent.memory as memory_module

# 命名空间与记录服务是本组隔离测试的直接被测对象。
from agent.memory import (
    MemoryConflictError,
    MemoryValidationError,
    build_memory_namespaces,
    delete_memory_record,
    list_memory_records,
    save_memory_record,
)


# ============================================================
# 1. 测试脚手架
# ============================================================

# pytest 的全局 fixture 会把默认 Store 换成内存实现，防止测试污染真实数据。
# 在 fixture 生效前保留生产构建函数，供本文件的持久化回归使用。
#
# 这里保留的是函数对象，不是 Store 实例；每个测试仍会创建自己的连接，
# 不会在测试之间共享数据库状态。
_BUILD_STORE = graph_module._build_store


def _sqlite_store(monkeypatch, path: Path) -> SqliteStore:
    """
    绕过全局测试隔离，直接构建一个临时 SQLite Store。

    monkeypatch 会在当前测试结束后恢复 graph_module.RUNTIME，避免这个
    最小配置对象影响其他测试。path 必须来自 tmp_path 的子路径。
    """
    # _build_store() 当前只需要 memory_backend 与 store_path，
    # 不必实例化包含大量无关字段的完整 RuntimeConfig。
    monkeypatch.setattr(
        graph_module,
        "RUNTIME",
        SimpleNamespace(memory_backend="sqlite", store_path=path),
    )

    # 使用前面保留的生产函数，避免 tests/conftest.py 的自动隔离 fixture
    # 把这次调用替换成 InMemoryStore。
    return _BUILD_STORE()


# ============================================================
# 2. SQLite 持久化与文件权限
# ============================================================

def test_sqlite_store_survives_reopen(monkeypatch, tmp_path):
    """关闭并重新打开数据库后，授权写入的长期记忆仍可读取。"""
    # 数据库与项目目录都位于 pytest 临时目录，绝不会碰真实用户数据。
    path = tmp_path / "store.sqlite"
    namespace = build_memory_namespaces(
        user_id="local",
        project_dir=tmp_path / "project",
    ).project

    # 第一个 Store 模拟应用首次启动并保存一条项目级偏好。
    first = _sqlite_store(monkeypatch, path)
    first.put(namespace, "preference", {"language": "zh-CN"})

    # 显式关闭连接模拟 sidecar / 应用进程退出。
    first.conn.close()

    # 使用相同数据库路径创建全新 Store，模拟下一次应用启动。
    second = _sqlite_store(monkeypatch, path)
    item = second.get(namespace, "preference")
    second.conn.close()

    # 能读到相同 value 才说明验证的是磁盘持久化，而不是对象内缓存。
    assert item is not None
    assert item.value == {"language": "zh-CN"}


def test_sqlite_store_file_is_private(monkeypatch, tmp_path):
    """长期记忆主数据库只允许当前系统用户读写。"""
    path = tmp_path / "store.sqlite"
    store = _sqlite_store(monkeypatch, path)
    store.conn.close()

    # 屏蔽文件类型等高位，只比较 rwx 权限位：0600 = 当前用户可读写，
    # 同组用户和其他用户均无权限。
    assert path.stat().st_mode & 0o777 == 0o600


# ============================================================
# 3. 后端选择与 Agent 装配
# ============================================================

def test_memory_backend_can_stay_ephemeral(monkeypatch):
    """显式选择 memory 后端时仍可获得纯内存 Store。"""
    # memory 分支不会访问 store_path，所以替身无需提供该字段。
    monkeypatch.setattr(
        graph_module,
        "RUNTIME",
        SimpleNamespace(memory_backend="memory"),
    )

    # 这条回归保证开发者仍可显式选择无磁盘副作用的运行方式。
    assert isinstance(_BUILD_STORE(), InMemoryStore)


def test_build_agent_uses_default_store_factory(monkeypatch):
    """调用方未注入 Store 时，Agent 装配必须使用持久化工厂。"""
    # marker 用来确认 build_agent() 传给 create_agent() 的就是工厂结果。
    marker = object()
    monkeypatch.setattr(graph_module, "_build_store", lambda: marker)

    # 本测试只检查装配参数，不需要真的编译 LangGraph，也不需要模型 API。
    # 用原样返回 kwargs 的替身可以直接观察最终传参。
    monkeypatch.setattr(graph_module, "create_agent", lambda **kwargs: kwargs)

    # model、tools、checkpointer 都显式注入，确保测试不会创建真实依赖；
    # store 故意不传，以覆盖默认分支。
    built = graph_module.build_agent(
        model=object(),
        tools=[],
        checkpointer=object(),
    )

    assert built["store"] is marker


# ============================================================
# 4. 命名空间稳定性、隐私与隔离
# ============================================================

def test_user_and_project_namespaces_are_stable_and_isolated(tmp_path):
    """同一项目身份稳定，不同项目和用户不会串长期记忆。"""
    project_a = tmp_path / "project-a"
    project_a.mkdir()

    # nested/.. 与 project-a 表示同一规范化路径，用来验证 resolve() 规则。
    alias_of_a = project_a / "nested" / ".."
    project_b = tmp_path / "project-b"

    local_a = build_memory_namespaces(
        user_id="local",
        project_dir=project_a,
    )
    local_a_again = build_memory_namespaces(
        user_id="local",
        project_dir=alias_of_a,
    )
    local_b = build_memory_namespaces(
        user_id="local",
        project_dir=project_b,
    )
    other_user_a = build_memory_namespaces(
        user_id="other",
        project_dir=project_a,
    )

    # 同一用户 + 同一规范化路径必须生成完全相同的两个命名空间。
    assert local_a == local_a_again

    # 用户级与项目级不是同一个作用域，不能把项目事实写进全局偏好桶。
    assert local_a.user != local_a.project

    # 同一用户切换项目时，项目级命名空间必须变化。
    assert local_a.project != local_b.project

    # 同一路径切换用户时，用户级和项目级命名空间都必须变化。
    assert local_a.user != other_user_a.user
    assert local_a.project != other_user_a.project


def test_namespaces_do_not_expose_local_identity_or_absolute_path(tmp_path):
    """命名空间只保存不可逆标识，不暴露用户名或本机项目路径。"""
    project = tmp_path / "private-project"
    namespaces = build_memory_namespaces(
        user_id="alice@example.com",
        project_dir=project,
    )
    # 将所有片段拼成一段文本，模拟数据库或诊断日志中的可见内容。
    serialized = "/".join((*namespaces.user, *namespaces.project))

    # 用户原文和项目绝对路径都不应直接出现在命名空间中。
    assert "alice@example.com" not in serialized
    assert str(project.resolve()) not in serialized

    # profile 与 projects 是同一用户根节点下的两个并列分支，
    # 查询用户级记忆时不会因前缀匹配混入项目级数据。
    assert namespaces.user[-1] == "profile"
    assert namespaces.project[-2] == "projects"
    assert namespaces.user[:-1] == namespaces.project[:-2]


def test_same_key_is_isolated_between_projects(monkeypatch, tmp_path):
    """相同用户在两个项目保存同名 key 时，各自读取自己的值。"""
    store = _sqlite_store(monkeypatch, tmp_path / "store.sqlite")
    project_a = build_memory_namespaces(
        user_id="local",
        project_dir=tmp_path / "project-a",
    ).project
    project_b = build_memory_namespaces(
        user_id="local",
        project_dir=tmp_path / "project-b",
    ).project

    # 两条记录故意使用相同 key；唯一隔离边界是 namespace。
    store.put(project_a, "preference", {"formatter": "ruff"})
    store.put(project_b, "preference", {"formatter": "black"})

    # 分别按项目命名空间读取，并在关闭连接前保存返回对象。
    item_a = store.get(project_a, "preference")
    item_b = store.get(project_b, "preference")
    store.conn.close()

    # 两边都存在且内容不同，证明不是只比较了 namespace 元组，
    # 而是实际经过 SqliteStore 完成了写入和读取隔离。
    assert item_a is not None
    assert item_b is not None
    assert item_a.value == {"formatter": "ruff"}
    assert item_b.value == {"formatter": "black"}


def test_empty_memory_user_id_is_rejected(tmp_path):
    """空用户身份不能退化成所有人的共享命名空间。"""
    # ^...$ 要求完整错误信息匹配，防止以后把空身份静默改成默认公共桶。
    with pytest.raises(ValueError, match="^memory user id 不能为空$"):
        build_memory_namespaces(user_id="  ", project_dir=tmp_path)


# ============================================================
# 5. 记录 CRUD 与显式保存边界
# ============================================================

def _use_memory_context(monkeypatch, project_dir: Path) -> None:
    """让记录服务在测试项目和本地测试用户的命名空间中运行。"""
    monkeypatch.setattr(
        memory_module.config,
        "RUNTIME",
        SimpleNamespace(memory_user_id="test-user"),
    )
    monkeypatch.setattr(memory_module.config, "DATA_DIR", project_dir)


def test_memory_records_can_be_created_updated_listed_and_deleted(monkeypatch, tmp_path):
    """显式记忆入口完整支持查看、新建、修改和删除。"""
    _use_memory_context(monkeypatch, tmp_path / "project")
    store = InMemoryStore()

    created = save_memory_record(
        store,
        scope="project",
        memory_kind="preference",
        content="  默认使用简体中文  ",
        source="settings",
    )
    assert created["id"].startswith("mem_")
    assert created["content"] == "默认使用简体中文"
    assert created["created_source"] == "settings"
    assert created["updated_source"] == "settings"
    assert created["revision"] == 1
    assert created["expiration_policy"] == "never"

    updated = save_memory_record(
        store,
        scope="project",
        memory_kind="fact",
        content="项目使用 Python 3.13",
        memory_id=created["id"],
        source="agent",
        expiration_policy="90_days",
        expected_revision=created["revision"],
    )
    assert updated["id"] == created["id"]
    assert updated["memory_type"] == "fact"
    assert updated["created_source"] == "settings"
    assert updated["updated_source"] == "agent"
    assert updated["revision"] == 2
    assert updated["expiration_policy"] == "90_days"
    assert updated["expires_at"] is not None

    listed = list_memory_records(store, scope="project")
    assert listed["items"] == [updated]
    assert listed["has_more"] is False

    assert delete_memory_record(
        store,
        scope="project",
        memory_id=created["id"],
        expected_revision=updated["revision"],
    ) is True
    assert delete_memory_record(
        store,
        scope="project",
        memory_id=created["id"],
        expected_revision=updated["revision"],
    ) is False


def test_user_and_project_record_lists_do_not_mix(monkeypatch, tmp_path):
    """用户级查询与项目级查询只返回各自精确命名空间的数据。"""
    _use_memory_context(monkeypatch, tmp_path / "project")
    store = InMemoryStore()
    user_record = save_memory_record(
        store,
        scope="user",
        memory_kind="preference",
        content="所有项目默认用中文",
        source="settings",
    )
    project_record = save_memory_record(
        store,
        scope="project",
        memory_kind="fact",
        content="当前项目使用 SQLite",
        source="settings",
    )

    assert list_memory_records(store, scope="user")["items"] == [user_record]
    assert list_memory_records(store, scope="project")["items"] == [project_record]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scope": "all", "memory_kind": "fact", "content": "x"}, "作用域"),
        ({"scope": "user", "memory_kind": "conversation", "content": "x"}, "类型"),
        ({"scope": "user", "memory_kind": "fact", "content": "   "}, "不能为空"),
    ],
)
def test_memory_record_validation_rejects_unsafe_shapes(
    monkeypatch,
    tmp_path,
    kwargs,
    message,
):
    """不允许公共作用域、整段对话类别或空内容进入 Store。"""
    _use_memory_context(monkeypatch, tmp_path / "project")
    with pytest.raises(MemoryValidationError, match=message):
        save_memory_record(InMemoryStore(), source="settings", **kwargs)


def test_expired_records_are_hidden_and_lazily_deleted(monkeypatch, tmp_path):
    """到期记录不再出现在列表中，并从 Store 惰性清理。"""
    _use_memory_context(monkeypatch, tmp_path / "project")
    store = InMemoryStore()
    start = datetime(2030, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(memory_module, "_utc_now", lambda: start)

    created = save_memory_record(
        store,
        scope="project",
        memory_kind="fact",
        content="临时迁移窗口持续 30 天",
        source="settings",
        expiration_policy="30_days",
    )
    assert created["expires_at"] == (start + timedelta(days=30)).isoformat()

    # 跨过截止时间后，业务列表先隐藏记录，再执行实际删除。
    monkeypatch.setattr(
        memory_module,
        "_utc_now",
        lambda: start + timedelta(days=31),
    )
    assert list_memory_records(store, scope="project")["items"] == []
    namespace = memory_module.current_memory_namespaces().project
    assert store.get(namespace, created["id"]) is None


def test_stale_revision_and_duplicate_content_are_rejected(monkeypatch, tmp_path):
    """重复新建和基于旧修订号的覆盖都返回明确冲突。"""
    _use_memory_context(monkeypatch, tmp_path / "project")
    store = InMemoryStore()
    created = save_memory_record(
        store,
        scope="project",
        memory_kind="preference",
        content="默认生成 Markdown 报告",
        source="settings",
    )

    with pytest.raises(MemoryConflictError, match="已经存在"):
        save_memory_record(
            store,
            scope="project",
            memory_kind="preference",
            content="默认生成 Markdown 报告",
            source="agent",
        )

    updated = save_memory_record(
        store,
        scope="project",
        memory_kind="preference",
        content="默认生成 HTML 报告",
        memory_id=created["id"],
        source="agent",
        expected_revision=created["revision"],
    )
    assert updated["revision"] == 2

    second = save_memory_record(
        store,
        scope="project",
        memory_kind="preference",
        content="默认生成纯文本报告",
        source="settings",
    )
    with pytest.raises(MemoryConflictError, match="已经存在"):
        save_memory_record(
            store,
            scope="project",
            memory_kind="preference",
            content=updated["content"],
            memory_id=second["id"],
            source="settings",
            expected_revision=second["revision"],
        )

    with pytest.raises(MemoryConflictError, match="刷新后重试"):
        save_memory_record(
            store,
            scope="project",
            memory_kind="preference",
            content="用旧表单覆盖",
            memory_id=created["id"],
            source="settings",
            expected_revision=created["revision"],
        )

    with pytest.raises(MemoryConflictError, match="刷新后重试"):
        delete_memory_record(
            store,
            scope="project",
            memory_id=created["id"],
            expected_revision=created["revision"],
        )
