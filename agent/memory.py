"""
长期记忆命名空间（memory.py）。

对应课件：
- courseware/langGraph/03-LangGraph持久化与记忆管理.md
- 第 7.2 节「长期记忆」中的 namespace + key 组织方式。

主要职责：
1. 定义长期记忆命名空间的固定版本和层级结构。
2. 为本地用户生成稳定、不可直接识别的用户标识。
3. 为当前工作目录生成稳定、不可直接识别的项目标识。
4. 同时返回用户级和项目级命名空间，供后续记忆工具选择作用域。
5. 提供长期记忆记录的校验、序列化、查询、写入和删除函数。

Store 与 Checkpointer 的职责不同：
- Checkpointer 按 thread_id 保存单个会话的消息和执行状态；
- Store 按 namespace + key 保存多个 thread 可以共享的长期数据。

本模块负责 Store 数据访问，但不负责：
- 从对话中自动提取事实；
- 判断一段内容是否应该长期保存；
- 在没有用户明确要求时自行调用写入函数；
- 提供查看、修改或删除记忆的用户界面。

保持这个边界很重要：接入持久化 Store 不等于允许系统把整段对话
自动复制成长期记忆。具体写入策略由后续显式记忆入口负责。
"""

from __future__ import annotations

# hashlib：使用 SHA-256 把用户身份和项目路径转换为稳定标识。
# 这里只需要确定性哈希，不需要额外第三方依赖。
import hashlib

# re：校验外部传入的记忆 ID，防止空值或异常长字符串进入 Store key。
import re

# uuid：新建记忆时生成不依赖内容的稳定记录 ID。
import uuid

# dataclass：声明只读的数据容器，集中返回两种命名空间。
from dataclasses import dataclass

# Path：展开用户目录、解析相对路径和符号链接，得到规范化项目路径。
from pathlib import Path

# Any / Literal：描述 Store 数据和两个受限枚举参数。
from typing import Any, Literal

# BaseStore：LangGraph Store 的同步接口类型。
# SqliteStore 和 InMemoryStore 都实现相同的 get / put / search / delete 方法。
from langgraph.store.base import BaseStore

# 导入 config 模块，而不是在这里复制 DATA_DIR / RUNTIME 的取值逻辑。
# 这样命令行覆盖工作目录后，current_memory_namespaces() 仍以统一配置为准。
from . import config


# ============================================================
# 1. 命名空间根节点
# ============================================================

# LangGraph Store 要求 namespace 使用 tuple[str, ...]。
# 元组中的每一段都代表一个层级，类似数据库中的复合分区键。
#
# 当前完整结构如下：
#
# 用户级：
# ("memories", "v1", "users", "user_<哈希>", "profile")
#
# 项目级：
# ("memories", "v1", "users", "user_<哈希>",
#  "projects", "project_<哈希>")
#
# profile 与 projects 是同级分支，而不是让项目命名空间直接以用户记录
# 命名空间为前缀。Store.search() 按前缀查询；使用同级分支可以保证查询
# 用户级记忆时不会把所有项目级记忆也一起返回。
#
# 固定加入 v1 的原因：
# 如果未来调整层级或标识算法，可以新增 v2 并执行显式迁移，
# 不必把新旧结构混在同一个前缀中，也不会误读旧数据。
MEMORY_NAMESPACE_ROOT = ("memories", "v1")


# ============================================================
# 2. 本地身份转不透明标识
# ============================================================

def _opaque_id(kind: str, value: str) -> str:
    """
    把本地身份信息转换成稳定且不直接泄露原文的短标识。

    参数：
        kind：标识类型，例如 user 或 project。
        value：待转换的规范化原文，例如用户 ID 或项目绝对路径。

    返回：
        ``<kind>_<20位十六进制摘要>`` 形式的命名空间片段。

    设计说明：
    - 相同输入始终得到相同输出，因此应用重启后仍能找到原记忆；
    - 数据库命名空间不会直接出现邮箱、用户名或本机绝对路径；
    - kind 前缀让调试时可以区分用户标识与项目标识；
    - 截取 20 个十六进制字符，即保留 80 bit，用于本地命名空间
      隔离已经足够，同时避免命名空间过长。

    注意：
    哈希只是减少原文暴露，不是加密，也不代替数据库文件权限控制。
    """
    # encode("utf-8") 把 Python 字符串转换为 SHA-256 接受的字节。
    # hexdigest() 返回固定长度的十六进制文本，便于作为 namespace 片段。
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]

    # 给摘要加类型前缀，避免两个领域的相同摘要在日志中难以辨认。
    return f"{kind}_{digest}"


# ============================================================
# 3. 命名空间结果模型
# ============================================================

@dataclass(frozen=True)
class MemoryNamespaces:
    """
    当前用户级与项目级长期记忆命名空间。

    frozen=True 表示实例创建后不能重新给字段赋值，防止一次任务执行期间
    意外切换作用域。tuple 本身也不可变，因此整个结果可以安全复用。
    """

    # 用户级作用域：适合跨项目共享、并且用户明确要求记住的偏好。
    # 例如回复语言、日期格式等。具体允许保存哪些内容由后续策略决定。
    user: tuple[str, ...]

    # 项目级作用域：适合只在当前工作目录对应项目中复用的事实。
    # 它以 user 为前缀，因此不同用户即使打开同一路径也不会共享记忆。
    project: tuple[str, ...]


# ============================================================
# 4. 构建指定用户和项目的命名空间
# ============================================================

def build_memory_namespaces(
    *,
    user_id: str,
    project_dir: str | Path,
) -> MemoryNamespaces:
    """
    为一个用户和一个项目生成稳定、分层的命名空间。

    参数：
        user_id：本地记忆主体标识。当前桌面版默认传入 local。
        project_dir：当前文件工具被限制到的项目工作目录。

    返回：
        MemoryNamespaces，其中同时包含 user 与 project 两种作用域。

    异常：
        user_id 去除首尾空白后为空时抛出 ValueError。
        空身份如果被允许，会让原本应隔离的数据落入一个隐式公共用户，
        因此这里选择立即失败。

    项目目录先 resolve，以便同一路径的 ``.``、``..`` 或符号链接
    得到同一个身份；随后只保存哈希，避免数据库暴露本机绝对路径。
    """
    # 去掉配置值两端的空白，避免 "local" 与 " local " 被当成两个用户。
    normalized_user_id = user_id.strip()

    # 不允许空用户标识退化成所有本地数据共享的默认桶。
    if not normalized_user_id:
        raise ValueError("memory user id 不能为空")

    # expanduser()：展开路径中的 ~。
    # resolve()：转成绝对路径并消解 .、.. 和已存在的符号链接。
    # Path.resolve() 默认 strict=False，因此新建但尚不存在的工作目录也能生成身份。
    canonical_project = str(Path(project_dir).expanduser().resolve())

    # 先构造当前用户的公共根节点。它本身只用于拼接，不直接保存记录。
    user_root = (
        *MEMORY_NAMESPACE_ROOT,
        "users",
        _opaque_id("user", normalized_user_id),
    )

    # 用户级记忆放在 profile 分支，只依赖 user_id，
    # 所以同一用户切换项目后仍保持一致。
    user_namespace = (*user_root, "profile")

    # 项目级命名空间与 profile 同属当前用户根节点：
    # - 同一用户、不同路径：project 摘要不同；
    # - 不同用户、同一路径：user 前缀不同；
    # 两种情况都不会读到对方的数据。
    project_namespace = (
        *user_root,
        "projects",
        _opaque_id("project", canonical_project),
    )

    # 使用数据类统一返回，调用方不需要自己记住元组层级。
    return MemoryNamespaces(
        user=user_namespace,
        project=project_namespace,
    )


# ============================================================
# 5. 构建当前运行环境的命名空间
# ============================================================

def current_memory_namespaces() -> MemoryNamespaces:
    """
    返回当前运行配置对应的用户级与项目级命名空间。

    RUNTIME.memory_user_id 来自 MEMORY_USER_ID；DATA_DIR 来自普通环境配置
    或桌面 sidecar 注入的 __CLI_DATA_DIR__。因此桌面版切换工作目录后，
    新启动的 sidecar 会自然落到新的项目命名空间，而 Store 数据库本身
    仍位于应用支持目录中。
    """
    return build_memory_namespaces(
        user_id=config.RUNTIME.memory_user_id,
        project_dir=config.DATA_DIR,
    )


# ============================================================
# 6. 长期记忆记录约束
# ============================================================

# scope 决定一条记录可以在哪些工作目录中复用：
# - user：同一用户的所有项目可见；
# - project：只在当前规范化工作目录对应的项目中可见。
MemoryScope = Literal["user", "project"]

# 当前只允许保存两类明确、短小的长期信息。
# 不提供 conversation / transcript 类型，避免把整段对话无差别落盘。
MemoryKind = Literal["preference", "fact"]

MEMORY_SCOPES: frozenset[str] = frozenset({"user", "project"})
MEMORY_KINDS: frozenset[str] = frozenset({"preference", "fact"})

# 单条内容上限用于防止误把长文档或完整聊天记录保存为记忆。
# 4,000 字符足以容纳一条偏好或事实，同时让列表和模型上下文保持可控。
MAX_MEMORY_CONTENT_CHARS = 4_000

# Store key 不承担展示职责，只作为稳定记录 ID。
# 格式固定为 mem_ + 32 位 UUID 十六进制文本；更新和删除接口也只接受此格式。
MEMORY_ID_PATTERN = re.compile(r"^mem_[0-9a-f]{32}$")


class MemoryValidationError(ValueError):
    """长期记忆输入不符合显式保存规则。"""


def _memory_namespace(scope: str) -> tuple[str, ...]:
    """把外部 scope 转换为当前运行环境的准确 Store namespace。"""
    if scope not in MEMORY_SCOPES:
        raise MemoryValidationError("记忆作用域必须是 user 或 project")
    namespaces = current_memory_namespaces()
    return namespaces.user if scope == "user" else namespaces.project


def _normalize_memory_content(content: str) -> str:
    """清理并校验一条明确的偏好或事实内容。"""
    if not isinstance(content, str):
        raise MemoryValidationError("记忆内容必须是文本")
    normalized = content.strip()
    if not normalized:
        raise MemoryValidationError("记忆内容不能为空")
    if len(normalized) > MAX_MEMORY_CONTENT_CHARS:
        raise MemoryValidationError(
            f"单条记忆不能超过 {MAX_MEMORY_CONTENT_CHARS} 个字符"
        )
    return normalized


def _normalize_memory_kind(memory_kind: str) -> MemoryKind:
    """校验记录类型，拒绝把任意数据类别塞进长期记忆。"""
    if memory_kind not in MEMORY_KINDS:
        raise MemoryValidationError("记忆类型必须是 preference 或 fact")
    return memory_kind  # type: ignore[return-value]


def _normalize_memory_id(memory_id: str) -> str:
    """校验由本模块生成的记录 ID。"""
    if not isinstance(memory_id, str) or not MEMORY_ID_PATTERN.fullmatch(memory_id):
        raise MemoryValidationError("记忆 ID 格式无效")
    return memory_id


def _iso_timestamp(value: object) -> str | None:
    """把 Store 返回的 datetime 转成 JSON 可直接返回的 ISO 文本。"""
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else None


def _serialize_memory_item(item: object, scope: MemoryScope) -> dict[str, Any]:
    """把 LangGraph Item / SearchItem 转成稳定的应用层字典。"""
    value = getattr(item, "value", {})
    if not isinstance(value, dict):
        value = {}
    return {
        "id": str(getattr(item, "key", "")),
        "scope": scope,
        "memory_type": value.get("memory_type", "fact"),
        "content": value.get("content", ""),
        "created_at": _iso_timestamp(getattr(item, "created_at", None)),
        "updated_at": _iso_timestamp(getattr(item, "updated_at", None)),
    }


# ============================================================
# 7. 长期记忆 CRUD
# ============================================================

def list_memory_records(
    store: BaseStore,
    *,
    scope: MemoryScope,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """分页列出当前用户或当前项目的长期记忆。"""
    if not 1 <= limit <= 100:
        raise MemoryValidationError("limit 必须在 1 到 100 之间")
    if offset < 0:
        raise MemoryValidationError("offset 不能小于 0")

    namespace = _memory_namespace(scope)
    # 多取一条用于判断 has_more；API 不需要额外执行 count 查询。
    items = store.search(namespace, limit=limit + 1, offset=offset)
    records = [
        _serialize_memory_item(item, scope)
        for item in items[:limit]
        # 当前命名空间下不设计子命名空间；仍做精确比较，防止未来扩展后
        # 前缀搜索把更深层的数据误展示为当前作用域的记忆。
        if tuple(getattr(item, "namespace", ())) == namespace
    ]
    return {
        "items": records,
        "scope": scope,
        "limit": limit,
        "offset": offset,
        "has_more": len(items) > limit,
    }


def save_memory_record(
    store: BaseStore,
    *,
    scope: MemoryScope,
    memory_kind: MemoryKind,
    content: str,
    memory_id: str | None = None,
) -> dict[str, Any]:
    """新建记忆，或在提供 memory_id 时修改已存在的记忆。"""
    namespace = _memory_namespace(scope)
    normalized_kind = _normalize_memory_kind(memory_kind)
    normalized_content = _normalize_memory_content(content)

    if memory_id is None:
        resolved_id = f"mem_{uuid.uuid4().hex}"
    else:
        resolved_id = _normalize_memory_id(memory_id)
        # 修改必须命中当前作用域的既有记录。这样把 user 记录 ID 误传到
        # project 作用域时不会静默创建一条重复记录。
        if store.get(namespace, resolved_id) is None:
            raise MemoryValidationError("要修改的长期记忆不存在")

    store.put(
        namespace,
        resolved_id,
        {
            "memory_type": normalized_kind,
            "content": normalized_content,
        },
    )
    item = store.get(namespace, resolved_id)
    if item is None:
        raise RuntimeError("长期记忆写入后无法读取")
    return _serialize_memory_item(item, scope)


def delete_memory_record(
    store: BaseStore,
    *,
    scope: MemoryScope,
    memory_id: str,
) -> bool:
    """删除当前作用域中的一条记忆；不存在时返回 False。"""
    namespace = _memory_namespace(scope)
    resolved_id = _normalize_memory_id(memory_id)
    if store.get(namespace, resolved_id) is None:
        return False
    store.delete(namespace, resolved_id)
    return True
