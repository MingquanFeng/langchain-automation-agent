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

Store 与 Checkpointer 的职责不同：
- Checkpointer 按 thread_id 保存单个会话的消息和执行状态；
- Store 按 namespace + key 保存多个 thread 可以共享的长期数据。

本模块只定义「记忆应该放在哪个命名空间」，不负责：
- 从对话中自动提取事实；
- 判断一段内容是否应该长期保存；
- 调用 store.put() 写入数据；
- 提供查看、修改或删除记忆的用户界面。

保持这个边界很重要：接入持久化 Store 不等于允许系统把整段对话
自动复制成长期记忆。具体写入策略由后续显式记忆入口负责。
"""

from __future__ import annotations

# hashlib：使用 SHA-256 把用户身份和项目路径转换为稳定标识。
# 这里只需要确定性哈希，不需要额外第三方依赖。
import hashlib

# dataclass：声明只读的数据容器，集中返回两种命名空间。
from dataclasses import dataclass

# Path：展开用户目录、解析相对路径和符号链接，得到规范化项目路径。
from pathlib import Path

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
# ("memories", "v1", "users", "user_<哈希>")
#
# 项目级：
# ("memories", "v1", "users", "user_<哈希>",
#  "projects", "project_<哈希>")
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

    # 用户级命名空间只依赖 user_id，所以同一用户切换项目后仍保持一致。
    user_namespace = (
        *MEMORY_NAMESPACE_ROOT,
        "users",
        _opaque_id("user", normalized_user_id),
    )

    # 项目级命名空间嵌套在用户级命名空间下：
    # - 同一用户、不同路径：project 摘要不同；
    # - 不同用户、同一路径：user 前缀不同；
    # 两种情况都不会读到对方的数据。
    project_namespace = (
        *user_namespace,
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
