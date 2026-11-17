"""
显式长期记忆工具。

主要职责：
1. 让 Agent 查看当前用户或当前项目已经保存的长期记忆。
2. 仅在用户明确要求时，新建或修改一条偏好 / 事实。
3. 仅在用户明确要求时，删除指定长期记忆。

安全边界：
- 本模块不分析聊天历史，也不自动提取记忆；
- save_memory 和 delete_memory 都是持久化副作用，必须经过逐次审批；
- list_memories 只读取当前用户或当前项目的命名空间；
- ToolRuntime 由 LangGraph 注入，模型看不到也不能伪造 Store 对象。
"""

from __future__ import annotations

# json：把结构化记录转换为模型容易稳定读取的文本结果。
import json

# ToolRuntime：工具执行时由 LangGraph 自动注入，其中包含当前 Agent 的 Store。
from langchain.tools import ToolRuntime

# @tool：把普通 Python 函数包装成模型可调用工具。
from langchain_core.tools import tool

# AgentContext：把 ToolRuntime 的上下文泛型标成项目真实类型，
# 避免 Pydantic 在序列化工具运行参数时把它误判为只能是 None。
from ..context import AgentContext

# 记忆服务函数统一处理 namespace、输入限制和 Store CRUD，
# 工具层只负责把参数和结果转换成对模型友好的文本。
from ..memory import (
    MemoryValidationError,
    delete_memory_record,
    list_memory_records,
    save_memory_record,
)


# ============================================================
# 1. Store 注入检查
# ============================================================

def _runtime_store(runtime: ToolRuntime[AgentContext, dict]):
    """读取 LangGraph 注入的 Store；未装配 Store 时给出明确错误。"""
    store = runtime.store
    if store is None:
        raise RuntimeError("当前 Agent 未配置长期记忆 Store")
    return store


# ============================================================
# 2. 查看长期记忆
# ============================================================

@tool(parse_docstring=True)
def list_memories(
    runtime: ToolRuntime[AgentContext, dict],
    scope: str = "project",
) -> str:
    """查看已经由用户明确保存的长期记忆，不要用它推断或自动新增记忆。

    Args:
        scope: project 表示仅当前项目，user 表示该用户跨项目共享。
    """
    try:
        result = list_memory_records(
            _runtime_store(runtime),
            scope=scope,
            limit=50,
        )
    except MemoryValidationError as exc:
        return f"无法查看长期记忆：{exc}"

    if not result["items"]:
        return f"{scope} 作用域尚未保存长期记忆。"
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================
# 3. 新建或修改长期记忆
# ============================================================

@tool(parse_docstring=True)
def save_memory(
    content: str,
    runtime: ToolRuntime[AgentContext, dict],
    scope: str = "project",
    memory_type: str = "preference",
    memory_id: str = "",
) -> str:
    """仅在用户明确要求“记住”时，新建或修改一条跨会话偏好或事实。

    Args:
        content: 一条简短、独立的偏好或事实；不要传整段对话或长文档。
        scope: project 只在当前项目复用；user 可在该用户的所有项目复用。
        memory_type: preference 表示用户偏好，fact 表示明确事实。
        memory_id: 修改已有记忆时传 list_memories 返回的 ID；新建时留空。
    """
    try:
        record = save_memory_record(
            _runtime_store(runtime),
            scope=scope,
            memory_kind=memory_type,
            content=content,
            memory_id=memory_id or None,
        )
    except MemoryValidationError as exc:
        # 参数错误不会通过抛异常触发三次工具重试；直接告诉模型如何修正。
        return f"长期记忆未保存：{exc}"

    action = "已修改" if memory_id else "已保存"
    return f"{action}长期记忆：{json.dumps(record, ensure_ascii=False)}"


# ============================================================
# 4. 删除长期记忆
# ============================================================

@tool(parse_docstring=True)
def delete_memory(
    memory_id: str,
    runtime: ToolRuntime[AgentContext, dict],
    scope: str = "project",
) -> str:
    """仅在用户明确要求遗忘或删除时，删除一条指定的长期记忆。

    Args:
        memory_id: list_memories 返回的长期记忆 ID。
        scope: 该记忆所在作用域，必须是 project 或 user。
    """
    try:
        deleted = delete_memory_record(
            _runtime_store(runtime),
            scope=scope,
            memory_id=memory_id,
        )
    except MemoryValidationError as exc:
        return f"长期记忆未删除：{exc}"

    if not deleted:
        return "指定作用域中没有这条长期记忆。"
    return f"已删除长期记忆：{memory_id}"


# ============================================================
# 5. 导出工具组
# ============================================================

# 只读工具放在前面；写入和删除工具的风险等级在 risk.py 中登记，
# 默认必须通过工作台的单次审批才能真正执行。
MEMORY_TOOLS = [list_memories, save_memory, delete_memory]
