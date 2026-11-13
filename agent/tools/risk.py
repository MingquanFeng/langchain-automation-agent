"""登记工具风险等级，并在工具清单装配时检查登记是否完整。

风险信息集中放在这里，供工具注册表和 TaskGateMiddleware 共用。
这样新增、重命名或删除工具时，只需维护一份风险表；清单遗漏会在
Agent 启动阶段尽早报错，不会因为漏改另一份写工具名单而绕过只读闸门。
"""

from __future__ import annotations

from typing import Iterable, Literal


# Literal 把合法的风险等级写进类型提示：编辑器和静态类型检查器可以
# 提前发现拼写错误。运行时仍需用下面的 RISK_LEVELS 再检查一次，
# 因为 Python 的类型标注本身不会阻止错误字符串进入字典。
ToolRiskLevel = Literal[
    "read_only",
    "internal_state",
    "workspace_write",
    "external_effect",
]
PermissionDecision = Literal["allow", "require_approval", "deny"]

# RISK_LEVELS 是运行时校验使用的合法值集合。
# frozenset 表示这组允许值定义后不应再被调用方修改。
RISK_LEVELS: frozenset[str] = frozenset({
    "read_only",
    "internal_state",
    "workspace_write",
    "external_effect",
})

# 工具名到风险等级的映射，是权限策略读取风险信息的唯一来源。
#
# read_only：只读取工作区数据，不修改文件，也不向外部发送信息。
# internal_state：只更新当前 Agent 运行中的内部任务状态。
# workspace_write：会在 Agent 工作目录内创建或修改文件。
# external_effect：会向项目外部发送通知等信息，产生外部副作用。
#
# 新增工具时，在工具清单中注册后也要在这里登记；不要再另建一份
# WRITE_TOOLS 名单，否则两份名单可能逐渐不一致。
TOOL_RISK_LEVELS: dict[str, ToolRiskLevel] = {
    # 只读文件工具：可在只读任务中直接使用。
    "list_files": "read_only",
    "read_file": "read_only",
    "read_json": "read_only",

    # 文件写入工具：会修改 data/ 沙箱内的项目数据。
    "write_file": "workspace_write",
    "append_file": "workspace_write",
    "save_json": "workspace_write",

    # 通知工具：可能向 webhook 接收方投递消息。
    "notify": "external_effect",
    "notify_summary": "external_effect",
}

# 中间件提供的内部状态工具不属于 Agent 的业务工具清单，单独登记，
# 避免它们被外部副作用闸门误判为未登记工具。不要在这里登记任意业务工具。
INTERNAL_TOOL_RISK_LEVELS: dict[str, ToolRiskLevel] = {
    "write_todos": "internal_state",
}


def decide_tool_permission(
    tool_name: str | None,
    *,
    allow_write: bool,
) -> PermissionDecision:
    """按工具风险等级给出统一的权限决策。

    返回值分三种：
        allow：工具可以进入执行阶段。
        require_approval：工具有副作用，需为本次调用取得授权。
        deny：工具没有风险登记，按默认拒绝处理。

    `allow_write` 是旧入口使用的整轮授权开关。为了兼容 Python API、
    定时任务和旧版 WebSocket 客户端，显式设为 True 时仍放行工具；
    当前浏览器工作台不发送这个开关，而是通过 `approval_handler` 对有副作用的
    工具逐次确认。未分类工具在没有整轮显式授权时始终拒绝。
    """
    # 旧入口显式授权整轮写操作时，保持原有行为。
    if allow_write:
        return "allow"

    name = tool_name or ""
    risk_level = TOOL_RISK_LEVELS.get(name)
    if risk_level is None:
        risk_level = INTERNAL_TOOL_RISK_LEVELS.get(name)
    if risk_level is None:
        return "deny"
    if risk_level in {"read_only", "internal_state"}:
        return "allow"
    return "require_approval"


def validate_tool_risk_levels(
    tools: Iterable[object],
    risk_levels: dict[str, str] = TOOL_RISK_LEVELS,
) -> None:
    """检查工具清单与风险表是否一一对应。

    参数：
        tools：最终传给 Agent 的工具对象集合；每个对象都应有非空 `name`。
        risk_levels：工具名到风险等级的映射。默认使用本模块的正式登记表；
            暴露此参数也便于调用方单独校验其他清单或验证错误情形。

    校验内容：
        1. 工具都有可用名称，且名称不重复。
        2. 工具清单中的每个名称都登记了风险等级。
        3. 风险表中没有已从工具清单移除或改名的旧条目。
        4. 每个等级都属于 RISK_LEVELS。

    抛出：
        RuntimeError：发现任何登记差异。工具注册表在模块导入时调用本函数，
        因此错误会在启动阶段暴露，而不是等到 Agent 尝试执行工具时才出现。
    """
    # 先取出名字，保留列表用于之后检测重复项。
    names = [getattr(tool, "name", None) for tool in tools]
    if any(not isinstance(name, str) or not name for name in names):
        raise RuntimeError("工具清单中存在没有有效 name 的工具")

    # 转成集合用于比较两份清单；转换前先检查长度，避免集合去重掩盖重复注册。
    tool_names = set(names)
    if len(names) != len(tool_names):
        raise RuntimeError("工具清单中存在重复名称")

    # missing 表示工具已经注册，但忘了声明权限等级。
    # stale 表示工具已被删除或改名，风险表却还留着旧名字。
    missing = tool_names - risk_levels.keys()
    stale = risk_levels.keys() - tool_names

    # 静态类型标注不能防止运行时拼错字符串，所以仍检查每个等级。
    invalid = {
        name: level for name, level in risk_levels.items()
        if level not in RISK_LEVELS
    }

    if missing or stale or invalid:
        details = []
        if missing:
            details.append(f"缺少风险等级: {sorted(missing)}")
        if stale:
            details.append(f"风险表中有未注册工具: {sorted(stale)}")
        if invalid:
            details.append(f"存在未知风险等级: {invalid}")
        raise RuntimeError("工具风险登记不完整；" + "；".join(details))
