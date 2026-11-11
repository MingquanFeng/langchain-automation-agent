"""工具风险登记与权限决策的单元测试。"""

from dataclasses import dataclass

import pytest

from agent.tools import ALL_TOOLS, TOOL_RISK_LEVELS
from agent.tools.risk import decide_tool_permission, validate_tool_risk_levels


def test_every_registered_tool_has_a_risk_level():
    """正式工具集合与风险表应一一对应。"""
    validate_tool_risk_levels(ALL_TOOLS)
    assert {tool.name for tool in ALL_TOOLS} == set(TOOL_RISK_LEVELS)


@pytest.mark.parametrize("tool_name", ["list_files", "read_file", "read_json"])
def test_read_only_tools_are_allowed_without_write_permission(tool_name):
    assert decide_tool_permission(tool_name, allow_write=False) == "allow"


@pytest.mark.parametrize(
    "tool_name",
    ["write_file", "append_file", "save_json", "notify", "notify_summary"],
)
def test_side_effect_tools_require_approval_by_default(tool_name):
    assert decide_tool_permission(tool_name, allow_write=False) == "require_approval"


@pytest.mark.parametrize("tool_name", ["write_file", "notify", "unregistered_tool"])
def test_legacy_write_permission_remains_an_explicit_override(tool_name):
    assert decide_tool_permission(tool_name, allow_write=True) == "allow"


@pytest.mark.parametrize("tool_name", ["unregistered_tool", None])
def test_unregistered_tools_are_denied_without_legacy_override(tool_name):
    assert decide_tool_permission(tool_name, allow_write=False) == "deny"


@dataclass
class _NamedTool:
    name: str


def test_validation_rejects_missing_risk_entry():
    with pytest.raises(RuntimeError, match="缺少风险等级"):
        validate_tool_risk_levels([_NamedTool("new_tool")], {})


def test_validation_rejects_stale_risk_entry():
    with pytest.raises(RuntimeError, match="风险表中有未注册工具"):
        validate_tool_risk_levels([], {"removed_tool": "read_only"})


def test_validation_rejects_duplicate_tool_names():
    with pytest.raises(RuntimeError, match="重复名称"):
        validate_tool_risk_levels([_NamedTool("tool"), _NamedTool("tool")], {"tool": "read_only"})


def test_validation_rejects_unknown_risk_level():
    with pytest.raises(RuntimeError, match="未知风险等级"):
        validate_tool_risk_levels([_NamedTool("tool")], {"tool": "dangerous"})
