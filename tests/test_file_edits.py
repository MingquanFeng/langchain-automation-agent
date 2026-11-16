"""结构化编辑的差异审批、并发校验、备份和恢复测试。

测试把工作区和备份库都替换成 pytest 临时目录，不触碰开发数据。预览与执行分成
两个明确阶段，用外部改写模拟用户审批期间的竞态，确保旧预览不能覆盖新内容。
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.context import AgentContext
from agent.graph import build_agent
from agent.tools import edits as edits_mod
from agent.tools import files as files_mod
from fakes import FakeModel


# ============================================================
# 1. 测试工作区和执行辅助函数
# ============================================================

@pytest.fixture
def edit_workspace(tmp_path, monkeypatch):
    """每项测试使用独立工作区、备份库和预备状态表。"""
    workspace = tmp_path / "workspace"
    backups = tmp_path / "backups"
    workspace.mkdir()
    monkeypatch.setattr(files_mod, "SANDBOX", workspace.resolve())
    monkeypatch.setattr(edits_mod, "BACKUP_DIR", backups.resolve())
    with edits_mod._PREPARED_LOCK:
        edits_mod._PREPARED.clear()
    yield workspace, backups
    with edits_mod._PREPARED_LOCK:
        edits_mod._PREPARED.clear()


def _runtime(context, call_id):
    """构造工具函数只使用的 ToolRuntime 字段替身。"""
    return SimpleNamespace(context=context, tool_call_id=call_id)


def _prepare_edit(context, call_id, *, path="src/app.py", old="旧值", new="新值", replace_all=False):
    return edits_mod.prepare_approval_preview(
        "edit_file",
        call_id,
        {"path": path, "old_text": old, "new_text": new, "replace_all": replace_all},
        context,
    )


# ============================================================
# 2. 预览、批准和备份
# ============================================================

def test_edit_preview_is_side_effect_free_and_approved_edit_creates_backup(edit_workspace):
    """预览不写文件；执行前保留权限和原文备份，结果返回 backup_id。"""
    workspace, backups = edit_workspace
    target = workspace / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("第一行\n旧值\n末行\n", encoding="utf-8")
    target.chmod(0o744)
    context = AgentContext()

    preview = _prepare_edit(context, "edit-1")
    assert preview.error is None
    assert preview.args["path"] == "src/app.py"
    assert "--- a/src/app.py" in preview.args["diff"]
    assert "+新值" in preview.args["diff"] and "-旧值" in preview.args["diff"]
    assert set(preview.args) == {"path", "summary", "diff"}
    assert target.read_text(encoding="utf-8") == "第一行\n旧值\n末行\n"
    assert not backups.exists()

    result = edits_mod.edit_file.func(
        path="src/app.py",
        old_text="旧值",
        new_text="新值",
        replace_all=False,
        runtime=_runtime(context, "edit-1"),
    )
    match = re.search(r"备份 (\d{8}T\d{6}Z-[0-9a-f]{12})", result)
    assert match, result
    assert target.read_text(encoding="utf-8") == "第一行\n新值\n末行\n"
    assert (target.stat().st_mode & 0o777) == 0o744
    backup = backups / f"{match.group(1)}.json"
    assert backup.is_file()
    assert (backup.stat().st_mode & 0o777) == 0o600

    listed = edits_mod.list_file_backups.invoke({"path": "src/app.py"})
    assert match.group(1) in listed and "src/app.py" in listed
    assert "第一行" not in listed  # 列表只给元数据，不泄露完整备份正文。


def test_restore_has_reverse_diff_and_creates_rollback_backup(edit_workspace):
    """恢复先展示反向 diff，覆盖前再备份当前状态，因此恢复也可撤销。"""
    workspace, backups = edit_workspace
    target = workspace / "note.txt"
    target.write_text("版本一\n", encoding="utf-8")
    context = AgentContext()
    preview = _prepare_edit(context, "edit-restore", path="note.txt", old="版本一", new="版本二")
    assert preview.error is None
    edit_result = edits_mod.edit_file.func(
        path="note.txt", old_text="版本一", new_text="版本二", replace_all=False,
        runtime=_runtime(context, "edit-restore"),
    )
    original_backup = re.search(r"备份 (\S+)。", edit_result).group(1)

    restore_preview = edits_mod.prepare_approval_preview(
        "restore_file_backup",
        "restore-1",
        {"backup_id": original_backup},
        context,
    )
    assert restore_preview.error is None
    assert restore_preview.args["backup_id"] == original_backup
    assert "-版本二" in restore_preview.args["diff"]
    assert "+版本一" in restore_preview.args["diff"]
    assert target.read_text(encoding="utf-8") == "版本二\n"

    restore_result = edits_mod.restore_file_backup.func(
        backup_id=original_backup,
        runtime=_runtime(context, "restore-1"),
    )
    assert target.read_text(encoding="utf-8") == "版本一\n"
    rollback = re.search(r"备份 (\S+)。", restore_result).group(1)
    assert rollback != original_backup
    assert len(list(backups.glob("*.json"))) == 2


# ============================================================
# 3. 拒绝、竞态与安全边界
# ============================================================

def test_rejected_agent_edit_shows_diff_but_never_writes_or_backs_up(edit_workspace):
    """真实中间件链把安全 diff 交给审批回调，拒绝后工具 handler 不执行。"""
    workspace, backups = edit_workspace
    target = workspace / "config.txt"
    target.write_text("mode=old\n", encoding="utf-8")
    captured = []

    tool_call = AIMessage(content="", tool_calls=[{
        "name": "edit_file",
        "args": {"path": "config.txt", "old_text": "mode=old", "new_text": "mode=new"},
        "id": "reject-edit",
        "type": "tool_call",
    }])
    agent = build_agent(
        model=FakeModel(messages=iter([tool_call, AIMessage(content="已取消修改")])),
        checkpointer=InMemorySaver(),
    )
    context = AgentContext(allow_write=False)

    def reject(name, call_id, args):
        captured.append((name, call_id, args))
        return False

    context.approval_handler = reject
    agent.invoke(
        {"messages": [HumanMessage(content="改配置")]},
        {"configurable": {"thread_id": "reject-edit"}},
        context=context,
    )
    assert target.read_text(encoding="utf-8") == "mode=old\n"
    assert not backups.exists()
    assert len(captured) == 1
    name, call_id, args = captured[0]
    assert (name, call_id) == ("edit_file", "reject-edit")
    assert "diff" in args and "old_text" not in args and "new_text" not in args
    with edits_mod._PREPARED_LOCK:
        assert edits_mod._PREPARED == {}


def test_file_changed_while_waiting_for_approval_is_not_overwritten(edit_workspace):
    """批准前后的 SHA-256 不一致时拒绝旧预览，且不创建误导性备份。"""
    workspace, backups = edit_workspace
    target = workspace / "race.txt"
    target.write_text("旧值\n", encoding="utf-8")
    context = AgentContext()
    assert _prepare_edit(context, "race", path="race.txt").error is None
    target.write_text("其他进程的新内容\n", encoding="utf-8")

    result = edits_mod.edit_file.func(
        path="race.txt", old_text="旧值", new_text="新值", replace_all=False,
        runtime=_runtime(context, "race"),
    )
    assert "等待批准期间发生了变化" in result
    assert target.read_text(encoding="utf-8") == "其他进程的新内容\n"
    assert not backups.exists()


def test_edit_rejects_ambiguous_sensitive_binary_and_oversized_files(edit_workspace):
    """歧义替换和不可安全编辑的文件在审批前失败，不产生预备状态。"""
    workspace, _ = edit_workspace
    (workspace / "many.txt").write_text("旧值 旧值", encoding="utf-8")
    (workspace / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (workspace / "binary.bin").write_bytes(b"\x00old")
    (workspace / "invalid.txt").write_bytes(b"\xffold")
    (workspace / "large.txt").write_bytes(b"x" * (edits_mod.MAX_EDIT_BYTES + 1))
    context = AgentContext()

    ambiguous = _prepare_edit(context, "ambiguous", path="many.txt")
    assert "出现 2 次" in ambiguous.error
    assert "安全原因" in _prepare_edit(context, "secret", path=".env").error
    assert "二进制" in _prepare_edit(context, "binary", path="binary.bin", old="old").error
    assert "UTF-8" in _prepare_edit(context, "invalid", path="invalid.txt", old="old").error
    assert "编辑上限" in _prepare_edit(context, "large", path="large.txt", old="x").error


def test_backup_ids_and_workspace_ownership_are_validated(edit_workspace, monkeypatch, tmp_path):
    """非法 ID 不能穿越备份目录，切换工作目录后也不能误恢复旧项目。"""
    workspace, _ = edit_workspace
    target = workspace / "owner.txt"
    target.write_text("one", encoding="utf-8")
    context = AgentContext()
    assert _prepare_edit(context, "owner", path="owner.txt", old="one", new="two").error is None
    result = edits_mod.edit_file.func(
        path="owner.txt", old_text="one", new_text="two", replace_all=False,
        runtime=_runtime(context, "owner"),
    )
    backup_id = re.search(r"备份 (\S+)。", result).group(1)

    invalid = edits_mod.prepare_approval_preview(
        "restore_file_backup", "bad-id", {"backup_id": "../../etc/passwd"}, context,
    )
    assert "格式无效" in invalid.error

    other = tmp_path / "other-workspace"
    other.mkdir()
    monkeypatch.setattr(files_mod, "SANDBOX", other.resolve())
    wrong_workspace = edits_mod.prepare_approval_preview(
        "restore_file_backup", "wrong-workspace", {"backup_id": backup_id}, context,
    )
    assert "另一个工作目录" in wrong_workspace.error
