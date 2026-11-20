"""受控开发命令的白名单、审批预览、审计与进程边界测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.context import AgentContext, TaskCancelledError
from agent.tools import commands as commands_mod
from agent.tools import files as files_mod
from agent.tools.edits import prepare_approval_preview


@pytest.fixture
def command_workspace(tmp_path, monkeypatch):
    """每个用例使用独立工作目录，避免读取真实项目清单。"""
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "tests").mkdir()
    monkeypatch.setattr(files_mod, "SANDBOX", workspace.resolve())
    return workspace


def test_preview_accepts_pytest_and_rejects_shell_and_outside_paths(command_workspace):
    preview, error = commands_mod.prepare_command_preview({
        "argv": [sys.executable, "-m", "pytest", "tests", "-q"],
        "cwd": ".",
        "timeout_seconds": 60,
    })
    assert not error
    assert preview["category"] == "python_test"

    _, shell_error = commands_mod.prepare_command_preview({
        "argv": ["sh", "-c", "pytest"], "cwd": ".", "timeout_seconds": 60,
    })
    assert "白名单" in shell_error

    _, path_error = commands_mod.prepare_command_preview({
        "argv": ["pytest", "../outside"], "cwd": ".", "timeout_seconds": 60,
    })
    assert "工作目录内" in path_error


def test_preview_only_allows_declared_node_scripts_and_redacts_arguments(command_workspace):
    (command_workspace / "package.json").write_text(
        '{"scripts":{"test":"vitest run","deploy":"dangerous","format:check":"prettier --check ."}}',
        encoding="utf-8",
    )
    preview, error = commands_mod.prepare_command_preview({
        "argv": ["npm", "run", "format:check"], "cwd": ".", "timeout_seconds": 60,
    })
    assert not error and preview["category"] == "node_script"

    _, error = commands_mod.prepare_command_preview({
        "argv": ["npm", "run", "deploy"], "cwd": ".", "timeout_seconds": 60,
    })
    assert "npm" in error

    preview, error = commands_mod.prepare_command_preview({
        "argv": ["git", "status", "api_key=secret-value"], "cwd": ".", "timeout_seconds": 60,
    })
    assert preview == {"argv": []}
    assert error


def test_event_preview_rejects_invalid_command_before_approval(command_workspace):
    preview = prepare_approval_preview(
        "run_command", "cmd-1", {"argv": ["pip", "install", "x"]}, AgentContext(),
    )
    assert preview.error is not None
    assert "白名单" in preview.error


def test_run_command_records_audited_success(command_workspace, monkeypatch):
    """真实 pytest --version 是安全的白名单命令，审计保存开始与终态。"""
    records = []

    class AuditStore:
        def start_command_execution(self, **kwargs):
            records.append(("start", kwargs))
            return 7

        def finish_command_execution(self, execution_id, **kwargs):
            records.append(("finish", execution_id, kwargs))

    import runtime.storage as storage_mod

    monkeypatch.setattr(storage_mod, "get_runtime_store", lambda: AuditStore())
    context = AgentContext(run_id="run-command-test")
    runtime = SimpleNamespace(context=context, tool_call_id="call-command-test")
    result = commands_mod.run_command.func(
        argv=[sys.executable, "-m", "pytest", "--version"],
        cwd=".",
        timeout_seconds=60,
        runtime=runtime,
    )
    assert "命令完成" in result
    assert records[0][1]["run_id"] == "run-command-test"
    assert records[1][2]["status"] == "succeeded"


def test_process_output_is_bounded_and_redacted(command_workspace):
    status, code, output = commands_mod._run_process(
        [sys.executable, "-c", "print('password=secret-value'); print('x' * 70000)"],
        command_workspace, 10, AgentContext(),
    )
    assert (status, code) == ("succeeded", 0)
    assert "secret-value" not in output
    assert "[输出已截断至 64 KiB]" in output


def test_safe_environment_includes_installed_node_directory(monkeypatch, tmp_path):
    """桌面 sidecar 的精简 PATH 仍能找到固定位置的 Homebrew npm/node。"""
    node_bin = tmp_path / "node-bin"
    node_bin.mkdir()
    (node_bin / "npm").touch()
    monkeypatch.setattr(commands_mod, "_NODE_BIN_DIRS", (str(node_bin),))
    monkeypatch.setattr(commands_mod.os, "environ", {"PATH": "/usr/bin", "HOME": "/tmp/home"})
    environment = commands_mod._safe_environment()
    assert environment["PATH"].split(commands_mod.os.pathsep)[0] == str(node_bin)
    assert environment["HOME"] == "/tmp/home"


def test_process_timeout_and_cancellation_terminate_the_child_group(command_workspace):
    """超时和取消均不让测试子进程留在后台。"""
    status, code, _ = commands_mod._run_process(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        command_workspace, 1, AgentContext(),
    )
    assert status == "timed_out"
    assert code is not None

    context = AgentContext()
    assert context.cancel_event is None
    import threading

    context.cancel_event = threading.Event()
    context.cancel_event.set()
    with pytest.raises(TaskCancelledError):
        commands_mod._run_process(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            command_workspace, 10, context,
        )
