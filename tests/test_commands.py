"""受控开发命令的白名单、审批预览、审计与进程边界测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.context import AgentContext, RepairSession, TaskCancelledError
from agent.middleware.events import EventBridgeMiddleware, _command_result, _repair_error
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
    assert preview["argv"][:3] == [sys.executable, "-I", "-m"]

    _, shell_error = commands_mod.prepare_command_preview({
        "argv": ["sh", "-c", "pytest"], "cwd": ".", "timeout_seconds": 60,
    })
    assert "白名单" in shell_error

    _, path_error = commands_mod.prepare_command_preview({
        "argv": ["pytest", "../outside"], "cwd": ".", "timeout_seconds": 60,
    })
    assert "工作目录内" in path_error


@pytest.mark.parametrize("invalid_first", [True, False])
def test_single_command_blocks_retries_and_resets_each_turn(command_workspace, monkeypatch, invalid_first):
    """首个调用校验失败或执行失败后，换解释器及额外探测均不执行。"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from agent.graph import build_agent
    from fakes import FakeModel

    processes, approvals = [], []

    def run_process(argv, *args):
        processes.append(argv)
        return "failed", 1, "No module named pytest"

    def approve(name, call_id, args):
        approvals.append(call_id)
        return True

    monkeypatch.setattr(commands_mod, "_run_process", run_process)
    first = ["python3", "-m", "-m", "pytest", "--version"] if invalid_first else ["python3", "-m", "pytest", "--version"]
    responses = []
    for turn in range(2):
        for index, (name, args) in enumerate([
            ("run_command", {"argv": first}),
            ("run_command", {"argv": ["python", "-m", "pytest", "--version"]}),
            ("list_files", {}),
        ]):
            responses.append(AIMessage(content="", tool_calls=[{
                "name": name, "args": args, "id": f"once-{turn}-{index}", "type": "tool_call",
            }]))
        responses.append(AIMessage(content="首次尝试失败，已停止。"))
    agent = build_agent(model=FakeModel(messages=iter(responses)), checkpointer=InMemorySaver())
    ctx = AgentContext(allow_write=False, approval_handler=approve)
    for turn in range(2):
        result = agent.invoke(
            {"messages": [HumanMessage(content="只运行一次 python3 -m pytest --version。失败后停止。 ")]},
            {"configurable": {"thread_id": "single-command"}}, context=ctx,
        )
        replies = {m.tool_call_id: m for m in result["messages"] if isinstance(m, ToolMessage)}
        for index in (1, 2):
            assert "禁止继续调用工具" in replies[f"once-{turn}-{index}"].content
        assert len(processes) == (0 if invalid_first else turn + 1)
        assert len(approvals) == len(processes)


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


def test_preview_finds_nearest_nested_python_node_and_rust_projects(command_workspace):
    """在工作区子目录运行命令时，按 cwd 识别最近的项目清单。"""
    python_project = command_workspace / "services" / "python-api"
    (python_project / "tests").mkdir(parents=True)
    (python_project / "pyproject.toml").write_text("[project]\nname = 'python-api'\n", encoding="utf-8")

    python_preview, python_error = commands_mod.prepare_command_preview({
        "argv": [sys.executable, "-m", "pytest", "tests", "-q"],
        "cwd": "services/python-api", "timeout_seconds": 60,
    })
    assert not python_error
    assert python_preview["category"] == "python_test"
    assert python_preview["cwd"] == "services/python-api"

    node_project = command_workspace / "desktop"
    node_project.mkdir()
    (node_project / "package.json").write_text(
        '{"scripts":{"test":"node --test"}}', encoding="utf-8",
    )
    node_preview, node_error = commands_mod.prepare_command_preview({
        "argv": ["npm", "run", "test"], "cwd": "desktop", "timeout_seconds": 60,
    })
    assert not node_error
    assert node_preview["category"] == "node_script"

    rust_project = command_workspace / "native"
    rust_project.mkdir()
    (rust_project / "Cargo.toml").write_text("[package]\nname='native'\nversion='0.1.0'\n", encoding="utf-8")
    rust_preview, rust_error = commands_mod.prepare_command_preview({
        "argv": ["cargo", "test"], "cwd": "native", "timeout_seconds": 60,
    })
    assert not rust_error
    assert rust_preview["category"] == "rust_check"


def test_nested_node_manifest_does_not_fall_back_to_workspace_scripts(command_workspace):
    """子项目清单优先；不能借工作区根清单放行子项目未声明脚本。"""
    (command_workspace / "package.json").write_text(
        '{"scripts":{"test":"node --test"}}', encoding="utf-8",
    )
    nested = command_workspace / "desktop"
    nested.mkdir()
    (nested / "package.json").write_text('{"scripts":{}}', encoding="utf-8")

    _, error = commands_mod.prepare_command_preview({
        "argv": ["npm", "run", "test"], "cwd": "desktop", "timeout_seconds": 60,
    })
    assert "package.json" in error


def test_preview_reports_missing_project_manifests(tmp_path, monkeypatch):
    """没有对应项目清单时，在审批前明确拒绝三类命令。"""
    empty_workspace = tmp_path / "empty-workspace"
    empty_workspace.mkdir()
    monkeypatch.setattr(files_mod, "SANDBOX", empty_workspace.resolve())

    _, python_error = commands_mod.prepare_command_preview({
        "argv": [sys.executable, "-m", "pytest", "--version"], "cwd": ".", "timeout_seconds": 60,
    })
    assert "未识别为 Python 项目" in python_error

    _, node_error = commands_mod.prepare_command_preview({
        "argv": ["npm", "run", "test"], "cwd": ".", "timeout_seconds": 60,
    })
    assert "package.json" in node_error

    _, rust_error = commands_mod.prepare_command_preview({
        "argv": ["cargo", "check"], "cwd": ".", "timeout_seconds": 60,
    })
    assert "未识别为 Rust 项目" in rust_error


def test_preview_reports_missing_node_and_cargo_before_approval(command_workspace, monkeypatch):
    """Node 或 Cargo 不可用时，在审批前给出明确提示。"""
    (command_workspace / "package.json").write_text(
        '{"scripts":{"check":"node --check app.js"}}', encoding="utf-8",
    )
    (command_workspace / "Cargo.toml").write_text(
        "[package]\nname='native'\nversion='0.1.0'\n", encoding="utf-8",
    )
    monkeypatch.setattr(commands_mod.shutil, "which", lambda executable, path=None: None)

    _, node_error = commands_mod.prepare_command_preview({
        "argv": ["npm", "run", "check"], "cwd": ".", "timeout_seconds": 60,
    })
    assert "未找到 node" in node_error

    _, cargo_error = commands_mod.prepare_command_preview({
        "argv": ["cargo", "check"], "cwd": ".", "timeout_seconds": 60,
    })
    assert "未找到 cargo" in cargo_error


def test_event_preview_rejects_invalid_command_before_approval(command_workspace):
    preview = prepare_approval_preview(
        "run_command", "cmd-1", {"argv": ["pip", "install", "x"]}, AgentContext(),
    )
    assert preview.error is not None
    assert "白名单" in preview.error


def test_python_pytest_runs_in_isolated_mode(command_workspace):
    """工作目录的 pytest.py 不能劫持 python -m pytest。"""
    (command_workspace / "pytest.py").write_text(
        "raise SystemExit('workspace module must not run')\n",
        encoding="utf-8",
    )

    normalized, _, category = commands_mod._validate_argv(
        [sys.executable, "-m", "pytest", "--version"], ".", 60,
    )

    assert category == "python_test"
    assert normalized == [sys.executable, "-I", "-m", "pytest", "--version"]
    status, _, output = commands_mod._run_process(
        normalized, command_workspace, 60, AgentContext(),
    )
    assert status == "succeeded"
    assert "workspace module must not run" not in output


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
    assert '"-I", "-m", "pytest"' in result
    assert "工作目录中的 pytest.py 不参与本次模块查找" in result
    assert records[0][1]["run_id"] == "run-command-test"
    assert records[1][2]["status"] == "succeeded"
    assert context.command_results["call-command-test"] == {"status": "succeeded", "exit_code": 0}


def test_repair_session_only_allows_source_and_three_validation_attempts():
    session = RepairSession(
        argv=["npm", "run", "lint"], cwd=".", timeout_seconds=120, output_summary="lint failed",
    )
    context = AgentContext(repair_session=session)

    assert "先完成一次" in _repair_error(context, "run_command", {
        "argv": ["npm", "run", "lint"], "cwd": ".", "timeout_seconds": 120,
    })
    assert "源码" in _repair_error(context, "edit_file", {"path": "package.json"})
    assert not _repair_error(context, "edit_file", {"path": "src/page.tsx"})
    assert not _repair_error(context, "edit_file", {"path": "tests/always-fail.cjs"})

    intentional_failure = RepairSession(
        argv=["npm", "run", "test"], cwd=".", timeout_seconds=60,
        output_summary="QA 固定失败：用于验证三轮上限",
    )
    intentional_context = AgentContext(repair_session=intentional_failure)
    assert "故意失败的 QA 夹具" in _repair_error(
        intentional_context, "edit_file", {"path": "tests/always-fail.cjs"},
    )
    assert "故意失败的 QA 夹具" in _repair_error(
        intentional_context, "edit_file", {"path": "src/page.tsx"},
    )

    session.edited_since_check = True
    assert "只允许一次编辑" in _repair_error(context, "edit_file", {"path": "src/page.tsx"})
    assert "原始命令" in _repair_error(context, "run_command", {
        "argv": ["npm", "run", "build"], "cwd": ".", "timeout_seconds": 120,
    })
    assert not _repair_error(context, "run_command", {
        "argv": ["npm", "run", "lint"], "cwd": ".", "timeout_seconds": 120,
    })
    session.edit_pending = True
    assert "正在等待" in _repair_error(context, "edit_file", {"path": "src/page.tsx"})
    session.edit_pending = False
    session.attempts = 3
    assert "3 次上限" in _repair_error(context, "edit_file", {"path": "src/page.tsx"})


@pytest.mark.parametrize(("name", "args", "session_state"), [
    ("edit_file", {"path": "src/other.js"}, {"edit_pending": True}),
    ("edit_file", {"path": "src/other.js"}, {"edited_since_check": True}),
    ("run_command", {"argv": ["npm", "test"], "cwd": ".", "timeout_seconds": 60}, {"edit_pending": True}),
])
def test_parallel_repair_call_does_not_close_session(monkeypatch, name, args, session_state):
    """同一批次的重复调用被拒绝时，保留首处编辑或待批检查的会话。"""
    from langchain_core.messages import ToolMessage

    session = RepairSession(argv=["npm", "test"], cwd=".", timeout_seconds=60, output_summary="failed")
    for field, value in session_state.items():
        setattr(session, field, value)
    context = AgentContext(allow_write=False, repair_session=session, single_command=False)

    class _Req:
        tool_call = {"name": name, "args": args, "id": "parallel-call"}
        tool = None
        state = {}
        runtime = type("_Runtime", (), {"context": context})()

    events = []
    middleware = EventBridgeMiddleware()
    middleware._emit = events.append

    def must_not_run(request):
        raise AssertionError("并发的第二处编辑不得执行")

    result = middleware.wrap_tool_call(_Req(), must_not_run)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.content
    assert session.terminal_status == ""
    assert events[-1]["repair_status"] is None


def test_repair_policy_violation_closes_session_and_emits_terminal_status(monkeypatch):
    """越权编辑被护栏拒绝后应结束修复，不能继续探测无关文件。"""
    from langchain_core.messages import ToolMessage

    session = RepairSession(argv=["npm", "test"], cwd=".", timeout_seconds=60, output_summary="failed")
    context = AgentContext(allow_write=False, repair_session=session, single_command=False)

    class _Req:
        tool_call = {"name": "edit_file", "args": {"path": "package.json"}, "id": "blocked-edit"}
        tool = None
        state = {}
        runtime = type("_Runtime", (), {"context": context})()

    events = []
    middleware = EventBridgeMiddleware()
    middleware._emit = events.append

    def must_not_run(request):
        raise AssertionError("被修复护栏拒绝的编辑不得执行")

    result = middleware.wrap_tool_call(_Req(), must_not_run)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert session.terminal_status == "policy_blocked"
    assert events[-1]["repair_status"] == "policy_blocked"


def test_repair_stops_before_approval_for_intentional_failure_fixture(monkeypatch):
    """固定失败 QA 样例不能触发编辑审批，也不能靠改文件变绿。"""
    from langchain_core.messages import ToolMessage

    session = RepairSession(
        argv=["npm", "run", "test"], cwd=".", timeout_seconds=60,
        output_summary="QA 固定失败：用于验证三轮上限",
    )
    context = AgentContext(allow_write=False, repair_session=session, single_command=False)

    class _Req:
        tool_call = {"name": "edit_file", "args": {"path": "tests/always-fail.cjs"}, "id": "fixture-edit"}
        tool = None
        state = {}
        runtime = type("_Runtime", (), {"context": context})()

    events = []
    middleware = EventBridgeMiddleware()
    middleware._emit = events.append

    def must_not_run(request):
        raise AssertionError("故意失败的 QA 夹具编辑不得请求审批或执行")

    result = middleware.wrap_tool_call(_Req(), must_not_run)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "故意失败的 QA 夹具" in result.content
    assert session.terminal_status == "policy_blocked"
    assert events[-1]["repair_status"] == "policy_blocked"
    assert "故意失败的 QA 夹具" in events[-1]["repair_message"]


def test_three_failed_repair_checks_stop_real_tool_chain(command_workspace):
    """真实编辑与 pytest 连续失败三轮后，第四次编辑不得申请审批或落盘。"""
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from agent.graph import build_agent
    from fakes import FakeModel

    target = command_workspace / "tests" / "test_probe.py"
    target.write_text("# attempt 0\ndef test_probe():\n    assert False\n", encoding="utf-8")
    argv = [sys.executable, "-m", "pytest", "tests/test_probe.py", "-q"]
    session = RepairSession(argv=argv, cwd=".", timeout_seconds=60, output_summary="failed")
    approvals = []

    def approve(name, call_id, args):
        approvals.append(name)
        return True

    replies = []
    for attempt in range(1, 5):
        replies.append(AIMessage(content="", tool_calls=[{
            "name": "edit_file", "id": f"edit-{attempt}", "type": "tool_call",
            "args": {"path": "tests/test_probe.py", "old_text": f"# attempt {attempt - 1}",
                     "new_text": f"# attempt {attempt}"},
        }]))
        replies.append(AIMessage(content="", tool_calls=[{
            "name": "run_command", "id": f"check-{attempt}", "type": "tool_call",
            "args": {"argv": argv, "cwd": ".", "timeout_seconds": 60},
        }]))
    replies.append(AIMessage(content="三次验证仍失败，停止修复。"))
    agent = build_agent(model=FakeModel(messages=iter(replies)), checkpointer=InMemorySaver())
    agent.invoke(
        {"messages": [HumanMessage(content="修复测试，最多三轮。")]},
        {"configurable": {"thread_id": "three-failed-checks"}},
        context=AgentContext(allow_write=False, repair_session=session, approval_handler=approve),
    )
    assert session.attempts == 3
    assert session.terminal_status == "failed"
    assert approvals == ["edit_file", "run_command"] * 3
    assert target.read_text(encoding="utf-8").startswith("# attempt 3\n")


def test_command_result_falls_back_to_controlled_tool_output():
    result = SimpleNamespace(content="命令失败（退出码 1）：\n诊断", status="success")
    assert _command_result(AgentContext(), "missing", result) == {"status": "failed", "exit_code": 1}

    block_result = SimpleNamespace(content=[{"type": "text", "text": "命令失败（退出码 2）：\n诊断"}], status="success")
    assert _command_result(AgentContext(), "missing", block_result) == {"status": "failed", "exit_code": 2}


def test_process_output_is_bounded_and_redacted(command_workspace):
    status, code, output = commands_mod._run_process(
        [sys.executable, "-c", "print('password=secret-value'); print('x' * 70000)"],
        command_workspace, 10, AgentContext(),
    )
    assert (status, code) == ("succeeded", 0)
    assert "secret-value" not in output
    assert "[输出已截断至 64 KiB]" in output


def test_safe_environment_includes_installed_node_and_cargo_directories(monkeypatch, tmp_path):
    """桌面 sidecar 的精简 PATH 只补入固定 Homebrew 与 rustup 工具目录。"""
    node_bin = tmp_path / "node-bin"
    node_bin.mkdir()
    (node_bin / "npm").touch()
    cargo_bin = tmp_path / ".cargo" / "bin"
    cargo_bin.mkdir(parents=True)
    (cargo_bin / "cargo").touch()
    monkeypatch.setattr(commands_mod, "_NODE_BIN_DIRS", (str(node_bin),))
    monkeypatch.setattr(commands_mod.os, "environ", {"PATH": "/usr/bin", "HOME": str(tmp_path)})
    environment = commands_mod._safe_environment()
    assert environment["PATH"].split(commands_mod.os.pathsep)[:2] == [str(node_bin), str(cargo_bin)]
    assert environment["HOME"] == str(tmp_path)


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
