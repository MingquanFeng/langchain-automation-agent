"""受控的本地开发命令工具。

本模块不是通用 Shell：只接受参数数组，并且只允许项目检查、测试、构建与
格式检查的白名单命令。所有调用仍由风险中间件逐次审批。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ..context import AgentContext, TaskCancelledError, check_cancelled
from . import files as files_mod


DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 300
MAX_OUTPUT_BYTES = 64 * 1024
_ALLOWED_NPM_SCRIPTS = frozenset({"test", "lint", "build", "check", "format:check", "format-check"})
_CARGO_SUBCOMMANDS = frozenset({"check", "test", "build", "clippy"})
_NODE_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")


def _redact(value: object) -> str:
    """延迟导入运行时模块，避免 Agent 装配阶段与 runner 形成循环依赖。"""
    from runtime.redaction import redact_sensitive_text

    return redact_sensitive_text(value)


class CommandValidationError(ValueError):
    """命令不属于受控开发白名单。"""


def _workspace_root() -> Path:
    """返回规范化的工作目录；命令及 cwd 永远不能走出这里。"""
    return files_mod.SANDBOX.resolve()


def _resolve_cwd(cwd: str) -> Path:
    root = _workspace_root()
    candidate = (root / (cwd or ".")).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_dir():
        raise CommandValidationError("cwd 必须是当前工作目录内已存在的目录。")
    return candidate


def _manifest_scripts(root: Path) -> set[str]:
    """读取工作目录根部 package.json 的脚本名，格式异常时安全地返回空集合。"""
    manifest = root / "package.json"
    if not manifest.is_file():
        return set()
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return set()
    scripts = payload.get("scripts")
    return set(scripts) if isinstance(scripts, dict) else set()


def _has_python_project(root: Path) -> bool:
    """用常见清单或测试目录识别 Python 项目，兼容未采用 pyproject 的旧项目。"""
    return any((root / name).exists() for name in ("pyproject.toml", "pytest.ini", "setup.py", "tests")) or bool(
        list(root.glob("requirements*.txt"))
    )


def _validate_argv(argv: list[str], cwd: str, timeout_seconds: int) -> tuple[list[str], Path, str]:
    """校验结构化命令并返回规范化 argv、cwd 与命令类别。"""
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item for item in argv):
        raise CommandValidationError("argv 必须是非空字符串数组，不能传 Shell 命令文本。")
    if len(argv) > 32 or any("\x00" in item or "\n" in item or "\r" in item for item in argv):
        raise CommandValidationError("命令参数过长或包含不允许的控制字符。")
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise CommandValidationError(f"timeout_seconds 必须在 1-{MAX_TIMEOUT_SECONDS} 秒之间。")

    resolved_cwd = _resolve_cwd(cwd)
    root = _workspace_root()
    command = argv[0]

    if command == "git":
        allowed = {
            ("status",), ("status", "--short"), ("diff",), ("diff", "--check"),
            ("log",), ("log", "--oneline"),
        }
        if tuple(argv[1:]) not in allowed:
            raise CommandValidationError("只允许 git status、git diff、git diff --check、git log 或 git log --oneline。")
        return argv, resolved_cwd, "git_check"

    if command in {"pytest", sys.executable, "python", "python3"}:
        suffix = argv[1:] if command == "pytest" else argv[1:]
        is_pytest = command == "pytest" or (len(suffix) >= 2 and suffix[:2] == ["-m", "pytest"])
        if not is_pytest:
            raise CommandValidationError("Python 仅允许 pytest 或 python -m pytest。")
        if not _has_python_project(root):
            raise CommandValidationError("当前工作目录未识别为 Python 项目，不能运行 pytest。")
        path_args = suffix if command == "pytest" else suffix[2:]
        for arg in path_args:
            if arg.startswith("-"):
                continue
            candidate = (resolved_cwd / arg).resolve()
            if not candidate.is_relative_to(root):
                raise CommandValidationError("pytest 路径必须位于当前工作目录内。")
        return argv, resolved_cwd, "python_test"

    if command == "npm" and len(argv) == 3 and argv[1] == "run":
        script = argv[2]
        if script not in _ALLOWED_NPM_SCRIPTS or script not in _manifest_scripts(root):
            raise CommandValidationError("npm 只允许 package.json 中的 test、lint、build、check 或格式检查脚本。")
        return argv, resolved_cwd, "node_script"

    if command == "cargo" and len(argv) >= 2:
        if not (root / "Cargo.toml").is_file():
            raise CommandValidationError("当前工作目录未识别为 Rust 项目，不能运行 Cargo。")
        subcommand = argv[1]
        if subcommand in _CARGO_SUBCOMMANDS and all(item != "--release" or subcommand == "build" for item in argv[2:]):
            return argv, resolved_cwd, "rust_check"
        if subcommand == "fmt" and argv[2:] == ["--check"]:
            return argv, resolved_cwd, "rust_format_check"
        raise CommandValidationError("Cargo 只允许 check、test、build、clippy 或 fmt --check。")

    raise CommandValidationError("命令未在当前项目的开发白名单中。")


def prepare_command_preview(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """在请求审批前校验命令，并返回不含凭据的展示参数。"""
    argv = args.get("argv")
    cwd = args.get("cwd", ".")
    timeout_seconds = args.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    try:
        normalized, resolved_cwd, category = _validate_argv(argv, cwd, timeout_seconds)
    except CommandValidationError as exc:
        return {"argv": []}, str(exc)
    return {
        "argv": [_redact(item) for item in normalized],
        "cwd": str(resolved_cwd.relative_to(_workspace_root())) or ".",
        "timeout_seconds": timeout_seconds,
        "category": category,
    }, ""


def _safe_environment() -> dict[str, str]:
    """只继承运行常见开发命令所需环境，不传递 API Key 等应用变量。"""
    # Tauri sidecar 的 PATH 可能不含 Homebrew Node；只补固定系统安装目录，
    # 不扫描项目目录或用户自定义脚本路径。
    node_paths = [path for path in _NODE_BIN_DIRS if Path(path, "npm").is_file()]
    path_parts = [*node_paths, os.environ.get("PATH", os.defpath)]
    environment = {"PATH": os.pathsep.join(path_parts), "HOME": os.environ.get("HOME", "")}
    for name in ("LANG", "LC_ALL", "LC_CTYPE", "TERM"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """先温和终止再兜底强杀，防止测试/构建留下子进程。"""
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except OSError:
            pass


def _run_process(argv: list[str], cwd: Path, timeout_seconds: int, context: AgentContext | None) -> tuple[str, int | None, str]:
    """运行命令并有界收集输出，返回 status、退出码和脱敏后的输出。"""
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=_safe_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=os.name == "posix",
    )
    chunks: list[bytes] = []
    captured = 0
    truncated = False

    def drain() -> None:
        nonlocal captured, truncated
        assert process.stdout is not None
        while chunk := process.stdout.read(4096):
            remaining = MAX_OUTPUT_BYTES - captured
            if remaining > 0:
                chunks.append(chunk[:remaining])
                captured += min(len(chunk), remaining)
            if len(chunk) > remaining:
                truncated = True

    reader = threading.Thread(target=drain, name="command-output-reader", daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout_seconds
    status = "succeeded"
    try:
        while process.poll() is None:
            check_cancelled(context)
            if time.monotonic() >= deadline:
                status = "timed_out"
                _terminate_process_group(process)
                break
            time.sleep(0.05)
    except TaskCancelledError:
        status = "cancelled"
        _terminate_process_group(process)
        raise
    finally:
        if process.poll() is None:
            _terminate_process_group(process)
        reader.join(timeout=2)

    exit_code = process.poll()
    output = _redact(b"".join(chunks).decode("utf-8", errors="replace"))
    if truncated:
        output += "\n[输出已截断至 64 KiB]"
    if status == "succeeded" and exit_code != 0:
        status = "failed"
    return status, exit_code, output


@tool(parse_docstring=True)
def run_command(
    argv: list[str],
    runtime: ToolRuntime[AgentContext],
    cwd: str = ".",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """在当前项目内运行已批准的开发检查命令，不能执行任意 Shell 或安装依赖。

    Args:
        argv: 参数数组，例如 ["python", "-m", "pytest", "tests", "-q"]；不要传 shell 字符串。
        cwd: 当前工作目录内的相对目录，默认项目根目录。
        timeout_seconds: 超时秒数，范围 1-300，默认 60。
    """
    context = getattr(runtime, "context", None)
    call_id = str(getattr(runtime, "tool_call_id", "") or "")
    preview, error = prepare_command_preview({
        "argv": argv, "cwd": cwd, "timeout_seconds": timeout_seconds,
    })
    if error:
        return f"命令未执行：{error}"

    def record_result(status: str, exit_code: int | None) -> None:
        if context is not None:
            context.command_results[call_id] = {"status": status, "exit_code": exit_code}

    # storage 依赖 runtime 包；只在工具真正被执行后导入，避免 graph → tools →
    # runtime → runner → graph 的启动循环。
    from runtime.storage import get_runtime_store

    audit_store = get_runtime_store()
    audit_id = audit_store.start_command_execution(
        run_id=str(getattr(context, "run_id", "") or ""),
        call_id=call_id,
        category=preview["category"],
        argv=json.dumps(preview["argv"], ensure_ascii=False),
        cwd=preview["cwd"],
        timeout_seconds=timeout_seconds,
    )
    try:
        status, exit_code, output = _run_process(
            argv, _resolve_cwd(cwd), timeout_seconds, context,
        )
    except TaskCancelledError:
        audit_store.finish_command_execution(
            audit_id, status="cancelled", exit_code=None, output_summary="用户取消命令",
        )
        record_result("cancelled", None)
        raise
    except OSError as exc:
        message = f"无法启动命令：{type(exc).__name__}: {exc}"
        audit_store.finish_command_execution(
            audit_id, status="failed", exit_code=None, output_summary=message,
        )
        record_result("failed", None)
        return message

    audit_store.finish_command_execution(
        audit_id, status=status, exit_code=exit_code, output_summary=output,
    )
    record_result(status, exit_code)
    label = {"succeeded": "完成", "failed": "失败", "timed_out": "超时"}[status]
    return f"命令{label}（退出码 {exit_code}）：\n{output or '（无输出）'}"


COMMAND_TOOLS = [run_command]
