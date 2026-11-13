from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


DESKTOP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DESKTOP_DIR.parent
BUILD_DIR = DESKTOP_DIR / "build" / "pyinstaller"
OUTPUT_DIR = BUILD_DIR / "dist"
TARGET_DIR = DESKTOP_DIR / "src-tauri" / "binaries"
ENTRY_POINT = DESKTOP_DIR / "runtime_sidecar.py"


def fail(message: str) -> None:
    raise SystemExit(f"桌面运行时打包失败：{message}")


def find_python() -> Path:
    configured = os.environ.get("AUTOMATION_AGENT_PYTHON", "").strip()
    python = Path(configured).expanduser() if configured else Path(sys.executable)
    if not python.is_file():
        fail(f"Python 解释器不存在：{python}")
    # Keep venv interpreter symlinks intact: resolve() follows a venv's Python
    # symlink and would bypass that venv's site-packages.
    return python.absolute()


def find_rustc() -> str:
    rustc = shutil.which("rustc")
    if rustc:
        return rustc
    candidate = Path.home() / ".cargo" / "bin" / "rustc"
    if candidate.is_file():
        return str(candidate)
    fail("找不到 rustc；请安装 Rust stable 工具链。")


def main() -> None:
    if sys.platform != "darwin":
        fail("当前只配置了 macOS sidecar 打包目标。")

    python = find_python()
    check = subprocess.run(
        [
            str(python),
            "-c",
            "import fastapi, uvicorn, langchain_deepseek, langchain_openai",
        ],
        capture_output=True,
        text=True,
    )
    if check.returncode:
        fail(
            "所选 Python 环境缺少项目运行依赖。请使用已安装项目依赖的解释器，"
            "并先安装 desktop/requirements-build.txt。\n"
            f"{check.stderr.strip()}"
        )

    pyinstaller_check = subprocess.run(
        [str(python), "-m", "PyInstaller", "--version"],
        capture_output=True,
        text=True,
    )
    if pyinstaller_check.returncode:
        fail(
            "所选 Python 环境尚未安装 PyInstaller；请执行 "
            f"\"{python}\" -m pip install -r desktop/requirements-build.txt。"
        )

    target = subprocess.check_output(
        [find_rustc(), "--print", "host-tuple"], text=True
    ).strip()
    if target not in {"aarch64-apple-darwin", "x86_64-apple-darwin"}:
        fail(f"暂不支持当前 Rust 构建目标：{target}")

    output_binary = OUTPUT_DIR / "automation-agent-runtime"
    args = [
        str(python),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        "automation-agent-runtime",
        "--paths",
        str(PROJECT_ROOT),
        "--add-data",
        f"{PROJECT_ROOT / 'frontend' / 'workbench.html'}:frontend",
        "--add-data",
        f"{PROJECT_ROOT / 'frontend' / 'automation-agent-logo.png'}:frontend",
        "--hidden-import",
        "langchain_deepseek",
        "--hidden-import",
        "langchain_openai",
        "--copy-metadata",
        "langchain-deepseek",
        "--copy-metadata",
        "langchain-openai",
        "--distpath",
        str(OUTPUT_DIR),
        "--workpath",
        str(BUILD_DIR / "work"),
        "--specpath",
        str(BUILD_DIR / "spec"),
        str(ENTRY_POINT),
    ]
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    target_binary = TARGET_DIR / f"automation-agent-runtime-{target}"
    shutil.copy2(output_binary, target_binary)
    target_binary.chmod(0o755)
    print(f"已生成 Tauri sidecar：{target_binary}")


if __name__ == "__main__":
    main()
