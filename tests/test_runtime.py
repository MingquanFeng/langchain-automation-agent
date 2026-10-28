"""
命令行入口的测试。

覆盖两个曾经失效的命令行开关：
- --no-schedule：调度器开关
- --reload：开发模式热重载

这些测试要守住的是「命令行的意图真的传到了运行配置里」，
而不是中间某个环节恰好没出问题。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------- 命令行覆盖通道 ----------


def test_cli_override_beats_env(monkeypatch):
    """命令行覆盖优先于普通环境变量。"""
    from agent.config import _env, set_cli_override

    monkeypatch.setenv("SCHEDULER_ENABLED", "1")
    monkeypatch.delenv("__CLI_SCHEDULER_ENABLED__", raising=False)

    # 没有覆盖时读环境变量
    assert _env("SCHEDULER_ENABLED", "1") == "1"

    set_cli_override("SCHEDULER_ENABLED", "0")
    try:
        assert _env("SCHEDULER_ENABLED", "1") == "0"
    finally:
        monkeypatch.delenv("__CLI_SCHEDULER_ENABLED__", raising=False)


def test_cli_override_survives_config_reload(monkeypatch):
    """回归防护：--no-schedule 曾经完全失效。

    原因：config.py 用 load_dotenv(override=True)，
    .env 里的 SCHEDULER_ENABLED=1 会盖掉命令行设进去的值，
    结果 reload 之后 RUNTIME.scheduler_enabled 仍然是 True，
    调度器照常启动。

    现在命令行覆盖走独立前缀的键，
    .env 碰不到它，reload 也不会清掉它。
    """
    import agent.config as config

    # 模拟 .env 里那行 SCHEDULER_ENABLED=1
    monkeypatch.setenv("SCHEDULER_ENABLED", "1")
    monkeypatch.delenv("__CLI_SCHEDULER_ENABLED__", raising=False)

    config.set_cli_override("SCHEDULER_ENABLED", "0")
    try:
        importlib.reload(config)
        assert config.RUNTIME.scheduler_enabled is False
    finally:
        monkeypatch.delenv("__CLI_SCHEDULER_ENABLED__", raising=False)
        importlib.reload(config)


# ---------- --reload 的启动方式 ----------


def test_reload_mode_passes_import_string(monkeypatch):
    """回归防护：--reload 曾经直接把服务对象交给 uvicorn。

    uvicorn 开启 reload（或 workers）时要求应用以
    「模块:属性」的导入字符串传入，传对象它会直接 sys.exit(1)，
    命令行表现为「启动一下就没了」。
    """
    import runtime.__main__ as entry

    captured: dict = {}

    def fake_run(target, **kwargs):
        captured["target"] = target
        captured["reload"] = kwargs.get("reload")

    monkeypatch.setattr(entry.uvicorn, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["runtime", "--reload"])

    entry.main()

    assert captured["reload"] is True
    assert captured["target"] == "runtime.server:app"


def test_normal_mode_passes_app_object(monkeypatch):
    """不开 reload 时仍然直接传应用对象，不走导入字符串。"""
    import runtime.__main__ as entry

    captured: dict = {}

    def fake_run(target, **kwargs):
        captured["target"] = target
        captured["reload"] = kwargs.get("reload")

    monkeypatch.setattr(entry.uvicorn, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["runtime"])

    entry.main()

    assert captured["reload"] is False
    assert captured["target"] is not "runtime.server:app"
