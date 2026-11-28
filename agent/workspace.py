"""单次运行工作区上下文，供并发工具调用共享只读目录边界。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator


_ACTIVE_WORKSPACE: ContextVar[Path | None] = ContextVar("agent_active_workspace", default=None)


def current_workspace(default: Path) -> Path:
    """返回当前异步任务绑定的工作区；未绑定时沿用应用默认目录。"""
    return _ACTIVE_WORKSPACE.get() or default


@contextmanager
def workspace_scope(path: str | Path | None) -> Iterator[None]:
    """在当前任务上下文临时绑定规范化工作区，并在退出时恢复原值。"""
    if path is None:
        yield
        return
    workspace = Path(path).expanduser().resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("工作目录必须是已存在的文件夹。")
    token = _ACTIVE_WORKSPACE.set(workspace)
    try:
        yield
    finally:
        _ACTIVE_WORKSPACE.reset(token)
