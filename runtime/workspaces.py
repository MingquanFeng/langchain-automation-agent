"""解析桌面宿主持久化的工作区清单，并拒绝清单之外的路径。"""

from __future__ import annotations

from pathlib import Path

from agent.config import APP_HOME, DATA_DIR


def _saved_workspaces() -> tuple[Path, ...]:
    """读取 Tauri 维护的目录清单，并再次规范化、确认目录仍然存在。"""
    if APP_HOME is None:
        return ()
    settings = APP_HOME / "working-directory.txt"
    try:
        entries = settings.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    workspaces = []
    for entry in entries:
        try:
            path = Path(entry).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if path.is_dir() and path not in workspaces:
            workspaces.append(path)
    return tuple(workspaces)


def resolve_workspace(requested: str | None, *, managed: bool) -> Path:
    """返回宿主允许的规范化目录；独立服务继续固定使用 DATA_DIR。"""
    if not managed:
        if requested is not None:
            try:
                candidate = Path(requested).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ValueError("工作目录无效或已不存在。") from exc
            if candidate != DATA_DIR.resolve():
                raise ValueError("独立服务不能切换配置之外的工作目录。")
        return DATA_DIR.resolve()

    allowed = _saved_workspaces()
    if not allowed:
        raise ValueError("桌面宿主没有可用的已确认工作目录。")
    if requested is None:
        return allowed[0]
    try:
        candidate = Path(requested).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("工作目录无效或已不存在。") from exc
    if not candidate.is_dir():
        raise ValueError("工作目录必须是已存在的文件夹。")
    if candidate not in allowed:
        raise ValueError("工作目录不在桌面宿主已确认的目录列表中。")
    return candidate
