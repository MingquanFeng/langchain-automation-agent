"""带差异预览、乐观并发校验和备份恢复的结构化文本编辑工具。

调用关系：
1. EventBridgeMiddleware 在请求用户批准前调用 `prepare_approval_preview()`，只在内存中
   计算 proposed content 和 unified diff，并按「运行上下文 + tool_call_id」保存预备状态。
2. 用户批准后，`edit_file` / `restore_file_backup` 通过 ToolRuntime 取得同一个上下文和
   call_id，再次核对当前文件摘要。文件在等待批准期间变化时拒绝写入。
3. 真正覆盖前把原文保存到应用数据目录的隐藏备份库；恢复操作也会先备份当前状态，
   因此恢复错版本后仍可再次恢复。

关键边界：
- 只编辑工作目录内的 UTF-8 文本，不读取敏感配置、密钥、二进制或超大文件。
- 审批事件只携带路径、变更统计和截断后的 diff，不重复发送完整 old/new 参数。
- 备份与用户选择的工作目录分开保存，扫描工具不会把备份内容送进模型上下文。
- 这是单进程内的乐观并发保护；摘要不一致时必须重新读取并生成新的编辑请求。
"""

from __future__ import annotations

# datetime：生成 UTC 时间和可排序的备份 ID。
from datetime import datetime, timezone

# dataclass：保存审批和执行之间的内存预备状态。
from dataclasses import dataclass

# difflib：生成用户可审核的 unified diff。
import difflib

# hashlib：对批准时和执行时的文件内容做乐观并发校验。
import hashlib

# json：保存自描述备份元数据和原文。
import json
import logging

# os：设置备份目录和文件权限。
import os

# Path：备份和工作区路径处理。
from pathlib import Path

# re：严格限制外部传入的 backup_id，避免目录穿越。
import re

# threading：多个不同会话可并发准备和执行编辑。
import threading

# uuid：为同一秒创建的多个备份分配唯一 ID。
import uuid

# Any：审批参数来自模型工具调用，运行时仍需逐字段校验。
from typing import Any

# ToolRuntime：注入 tool_call_id 和本次 AgentContext，不暴露给模型 schema。
from langchain.tools import ToolRuntime, tool

# RUNTIME.sqlite_path 始终位于应用数据目录，切换工作目录不会搬走备份。
from agent.config import RUNTIME

# AgentContext：约束 ToolRuntime.context 的真实类型，避免运行时序列化按 None 处理。
from agent.context import AgentContext

# 复用文件工具的沙箱、敏感路径判断和原子写实现，确保安全规则一致。
from agent.tools import files as files_mod
from agent.tools.commands import prepare_command_preview


# ============================================================
# 1. 配置与预备状态
# ============================================================

# 单个可编辑文件上限。模型发来的替换片段远小于这个值，1MB 足够覆盖常见源码，
# 同时避免为了生成 diff 和备份把任意大文件读进内存。
MAX_EDIT_BYTES = 1_000_000

# 审批卡片只展示有限 diff；完整变更仍由 proposed content 在内存中执行。
MAX_DIFF_LINES = 400
MAX_DIFF_CHARS = 20_000

# 每个安装最多保留最近 100 份备份，防止长期编辑无限占用磁盘。
MAX_BACKUPS = 100
MAX_BACKUP_LIST = 50

# 默认备份库位于记忆数据库旁的隐藏目录；测试会替换为 pytest 临时目录。
BACKUP_DIR = Path(RUNTIME.sqlite_path).expanduser().resolve().parent / ".automation-agent-backups"

# ID 只允许程序生成的 UTC 时间 + 12 位十六进制随机串。
BACKUP_ID_RE = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{12}")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApprovalPreview:
    """EventBridge 使用的安全展示参数和可选的预处理错误。"""

    args: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class _PreparedChange:
    """批准后执行所需的不可变快照，不包含在 WebSocket 事件里。"""

    operation: str
    path: str
    target: Path
    expected_sha256: str
    proposed_text: str
    diff: str
    summary: str
    source_backup_id: str | None = None


# (id(AgentContext), tool_call_id) → 预备状态。上下文对象在整次 run 中保持同一身份，
# 即使不同 run 的模型重复使用 call_id，也不会串用预览。
_PREPARED: dict[tuple[int, str], _PreparedChange] = {}
_PREPARED_LOCK = threading.Lock()


# ============================================================
# 2. 文本、差异与备份辅助函数
# ============================================================

def _digest(text: str) -> str:
    """返回 UTF-8 文本的 SHA-256 十六进制摘要。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_editable(path: str) -> tuple[Path | None, str | None, str]:
    """读取可编辑文本；失败时返回给模型可执行的中文说明。"""
    target, err = files_mod._resolve_safe(path)
    if target is None:
        return None, None, err
    if files_mod._is_sensitive_path(path, target):
        return None, None, "出于安全原因，不能编辑敏感配置或密钥文件。"
    if not target.exists():
        return None, None, f"文件不存在：{path}。新建文件请使用 write_file。"
    if target.is_dir():
        return None, None, f"{path} 是目录，不是文件。"
    if target.name in files_mod.BLOCKED_NAMES:
        return None, None, f"{path} 是运行时自己的数据库，不能编辑。"

    try:
        size = target.stat().st_size
        if size > MAX_EDIT_BYTES:
            return None, None, f"文件共 {size} 字节，超过 {MAX_EDIT_BYTES} 字节编辑上限。"
        with target.open("rb") as handle:
            raw = handle.read(MAX_EDIT_BYTES + 1)
    except OSError as exc:
        return None, None, f"无法读取 {path}：{type(exc).__name__}: {exc}"
    if b"\x00" in raw[:8192]:
        return None, None, f"{path} 是二进制文件，不能用文本编辑工具修改。"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, None, f"{path} 不是有效的 UTF-8 文本，不能安全编辑。"
    return target, text, ""


def _replacement(
    current: str,
    old_text: str,
    new_text: str,
    replace_all: bool,
) -> tuple[str | None, str]:
    """在内存中应用精确替换；歧义或无变化时拒绝生成编辑。"""
    if not old_text:
        return None, "old_text 不能为空；请提供能唯一定位修改位置的原文。"
    if old_text == new_text:
        return None, "old_text 与 new_text 相同，没有需要保存的变化。"
    occurrences = current.count(old_text)
    if occurrences == 0:
        return None, "文件中找不到 old_text；请重新读取文件后使用完全一致的原文。"
    if occurrences > 1 and not replace_all:
        return None, (
            f"old_text 在文件中出现 {occurrences} 次。请扩大上下文使它唯一，"
            "或明确设置 replace_all=true。"
        )
    if replace_all:
        return current.replace(old_text, new_text), ""
    return current.replace(old_text, new_text, 1), ""


def _unified_diff(path: str, before: str, after: str) -> tuple[str, str]:
    """生成带统计信息且限制长度的 unified diff。"""
    lines = list(difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    ))
    additions = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    summary = f"新增 {additions} 行，删除 {deletions} 行"

    shown = lines[:MAX_DIFF_LINES]
    diff = "\n".join(line.rstrip("\n") for line in shown)
    truncated = len(lines) > len(shown)
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS]
        truncated = True
    if truncated:
        diff += "\n…（差异过长，审批预览已截断）"
    return diff or "（没有文本差异）", summary


def _backup_path(backup_id: str) -> Path | None:
    """把合法 ID 映射到备份文件；非法 ID 不参与任何文件操作。"""
    if not BACKUP_ID_RE.fullmatch(backup_id):
        return None
    return BACKUP_DIR / f"{backup_id}.json"


def _load_backup(backup_id: str) -> tuple[dict[str, Any] | None, str]:
    """读取并校验备份元数据、工作区归属和内容摘要。"""
    path = _backup_path(backup_id)
    if path is None:
        return None, "backup_id 格式无效。请先调用 list_file_backups 获取可用 ID。"
    if not path.is_file():
        return None, f"找不到备份：{backup_id}"
    try:
        if path.stat().st_size > MAX_EDIT_BYTES * 2:
            return None, f"备份 {backup_id} 异常过大，已拒绝读取。"
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"无法读取备份 {backup_id}：{type(exc).__name__}: {exc}"
    if not isinstance(record, dict):
        return None, f"备份 {backup_id} 格式无效。"
    if record.get("workspace") != str(files_mod.SANDBOX):
        return None, "该备份属于另一个工作目录，不能恢复到当前工作目录。"
    content = record.get("content")
    relative = record.get("path")
    if not isinstance(content, str) or not isinstance(relative, str):
        return None, f"备份 {backup_id} 缺少必要字段。"
    if record.get("before_sha256") != _digest(content):
        return None, f"备份 {backup_id} 内容校验失败，已拒绝恢复。"
    return record, ""


def _prune_backups() -> None:
    """只保留最新 MAX_BACKUPS 个 JSON 备份；清理失败不影响本次编辑。"""
    try:
        backups = sorted(
            BACKUP_DIR.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for stale in backups[MAX_BACKUPS:]:
            stale.unlink(missing_ok=True)
    except OSError:
        pass


def _save_backup(target: Path, before: str, after: str, reason: str) -> str:
    """在覆盖前保存原文，返回后续可用于恢复的 backup_id。"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(BACKUP_DIR, 0o700)
    except OSError:
        pass
    now = datetime.now(timezone.utc)
    backup_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    record = {
        "version": 1,
        "backup_id": backup_id,
        "workspace": str(files_mod.SANDBOX),
        "path": str(target.relative_to(files_mod.SANDBOX)),
        "created_at": now.isoformat(),
        "reason": reason,
        "before_sha256": _digest(before),
        "after_sha256": _digest(after),
        "content": before,
    }
    destination = BACKUP_DIR / f"{backup_id}.json"
    files_mod._atomic_write(
        destination,
        json.dumps(record, ensure_ascii=False, indent=2),
    )
    try:
        os.chmod(destination, 0o600)
    except OSError:
        pass
    _prune_backups()
    return backup_id


# ============================================================
# 3. 审批预览的准备、获取和清理
# ============================================================

def _key(context: object | None, call_id: str) -> tuple[int, str] | None:
    """构造 run 内唯一键；无上下文或空 call_id 时不保存跨阶段状态。"""
    if context is None or not call_id:
        return None
    return id(context), call_id


def _store_prepared(context: object | None, call_id: str, change: _PreparedChange) -> None:
    key = _key(context, call_id)
    if key is None:
        return
    with _PREPARED_LOCK:
        _PREPARED[key] = change


def _take_prepared(context: object | None, call_id: str) -> _PreparedChange | None:
    key = _key(context, call_id)
    if key is None:
        return None
    with _PREPARED_LOCK:
        return _PREPARED.pop(key, None)


def discard_approval_preview(context: object | None, call_id: str) -> None:
    """拒绝、取消、校验失败或 handler 结束后清理尚未消费的预备状态。"""
    _take_prepared(context, call_id)


def _prepare_edit(args: dict[str, Any]) -> tuple[_PreparedChange | None, str]:
    """校验 edit_file 参数并生成不落盘的拟议内容。"""
    path = args.get("path")
    old_text = args.get("old_text")
    new_text = args.get("new_text")
    replace_all = args.get("replace_all", False)
    if not all(isinstance(value, str) for value in (path, old_text, new_text)):
        return None, "edit_file 的 path、old_text、new_text 必须是字符串。"
    if type(replace_all) is not bool:
        return None, "edit_file 的 replace_all 必须是布尔值。"
    target, current, err = _read_editable(path)
    if target is None or current is None:
        return None, err
    proposed, err = _replacement(current, old_text, new_text, replace_all)
    if proposed is None:
        return None, err
    relative = str(target.relative_to(files_mod.SANDBOX))
    diff, summary = _unified_diff(relative, current, proposed)
    return _PreparedChange(
        operation="edit",
        path=relative,
        target=target,
        expected_sha256=_digest(current),
        proposed_text=proposed,
        diff=diff,
        summary=summary,
    ), ""


def _prepare_restore(args: dict[str, Any]) -> tuple[_PreparedChange | None, str]:
    """读取指定备份并生成恢复前的反向差异。"""
    backup_id = args.get("backup_id")
    if not isinstance(backup_id, str):
        return None, "restore_file_backup 的 backup_id 必须是字符串。"
    record, err = _load_backup(backup_id)
    if record is None:
        return None, err
    target, current, err = _read_editable(record["path"])
    if target is None or current is None:
        return None, err
    proposed = record["content"]
    if proposed == current:
        return None, "当前文件与该备份内容相同，无需恢复。"
    relative = str(target.relative_to(files_mod.SANDBOX))
    diff, summary = _unified_diff(relative, current, proposed)
    return _PreparedChange(
        operation="restore",
        path=relative,
        target=target,
        expected_sha256=_digest(current),
        proposed_text=proposed,
        diff=diff,
        summary=summary,
        source_backup_id=backup_id,
    ), ""


def prepare_approval_preview(
    tool_name: str,
    call_id: str,
    args: dict[str, Any],
    context: object | None,
) -> ApprovalPreview:
    """为支持预览的写工具生成安全展示参数；其他工具原样返回。"""
    if tool_name == "edit_file":
        change, err = _prepare_edit(args)
    elif tool_name == "restore_file_backup":
        change, err = _prepare_restore(args)
    elif tool_name == "run_command":
        display, err = prepare_command_preview(args)
        return ApprovalPreview(args=display, error=err)
    else:
        return ApprovalPreview(args=dict(args))
    if change is None:
        return ApprovalPreview(args={"path": args.get("path", "")}, error=err)
    _store_prepared(context, call_id, change)
    display = {
        "path": change.path,
        "summary": change.summary,
        "diff": change.diff,
    }
    if change.source_backup_id:
        display["backup_id"] = change.source_backup_id
    return ApprovalPreview(args=display)


def _execute_prepared(change: _PreparedChange) -> str:
    """再次校验摘要，保存原文备份，然后原子覆盖目标文件。"""
    target, current, err = _read_editable(change.path)
    if target is None or current is None:
        return err
    if target != change.target or _digest(current) != change.expected_sha256:
        return "文件在等待批准期间发生了变化，本次操作未写入。请重新读取并生成新的预览。"
    try:
        rollback_id = _save_backup(
            target,
            current,
            change.proposed_text,
            reason=change.operation,
        )
        files_mod._atomic_write(target, change.proposed_text)
    except OSError as exc:
        return f"保存失败，原文件未被非原子覆盖：{type(exc).__name__}: {exc}"
    action = "已恢复" if change.operation == "restore" else "已编辑"
    return (
        f"{action} {change.path}（{change.summary}）。"
        f"覆盖前内容已保存为备份 {rollback_id}。"
    )


# ============================================================
# 4. 给模型调用的编辑与恢复工具
# ============================================================

@tool(parse_docstring=True)
def edit_file(
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
    runtime: ToolRuntime[AgentContext] = None,
) -> str:
    """精确替换现有文本；交互入口会先展示 diff，批准后备份并原子保存。

    Args:
        path: 工作目录内的现有文本文件。
        old_text: 文件中完全一致的原文；默认必须只出现一次。
        new_text: 替换后的文本。
        replace_all: 明确设为 true 时替换所有匹配；默认 false，遇到多处匹配会拒绝。

    Returns:
        编辑结果和可恢复的 backup_id；原文不匹配或文件已变化时不会写入。
    """
    context = getattr(runtime, "context", None)
    call_id = str(getattr(runtime, "tool_call_id", "") or "")
    change = _take_prepared(context, call_id)
    if change is None:
        change, err = _prepare_edit({
            "path": path,
            "old_text": old_text,
            "new_text": new_text,
            "replace_all": replace_all,
        })
        if change is None:
            return err
    result = _execute_prepared(change)
    # LangChain 会将普通字符串包装为成功的 ToolMessage。保存机器可读回执后，
    # 修复状态才能由实际写入结果而不是 ToolMessage 形态驱动。
    if context is not None and call_id:
        if result.startswith("已编辑 "):
            context.edit_results[call_id] = "succeeded"
            match = re.search(r"备份 (\d{8}T\d{6}Z-[0-9a-f]{12})", result)
            if match and getattr(context, "run_id", None):
                try:
                    from runtime.storage import get_runtime_store
                    get_runtime_store().update_repair_session(context.run_id, backup_id=match.group(1))
                except Exception:
                    logger.exception("无法保存修复备份标识：%s", context.run_id)
        elif "等待批准期间发生了变化" in result:
            context.edit_results[call_id] = "conflicted"
        else:
            context.edit_results[call_id] = "failed"
    return result


@tool(parse_docstring=True)
def list_file_backups(path: str = "", limit: int = 20) -> str:
    """列出当前工作目录可恢复的最近文件备份。

    Args:
        path: 可选的相对文件路径；为空时列出当前工作目录的全部备份。
        limit: 最多返回多少条，范围 1-50。

    Returns:
        每行包含 backup_id、时间、文件、原因和原文大小；不返回备份正文。
    """
    if not 1 <= limit <= MAX_BACKUP_LIST:
        return f"limit 必须在 1-{MAX_BACKUP_LIST} 之间。"
    relative_filter = ""
    if path:
        target, err = files_mod._resolve_safe(path)
        if target is None:
            return err
        if files_mod._is_sensitive_path(path, target):
            return "出于安全原因，不能查看敏感配置或密钥文件的备份。"
        relative_filter = str(target.relative_to(files_mod.SANDBOX))
    if not BACKUP_DIR.is_dir():
        return "当前工作目录还没有文件备份。"

    records = []
    for candidate in sorted(BACKUP_DIR.glob("*.json"), reverse=True):
        backup_id = candidate.stem
        record, _ = _load_backup(backup_id)
        if record is None:
            continue
        if relative_filter and record["path"] != relative_filter:
            continue
        records.append(record)
        if len(records) >= limit:
            break
    if not records:
        return "没有匹配当前工作目录和路径的文件备份。"
    return "\n".join(
        f"{record['backup_id']}  {record['created_at']}  {record['path']}  "
        f"({record['reason']}, {len(record['content'])} 字符)"
        for record in records
    )


@tool(parse_docstring=True)
def restore_file_backup(
    backup_id: str,
    runtime: ToolRuntime[AgentContext] = None,
) -> str:
    """恢复一份文件备份；交互入口会先展示反向 diff，并备份恢复前状态。

    Args:
        backup_id: `list_file_backups` 返回的备份 ID。

    Returns:
        恢复结果，以及恢复前状态的新 backup_id；文件已变化时拒绝写入。
    """
    context = getattr(runtime, "context", None)
    call_id = str(getattr(runtime, "tool_call_id", "") or "")
    change = _take_prepared(context, call_id)
    if change is None:
        change, err = _prepare_restore({"backup_id": backup_id})
        if change is None:
            return err
    return _execute_prepared(change)


# 只读列表在前；编辑和恢复会修改工作区，风险表登记为 workspace_write。
EDIT_TOOLS = [list_file_backups, edit_file, restore_file_backup]
