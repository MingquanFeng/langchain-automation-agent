"""
本地文件读写工具。

对应课件 05 章（courseware/langChain/尚硅谷-05-Tools.pdf）：@tool 装饰器定义工具，docstring 即给模型的工具说明。

主要职责：
1. 提供 Agent 工作目录内的文件列举、读取、写入、JSON 读写能力。
2. 用沙箱把所有读写限制在配置的工作目录内，防止 Agent 越界。
3. 对大文件、二进制文件和运行时自己的记忆库做保护，避免撑爆上下文。
4. 阻止读取 `.env` 和常见密钥文件，避免凭据进入模型上下文。
5. 列举时跳过常见版本控制、依赖和构建产物目录，避免扫描仓库中的无关文件。

注意：
    被 @tool 装饰的函数，它的 docstring 会被当作「工具描述」发给模型，
    相当于提示词的一部分。所以这些 docstring 只写模型需要知道的功能性说明，
    深度解释放在周围的 # 注释里 —— 改 docstring 等于改模型行为。

    这几个工具都开了 parse_docstring=True（课件 05 章 3.1）：
    docstring 里「Args:」段会被解析成每个参数的 description，
    塞进发给模型的 JSON Schema。模型因此能直接看到
    「pattern 是干什么的」「offset 从第几行开始」，
    而不是靠参数名猜。不开的话整段 docstring 会原样进 description，
    参数仍然是没说明的白板。
"""

from __future__ import annotations

# datetime：标准库，取文件的修改时间并格式化成可读文本。
from datetime import datetime

# json：标准库，处理 JSON 序列化与反序列化。
import json

# os：标准库，os.replace 用于原子替换文件。
import os

# Path：标准库的路径对象，负责路径拼接、规范化、目录创建。
from pathlib import Path

# tool：LangChain 提供的装饰器，把普通函数变成模型可调用的工具。
#
# 装饰之后：
# 1. 函数名变成工具名（例如 write_file）。
# 2. 函数的 docstring 变成工具描述。
# 3. 函数的类型注解变成参数的 JSON Schema。
# 4. 函数体多出 .invoke() 等调用方式。
from langchain_core.tools import tool

# DATA_DIR：项目的数据目录，定义在 agent/config.py。
#
# 所有工具都只能在这个目录里操作。
from ..config import DATA_DIR


# ============================================================
# 1. 沙箱与保护参数
# ============================================================

# SANDBOX：允许 Agent 触碰的根目录。
#
# resolve() 会把路径规范化（消掉 ..、软链接等），
# 之后再做包含关系判断，避免用 ../ 绕出去。
SANDBOX = DATA_DIR.resolve()

# MAX_READ_BYTES：单次读取的字节上限。
#
# 没有这个上限，Agent 一句「看下这个文件」就能把几十 MB 塞进上下文，
# 直接撑爆模型的上下文窗口（MiniMax 会报 context window exceeds limit）。
#
# 200KB 大约是几万字，足够读完一份报告或一段代码；
# 再大的文件应该用 offset / limit 分段读。
MAX_READ_BYTES = 200_000

# DEFAULT_READ_LINES：read_file 不指定 limit 时返回多少行。
#
# 为什么按行也限一手：
#    MAX_READ_BYTES 挡的是「单个巨文件」，
#    行数挡的是「文件不大但行数极多」的情况。
#    两个维度一起卡，才不会出现「一行 200KB」这种把上下文一次吃掉的读取。
DEFAULT_READ_LINES = 200

# MAX_LIST_ROWS：list_files 最多列出多少个文件。
#
# 目录里堆了几千个文件时，全列出来既撑上下文又淹没有用信息。
# 截断后告诉模型总共多少个，让它自己缩小 pattern。
MAX_LIST_ROWS = 200

# 列举项目仓库时跳过常见版本控制、依赖和构建产物目录。
# 这些目录仍在工作区边界内，但不会被默认递归扫描。
LIST_IGNORED_DIR_NAMES = frozenset({
    ".git",
    ".next",
    ".turbo",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    "coverage",
    "target",
})

# BLOCKED_NAMES：禁止读取的文件名。
#
# 这些是运行时自己的产物（SQLite 记忆库及其 WAL/SHM 附属文件）。
# 读了既没意义 —— Agent 看不懂二进制，
# 又会触发上面那个撑爆上下文的坑。
BLOCKED_NAMES = {"memory.sqlite", "memory.sqlite-wal", "memory.sqlite-shm"}

# 常见凭据文件名和私钥扩展名。读取和列举均会隐藏这些路径。
SENSITIVE_FILE_NAMES = frozenset({
    ".envrc",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials.json",
    "service-account.json",
    "service_account.json",
    "token.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "secrets.toml",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
})
SENSITIVE_FILE_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx"})
SENSITIVE_DIR_NAMES = frozenset({".ssh", ".aws"})
SAFE_ENV_TEMPLATE_NAMES = frozenset({
    ".env.example",
    ".env.sample",
    ".env.template",
})

# TMP_SUFFIX：原子写时用的临时文件后缀。
#
# 写文件先落到「xxx.md.tmp」，再 os.replace 覆盖正式文件。
# 临时文件在完成前是存在的，所以列表工具要把它们滤掉。
TMP_SUFFIX = ".tmp"


# ============================================================
# 2. 内部辅助函数
# ============================================================

def _resolve(path: str) -> Path:
    """
    把相对路径解析到沙箱内，越界则抛错。

    参数：
        path：相对工作目录的路径，也可以是工作目录内的绝对路径。

    返回：
        规范化后的绝对 Path，保证落在 SANDBOX 之内。

    为什么用 resolve() 再判断：
        直接判断字符串里有没有 ".." 是不够的，
        绝对路径、软链接都能绕出去。
        先规范化再判断包含关系才是可靠的。
    """
    target = (SANDBOX / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not target.is_relative_to(SANDBOX):
        raise ValueError(f"路径超出允许范围（仅允许 {SANDBOX}）：{path}")
    return target


def _resolve_safe(path: str) -> tuple[Path | None, str]:
    """
    沙箱解析的安全版本：越界不抛异常，返回一句给模型看的说明。

    参数：
        path：相对工作目录的路径。

    返回：
        (目标路径, 错误说明)
        成功时错误说明为空串；越界时路径为 None、说明非空。

    为什么工具层不直接抛异常：
        沙箱越界基本是模型自己把路径拼错了，不是系统故障。
        抛异常会被 ToolRetryMiddleware 接住重试
        （默认 retry_on=(Exception,)，实测同一个错参数会被重试 3 次、
        空耗约 3 秒），而模型最后收到的还是
        「failed after 3 attempts with ValueError: ... Please try again.」
        这种没告诉它该怎么改的英文提示，它很可能再拿同一个错路径试一轮。

        返回一句中文说明，模型能立刻换成工作目录内的相对路径。
        这跟 _read_text 对二进制文件、对超大文件的处理是同一套哲学：
        给模型一句能照做的说明，而不是把整轮推理打断。

    注意：
        Python 调用方想拿到异常，仍然可以直接用 _resolve()。
    """
    try:
        return _resolve(path), ""
    except ValueError as exc:
        return None, (
            f"{exc}。"
            "请改用工作目录内的相对路径，例如 'out/report.md'、'notes.md'；"
            "先用 list_files 看看有哪些文件可用。"
        )


def _is_hidden(target: Path) -> bool:
    """
    判断这个文件要不要对模型隐藏。

    隐藏两类：
    1. 记忆库（memory.sqlite 及其 WAL/SHM）—— 读了没意义，还撑上下文。
    2. 原子写的临时文件（xxx.md.tmp）—— 写了一半的中间态，
       让模型看到只会困惑（内容是残缺的）。
    """
    return target.name in BLOCKED_NAMES or target.name.endswith(TMP_SUFFIX)


def _is_sensitive_file(target: Path) -> bool:
    """判断文件名或所在目录是否属于常见密钥、凭据存放位置。"""
    name = target.name.lower()
    if name in SAFE_ENV_TEMPLATE_NAMES:
        return False
    if name == ".env" or name.startswith(".env.") or name == ".envrc":
        return True
    if name in SENSITIVE_FILE_NAMES:
        return True
    if target.suffix.lower() in SENSITIVE_FILE_SUFFIXES:
        return True

    try:
        relative_parts = target.relative_to(SANDBOX).parts[:-1]
    except ValueError:
        return True
    return any(part.lower() in SENSITIVE_DIR_NAMES for part in relative_parts)


def _is_sensitive_path(path: str, resolved: Path) -> bool:
    """同时检查用户给出的路径和解析后的目标，避免用软链接绕过保护。"""
    requested = Path(path)
    if not requested.is_absolute():
        requested = SANDBOX / requested
    return _is_sensitive_file(requested) or _is_sensitive_file(resolved)


def _list_matching_files(
    root: Path,
    pattern: str,
) -> tuple[list[Path], set[str], bool]:
    """列出匹配文件，并在遍历时剪掉常见生成目录。"""
    matches: list[Path] = []
    skipped_dirs: set[str] = set()
    skipped_sensitive = False

    for current, dirnames, filenames in os.walk(root):
        excluded = {name for name in dirnames if name in LIST_IGNORED_DIR_NAMES}
        skipped_dirs.update(excluded)
        dirnames[:] = sorted(name for name in dirnames if name not in excluded)

        for filename in filenames:
            candidate = Path(current) / filename
            if _is_hidden(candidate):
                continue
            try:
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve()
                if not resolved.is_relative_to(root):
                    continue
            except (OSError, RuntimeError):
                continue
            if _is_sensitive_file(candidate) or _is_sensitive_file(resolved):
                skipped_sensitive = True
                continue
            relative = candidate.relative_to(root)
            if relative.match(pattern):
                matches.append(candidate)

    matches.sort()
    return matches, skipped_dirs, skipped_sensitive


def _list_filter_note(skipped_dirs: set[str], skipped_sensitive: bool) -> str:
    """说明列举时应用的过滤规则，避免把过滤结果误认为全量结果。"""
    notes = []
    if skipped_dirs:
        notes.append(f"已跳过目录：{', '.join(sorted(skipped_dirs))}")
    if skipped_sensitive:
        notes.append("出于安全原因，已隐藏敏感配置或密钥文件")
    return f"\n\n（{'；'.join(notes)}。）" if notes else ""


def _describe(target: Path) -> str:
    """
    把一个文件描述成一行「相对路径  (字节数, 修改时间)」。

    为什么带修改时间：
        模型要判断「哪些文件很久没动了」时，光有文件名和大小是没有依据的 ——
        只能靠猜。给出修改时间，它才能真的按时间做筛选。
        （runtime/tasks.py 里的「数据目录巡检」任务就是干这个的。）

    为什么用本地时间而不是 ISO 格式带时区：
        模型读的是给人看的清单，本地时间更直观；
        跨时区比较不是这个工具的用途。
    """
    stat = target.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"{target.relative_to(SANDBOX)}  ({stat.st_size} bytes, {mtime})"


def _read_text(target: Path) -> str:
    """
    读文本，带上限和二进制检测。

    返回的是可以直接放进上下文的内容，
    所以遇到超限或二进制时会返回一句说明文字，
    而不是抛异常 —— 抛异常会打断 Agent 的推理，
    一句说明能让模型换个思路。
    """
    size = target.stat().st_size

    # 只读前 MAX_READ_BYTES 字节，避免大文件把内存和上下文一起撑爆。
    raw = target.read_bytes()[:MAX_READ_BYTES]

    # 二进制判定：只看开头 8KB 里有没有 NUL 字节。
    #
    # NUL 字节（\x00）几乎只出现在二进制文件里，
    # UTF-8 文本不会出现，所以这个判据够用且开销极小。
    if b"\x00" in raw[:8192]:
        return f"{target.name} 是二进制文件（{size} 字节），无法作为文本读取。"

    # errors="replace"：遇到非法字节序列时替换成替代字符，
    # 而不是抛 UnicodeDecodeError 让整个读取失败。
    text = raw.decode("utf-8", errors="replace")

    if size > MAX_READ_BYTES:
        return (
            f"{text}\n\n"
            f"（文件共 {size} 字节，已截断到前 {MAX_READ_BYTES} 字节。"
            f"需要更多内容请分段读取。）"
        )
    return text


def _read_lines(target: Path, offset: int, limit: int) -> str:
    """
    按行读取一段内容，供 read_file 的分段读取用。

    参数：
        target：已确认在沙箱内的文件路径。
        offset：起始行号，从 0 开始。
        limit：最多返回多少行；<= 0 表示不按行数限制。

    返回：
        选中的行 + 一句行号说明，让模型知道读到哪儿了、还剩多少。
    """
    size = target.stat().st_size
    raw = target.read_bytes()[:MAX_READ_BYTES]

    if b"\x00" in raw[:8192]:
        return f"{target.name} 是二进制文件（{size} 字节），无法作为文本读取。"

    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    total = len(lines)

    start = max(0, offset)
    if start >= total:
        return (
            f"（offset={offset} 超出文件范围：文件共 {total} 行，"
            f"没有更多内容可读。）"
        )

    # limit <= 0 时用默认行数上限，不是无限 ——
    # 行数不设防的话，几千行的小文件也能一次灌满上下文。
    want = DEFAULT_READ_LINES if limit <= 0 else limit
    end = min(total, start + want)
    selected = lines[start:end]

    body = "\n".join(selected)

    # 末尾的说明：把「这次读到哪儿」「还剩多少」「下一步怎么传」讲清楚。
    # 模型靠这段决定下一次 read_file 的 offset 传多少。
    notes = []

    # 字节超限也要说 —— 这是「文件本身被截断」，
    # 和「只返回了一段」是两回事，模型需要区分。
    if size > MAX_READ_BYTES:
        notes.append(
            f"文件共 {size} 字节，已截断到前 {MAX_READ_BYTES} 字节，后面的内容读不到"
        )

    notes.append(f"第 {start + 1}-{end} 行，共 {total} 行")
    if end < total:
        notes.append(f"还有 {total - end} 行，用 read_file 传 offset={end} 继续读")

    return f"{body}\n\n（" + "；".join(notes) + "）"


def _atomic_write(target: Path, text: str) -> None:
    """
    原子写：先写同目录临时文件，再用 os.replace 覆盖正式文件。

    为什么：
        直接 write_text 在写到一半时进程被杀、磁盘写满、或者被 Ctrl-C，
        会留下一个半截文件。而 Agent 下次读到它会当成正常内容，
        基于残缺数据往下推理 —— 这种错误极难排查。

        os.replace 在同一文件系统内是原子的：
        要么还是旧内容，要么就是完整的新内容，不存在中间态。
    """
    tmp = target.with_name(target.name + TMP_SUFFIX)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    except BaseException:
        # 失败时清掉半截临时文件，别留在目录里误导模型。
        tmp.unlink(missing_ok=True)
        raise


# ============================================================
# 3. 给模型调用的工具
# ============================================================

@tool(parse_docstring=True)
def list_files(pattern: str = "*", max_results: int = MAX_LIST_ROWS) -> str:
    """列出工作目录下的文件，跳过常见生成目录和敏感文件。

    Args:
        pattern: glob 匹配模式，匹配任意层级。例如 '*.md' 找出所有层级的 md，
            'out/*' 只看 out 目录。默认 '*' 列出全部。
        max_results: 最多返回多少条，超出会截断并说明总数。

    Returns:
        每行一个「相对路径  (字节数, 修改时间)」，以及必要时的截断提示。
    """
    target = _resolve(".")

    matches, skipped_dirs, skipped_sensitive = _list_matching_files(target, pattern)
    filter_note = _list_filter_note(skipped_dirs, skipped_sensitive)

    total = len(matches)
    shown = matches[:max_results]

    if not shown:
        return f"没有匹配 {pattern} 的文件。{filter_note}"

    lines = [_describe(p) for p in shown]
    out = "\n".join(lines)

    if total > len(shown):
        out += (
            f"\n\n（共 {total} 个文件，只返回前 {len(shown)} 个。"
            f"缩小 pattern 范围再看其余的。）"
        )
    return out + filter_note


@tool(parse_docstring=True)
def read_file(path: str, offset: int = 0, limit: int = DEFAULT_READ_LINES) -> str:
    """读取文件内容，可按行分段读取。

    Args:
        path: 目标路径，相对于工作目录。例如 'out/report.md'。
        offset: 起始行号，从 0 开始。用来接着上次读到的位置继续读。
        limit: 最多返回多少行。默认 200；传 0 也按 200 处理（不设无限，
            免得几千行的小文件一次灌满上下文）。

    Returns:
        文件的文本内容，末尾附行号说明（这次是第几到第几行、共多少行）。
        文件不存在、是目录、是二进制或路径越界时，返回对应的说明文字。
    """
    target, err = _resolve_safe(path)
    if target is None:
        return err
    if _is_sensitive_path(path, target):
        return "出于安全原因，不能读取敏感配置或密钥文件。"
    if not target.exists():
        return f"文件不存在：{path}"
    if target.is_dir():
        return f"{path} 是目录，不是文件。用 list_files 看里面有什么。"
    if target.name in BLOCKED_NAMES:
        return f"{path} 是运行时自己的记忆库，不能读取。"
    return _read_lines(target, offset, limit)


@tool(parse_docstring=True)
def write_file(path: str, content: str) -> str:
    """把文本写入文件（覆盖原有内容）。

    Args:
        path: 目标路径，相对于工作目录。父目录会自动创建。
        content: 要写入的完整文本。

    Returns:
        写入后的相对路径和字符数。
    """
    target, err = _resolve_safe(path)
    if target is None:
        return err
    # parents=True 递归建父目录，exist_ok=True 目录已存在也不报错。
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, content)
    return f"已写入 {target.relative_to(SANDBOX)}（{len(content)} 字符）"


@tool(parse_docstring=True)
def append_file(path: str, content: str) -> str:
    """把文本追加到文件末尾。

    Args:
        path: 目标路径，相对于工作目录。父目录会自动创建，文件不存在则新建。
        content: 要追加的文本。

    Returns:
        追加后的相对路径和追加的字符数。
    """
    target, err = _resolve_safe(path)
    if target is None:
        return err
    target.parent.mkdir(parents=True, exist_ok=True)
    # 用 with 保证文件句柄一定被关闭。
    #
    # 追加不用原子写：append 的语义本身就是「接在后面」，
    # 原子替换反而要把整个旧内容读进内存再写回，大文件上不划算。
    with target.open("a", encoding="utf-8") as f:
        f.write(content)
    return f"已追加 {len(content)} 字符到 {target.relative_to(SANDBOX)}"


@tool(parse_docstring=True)
def save_json(path: str, data: dict | list) -> str:
    """把数据保存成 JSON 文件（覆盖原有内容）。

    Args:
        path: 目标路径，相对于工作目录。父目录会自动创建。
        data: 要保存的数据。顶层可以是对象（字典），也可以是数组（列表）。

    Returns:
        保存位置的相对路径。
    """
    target, err = _resolve_safe(path)
    if target is None:
        return err
    target.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False：保留中文原样，不要转成 \uXXXX。
    # indent=2：缩进两格，方便人和模型都能读。
    _atomic_write(target, json.dumps(data, ensure_ascii=False, indent=2))
    return f"已保存 JSON 到 {target.relative_to(SANDBOX)}"


@tool(parse_docstring=True)
def read_json(path: str) -> str:
    """读取并解析 JSON 文件。

    Args:
        path: 目标路径，相对于工作目录。

    Returns:
        格式化后的 JSON 文本（缩进两格）。
        解析失败时返回错误说明，含出错位置，方便模型改。
    """
    target, err = _resolve_safe(path)
    if target is None:
        return err
    if _is_sensitive_path(path, target):
        return "出于安全原因，不能读取敏感配置或密钥文件。"
    if not target.exists():
        return f"文件不存在：{path}"
    if target.name in BLOCKED_NAMES:
        return f"{path} 是运行时自己的记忆库，不能读取。"

    raw = _read_text(target)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return (
            f"{path} 不是合法的 JSON：{exc.msg}"
            f"（第 {exc.lineno} 行第 {exc.colno} 列）。"
            f"请检查该文件的格式，或改用 read_file 读取原文。"
        )
    return json.dumps(parsed, ensure_ascii=False, indent=2)


# ============================================================
# 4. 工具清单
# ============================================================

# FILE_TOOLS：本模块提供的全部工具。
#
# agent/tools/__init__.py 会把它并进 ALL_TOOLS，
# 而 ALL_TOOLS 才是最终注入 create_agent 的那份清单。
#
# 顺序不影响功能，但保持「只读在前、写操作在后」，
# 方便人一眼扫过去。
FILE_TOOLS = [list_files, read_file, write_file, append_file, save_json, read_json]
