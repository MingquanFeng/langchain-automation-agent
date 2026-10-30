"""
本地文件读写工具。

对应课件 05 章（courseware/尚硅谷-05-Tools.pdf）：@tool 装饰器定义工具，docstring 即给模型的工具说明。

主要职责：
1. 提供数据目录（data/）内的文件列举、读取、写入、JSON 读写能力。
2. 用沙箱把所有读写限制在 data/ 目录内，防止 Agent 越界。
3. 对大文件、二进制文件、运行时自己的记忆库做保护，
   避免一次读取就把模型的上下文窗口撑爆。

注意：
    被 @tool 装饰的函数，它的 docstring 会被当作「工具描述」发给模型，
    相当于提示词的一部分。所以这些 docstring 只写模型需要知道的功能性说明，
    深度解释放在周围的 # 注释里 —— 改 docstring 等于改模型行为。
"""

from __future__ import annotations

# json：标准库，处理 JSON 序列化与反序列化。
import json

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
# 再大的文件应该分段处理，或者先看目录结构再定点读。
MAX_READ_BYTES = 200_000

# BLOCKED_NAMES：禁止读取的文件名。
#
# 这些是运行时自己的产物（SQLite 记忆库及其 WAL/SHM 附属文件）。
# 读了既没意义 —— Agent 看不懂二进制，
# 又会触发上面那个撑爆上下文的坑。
BLOCKED_NAMES = {"memory.sqlite", "memory.sqlite-wal", "memory.sqlite-shm"}


# ============================================================
# 2. 内部辅助函数
# ============================================================

def _resolve(path: str) -> Path:
    """
    把相对路径解析到沙箱内，越界则抛错。

    参数：
        path：相对 data/ 的路径，也可以是绝对路径。

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


# ============================================================
# 3. 给模型调用的工具
# ============================================================

@tool
def list_files(pattern: str = "*") -> str:
    """列出数据目录下的文件。pattern 是 glob 模式，例如 '*.md'。返回文件名和大小列表。"""
    target = _resolve(".")
    matches = sorted(p for p in target.glob(pattern) if p.is_file())
    if not matches:
        return "没有匹配的文件"
    # 带上大小，是为了让模型能自己判断值不值得读
    # —— 几十万字节的文件读进来会被截断。
    return "\n".join(f"{p.relative_to(SANDBOX)}  ({p.stat().st_size} bytes)" for p in matches)


@tool
def read_file(path: str) -> str:
    """读取文件内容。path 相对于数据目录。返回文本内容。

    超过 200KB 会截断，二进制文件会拒绝读取。
    """
    target = _resolve(path)
    if not target.exists():
        return f"文件不存在：{path}"
    if target.name in BLOCKED_NAMES:
        return f"{path} 是运行时自己的记忆库，不能读取。"
    return _read_text(target)


@tool
def write_file(path: str, content: str) -> str:
    """写入文件。path 相对于数据目录，父目录会自动创建。返回写入后的路径。"""
    target = _resolve(path)
    # parents=True 递归建父目录，exist_ok=True 目录已存在也不报错。
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"已写入 {target.relative_to(SANDBOX)}（{len(content)} 字符）"


@tool
def append_file(path: str, content: str) -> str:
    """追加内容到文件末尾。path 相对于数据目录。"""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # 用 with 保证文件句柄一定被关闭。
    with target.open("a", encoding="utf-8") as f:
        f.write(content)
    return f"已追加 {len(content)} 字符到 {target.relative_to(SANDBOX)}"


@tool
def save_json(path: str, data: dict) -> str:
    """把字典保存成 JSON 文件。path 相对于数据目录。"""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False：保留中文原样，不要转成 \uXXXX。
    # indent=2：缩进两格，方便人和模型都能读。
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"已保存 JSON 到 {target.relative_to(SANDBOX)}"


@tool
def read_json(path: str) -> str:
    """读取 JSON 文件。path 相对于数据目录。返回格式化后的 JSON 文本。"""
    target = _resolve(path)
    if not target.exists():
        return f"文件不存在：{path}"
    if target.name in BLOCKED_NAMES:
        return f"{path} 是运行时自己的记忆库，不能读取。"
    return _read_text(target)


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
