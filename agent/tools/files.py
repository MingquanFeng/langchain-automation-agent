"""本地文件读写工具。

对应课件 05 章：@tool 装饰器定义工具，docstring 即给模型的工具说明。
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from ..config import DATA_DIR

# 允许 Agent 触碰的根目录，防止越界读写
SANDBOX = DATA_DIR.resolve()


def _resolve(path: str) -> Path:
    """把相对路径解析到沙箱内。越界则抛错。"""
    target = (SANDBOX / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not target.is_relative_to(SANDBOX):
        raise ValueError(f"路径超出允许范围（仅允许 {SANDBOX}）：{path}")
    return target


@tool
def list_files(pattern: str = "*") -> str:
    """列出数据目录下的文件。pattern 是 glob 模式，例如 '*.md'。返回文件名和大小列表。"""
    target = _resolve(".")
    matches = sorted(p for p in target.glob(pattern) if p.is_file())
    if not matches:
        return "没有匹配的文件"
    return "\n".join(f"{p.relative_to(SANDBOX)}  ({p.stat().st_size} bytes)" for p in matches)


@tool
def read_file(path: str) -> str:
    """读取文件内容。path 相对于数据目录。返回文件的完整文本。"""
    target = _resolve(path)
    if not target.exists():
        return f"文件不存在：{path}"
    return target.read_text(encoding="utf-8", errors="replace")


@tool
def write_file(path: str, content: str) -> str:
    """写入文件。path 相对于数据目录，父目录会自动创建。返回写入后的路径。"""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"已写入 {target.relative_to(SANDBOX)}（{len(content)} 字符）"


@tool
def append_file(path: str, content: str) -> str:
    """追加内容到文件末尾。path 相对于数据目录。"""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(content)
    return f"已追加 {len(content)} 字符到 {target.relative_to(SANDBOX)}"


@tool
def save_json(path: str, data: dict) -> str:
    """把字典保存成 JSON 文件。path 相对于数据目录。"""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"已保存 JSON 到 {target.relative_to(SANDBOX)}"


@tool
def read_json(path: str) -> str:
    """读取 JSON 文件。path 相对于数据目录。返回格式化后的 JSON 文本。"""
    target = _resolve(path)
    if not target.exists():
        return f"文件不存在：{path}"
    return json.dumps(json.loads(target.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)


FILE_TOOLS = [list_files, read_file, write_file, append_file, save_json, read_json]
