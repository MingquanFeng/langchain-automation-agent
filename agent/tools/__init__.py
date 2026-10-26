"""工具集合。课件 05 章：@tool 定义，docstring 就是给模型看的说明。"""

from .files import FILE_TOOLS
from .notify import NOTIFY_TOOLS

ALL_TOOLS = [*FILE_TOOLS, *NOTIFY_TOOLS]

__all__ = ["ALL_TOOLS", "FILE_TOOLS", "NOTIFY_TOOLS"]
