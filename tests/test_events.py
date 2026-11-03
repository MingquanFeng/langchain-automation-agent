"""
事件归一化层的测试。全部用假模型，不调真实 API。

主要职责：
1. 守住思考内容过滤 —— 滤漏了会被大段自说自话刷屏，误伤了会丢正文。
2. 守住中间件噪音判定 —— 滤多了会丢掉「写操作被拦截」这类必须显示的消息。

为什么这两条要在归一化层测：
它们是任何消费流的地方都要做的判断（终端、将来的 TUI、桌面壳），
不随界面形态改变，所以测在公共模块上而不是某个界面里。
"""

from __future__ import annotations

# 标准库：sys / pathlib 处理导入路径。
import sys
from pathlib import Path

# 把项目根目录插到 sys.path 最前面。
# 这样不管从哪个目录跑 pytest，都能 import 到 agent / runtime 包。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 项目内部：
# ThinkFilter：流式滤掉 <think> 思考块。
# is_hint_node：判断一条消息是不是中间件插入的例行提示。
from runtime.events import ThinkFilter, is_hint_node


# ---------- 思考内容过滤 ----------

# MiniMax 这类推理模型会把思考过程混在 content 里返回，
# 默认要滤掉，否则界面会被大段自说自话刷屏。


def test_think_filter_removes_think_block():
    """完整的 <think> 块要被整段拿掉，只留下正文。"""
    f = ThinkFilter()
    assert f.feed("<think>思考中</think>答案") == "答案"


def test_think_filter_handles_split_chunks():
    """闭标签还没到时也不能漏出思考内容。"""
    f = ThinkFilter()
    assert f.feed("<think>我在") == ""
    assert f.feed("思考</think>正文") == "正文"


def test_think_filter_keeps_plain_text():
    """普通文本要原样透传，过滤器不能误伤。"""
    f = ThinkFilter()
    assert f.feed("普通") == "普通"
    assert f.feed("回答") == "回答"


def test_think_filter_reset_clears_buffer():
    """reset 要把缓冲清空，否则上一轮的思考片段会混进这一轮。"""
    f = ThinkFilter()
    assert f.feed("<think>我在思考") == ""
    f.reset()
    assert f.feed("新的一轮") == "新的一轮"


# ---------- 中间件噪音判定 ----------


def test_is_hint_node_filters_context_injection():
    """中间件注入的例行提示要能被认出来。"""
    assert is_hint_node("ContextInjectionMiddleware") is True
    # 同一个节点可能带后缀，用 startswith 比较。
    assert is_hint_node("ContextInjectionMiddleware:1") is True


def test_is_hint_node_keeps_other_nodes():
    """模型节点和工具节点的消息必须留着 —— 写操作被拦的说明在那儿。"""
    assert is_hint_node("model") is False
    assert is_hint_node("tools") is False
    assert is_hint_node("") is False
