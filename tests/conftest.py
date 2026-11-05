"""
pytest 的全局配置。

主要职责：
在测试期间关掉 LangSmith 追踪。

为什么必须关：
1. **追踪会改变模型的调用路径** —— 追踪要记录增量 token，
   于是 LangChain 让模型走流式路径（`_stream`）而不是非流式路径（`_generate`）。
   而测试用的假模型（`tests/fakes.py` 的 `FakeModel`、
   `tests/test_e2e.py` 的 `ToolCapableFakeModel`）给的响应是**一次性的有限资源**，
   走流式路径会更快消耗掉它们，消息不够就 `StopIteration`，
   表现为「Model call failed after 4 attempts」。
2. **测试会慢几十倍** —— 每次模型调用都要往 LangSmith 发一次 HTTP。
   实测同一组用例：追踪开着 6.5 秒，关掉 0.18 秒。

注意：
这只作用于 pytest 进程。生产运行时（`python -m runtime`）的追踪开关
仍然由 `.env` 的 `LANGSMITH_TRACING` 决定，不受影响。
"""

import langsmith.utils as ls_utils
import pytest


@pytest.fixture(autouse=True)
def disable_langsmith_tracing(monkeypatch):
    """测试期间让 LangSmith 判定「追踪关闭」。

    判定的入口是 `langsmith.utils.tracing_is_enabled()`，
    `langchain_core.tracers.context` 通过模块属性调用它 ——
    所以替换这个属性就能生效，不用改环境变量
    （项目的 .env 用 `load_dotenv(override=True)`，会盖掉环境变量）。
    """
    monkeypatch.setattr(ls_utils, "tracing_is_enabled", lambda *a, **k: False)
