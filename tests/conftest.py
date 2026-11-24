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

import os
import shutil
import tempfile
from pathlib import Path

import langsmith.utils as ls_utils
import pytest
from langgraph.store.memory import InMemoryStore


# ============================================================
# 1. 测试进程隔离
# ============================================================

# 必须在收集测试模块前设置。agent.config 在首次导入时读取
# AUTOMATION_AGENT_HOME：把它指到每次 pytest 独有的临时目录后，
# 测试既不会读取开发者的项目 .env，也不会复用 data/ 下的 SQLite 数据库。
#
# 临时目录由 pytest 会话结束钩子删除；即使某个测试失败，也不会把测试
# 产生的记忆库、运行记录或备份留在仓库中。
_TEST_APP_HOME = Path(tempfile.mkdtemp(prefix="automation-agent-tests-"))
os.environ["AUTOMATION_AGENT_HOME"] = str(_TEST_APP_HOME)


def pytest_sessionfinish(session, exitstatus) -> None:
    """删除本次测试专用的应用目录，避免 /tmp 遗留 SQLite 数据。"""
    shutil.rmtree(_TEST_APP_HOME, ignore_errors=True)


@pytest.fixture(autouse=True)
def disable_langsmith_tracing(monkeypatch):
    """测试期间让 LangSmith 判定「追踪关闭」。

    判定的入口是 `langsmith.utils.tracing_is_enabled()`，
    `langchain_core.tracers.context` 通过模块属性调用它 ——
    所以替换这个属性就能生效，不用改环境变量
    （项目的 .env 用 `load_dotenv(override=True)`，会盖掉环境变量）。
    """
    monkeypatch.setattr(ls_utils, "tracing_is_enabled", lambda *a, **k: False)


@pytest.fixture(autouse=True)
def isolate_long_term_store(monkeypatch):
    """测试默认使用内存 Store，避免把假数据写入真实长期记忆库。

    生产代码的 build_agent() 仍默认调用 SQLite Store；专门的 Store
    回归测试会直接调用原始构建函数验证持久化、权限和命名空间隔离。
    """
    import agent.graph as graph_module

    monkeypatch.setattr(graph_module, "_build_store", InMemoryStore)
