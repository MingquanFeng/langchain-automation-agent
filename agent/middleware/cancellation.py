"""模型与工具的协作式取消边界。

对应 LangGraph 课件 04 章 10.2.3 的调用包装机制。中间件放在重试层内侧，
每次实际尝试前后检查上下文中的取消标志；请求已经取消时不再开始新调用。
正在执行的同步 I/O 仍须等待自身超时或正常返回，不能用遗留后台线程伪造终止。
"""

from langchain.agents.middleware import AgentMiddleware

# 取消异常单独传播，避免重试中间件把它当作普通失败或模型回答。
from agent.context import TaskCancelledError, check_cancelled


# ============================================================
# 1. 重试分类与工具失败转换
# ============================================================

def retry_unless_cancelled(exc: Exception) -> bool:
    """只允许普通失败进入现有重试策略。"""
    return not isinstance(exc, TaskCancelledError)


def tool_failure_message(exc: Exception) -> str:
    """普通工具失败交给模型处理；取消直接终止整轮任务。"""
    if isinstance(exc, TaskCancelledError):
        raise exc
    return f"工具执行失败：{type(exc).__name__}: {exc}"


# ============================================================
# 2. 实际调用前后的取消检查
# ============================================================

class CancellationMiddleware(AgentMiddleware):
    """守住每次模型请求和工具尝试；允许没有取消上下文的旧入口调用。"""

    def _call(self, request, handler):
        ctx = getattr(getattr(request, "runtime", None), "context", None)
        check_cancelled(ctx)
        try:
            result = handler(request)
        except Exception:
            # 取消后才返回的网络错误也按取消结束，不能再次发起请求。
            check_cancelled(ctx)
            raise
        check_cancelled(ctx)
        return result

    def wrap_model_call(self, request, handler):
        """每次模型尝试都检查，包括重试层再次进入的调用。"""
        return self._call(request, handler)

    def wrap_tool_call(self, request, handler):
        """每次工具尝试都检查；已开始的副作用不会被回滚。"""
        return self._call(request, handler)
