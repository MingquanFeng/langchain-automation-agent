"""进程内会话占用表，统一 WebSocket、同步任务和会话删除的互斥规则。

同一个 thread_id 的检查点不能被两个执行入口同时修改。申请失败立即报忙，
避免后台排队的任务在用户不知情时执行；不同会话仍可并发。
占用必须保持到实际工作线程退出，取消 asyncio 等待器不代表同步工作已经停止。
本模块只协调同一个 Python 进程，多进程部署需要额外的跨进程锁。
"""

from __future__ import annotations

# threading：同步任务与事件循环共用同一张占用表。
import threading


# ============================================================
# 1. 占用异常与可释放的会话凭据
# ============================================================

class RunBusyError(RuntimeError):
    """连接或会话已有运行中的任务，调用方应拒绝本次启动。"""


class ThreadLease:
    """一次会话占用；支持 with，并允许重复释放。"""

    def __init__(self, registry: ThreadRegistry, thread_id: str) -> None:
        self.registry = registry
        self.thread_id = thread_id

    def release(self) -> None:
        """仅释放自己的占用，不影响后来取得同一会话的调用方。"""
        with self.registry._lock:
            if self.registry._leases.get(self.thread_id) is self:
                del self.registry._leases[self.thread_id]

    def __enter__(self) -> ThreadLease:
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


# ============================================================
# 2. 共享登记表
# ============================================================

class ThreadRegistry:
    """按 thread_id 原子申请占用，锁只保护登记，不包住实际任务。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._leases: dict[str, ThreadLease] = {}

    def claim(self, thread_id: str) -> ThreadLease:
        """取得会话凭据；会话正忙时抛 RunBusyError，不创建运行记录。"""
        with self._lock:
            if thread_id in self._leases:
                raise RunBusyError("该会话仍有任务运行或正在取消，请等待结束后再试")
            lease = ThreadLease(self, thread_id)
            self._leases[thread_id] = lease
            return lease


# 所有执行入口复用这个实例，不能每个 RunManager 单独创建。
THREAD_RUNS = ThreadRegistry()
