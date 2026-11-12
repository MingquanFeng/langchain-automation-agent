
"""
FastAPI 服务：定时任务管理 + 调度状态查询。

课件未涉及部署，本层按 langgraph dev 之外的路径用 FastAPI 实现。

对应课件：
课件未覆盖 HTTP 服务，需要自行实现。

主要职责：
1. 在应用启动时拉起调度器，关闭时停掉。
2. 暴露定时任务的增删查接口与调度器状态查询。
3. 健康检查。

注意：
FastAPI 的 @app.on_event 已废弃，
本文件统一用 lifespan 上下文管理器写启动/关闭逻辑。
"""

from __future__ import annotations

# asyncio：起推送任务。
import asyncio

# json：解析 WebSocket 收到的消息。
import json

# logging：标准库的日志模块。
import logging
from pathlib import Path

# uuid：生成连接 id。
import uuid

# asynccontextmanager：把生成器包装成异步上下文管理器，
# 用于定义 FastAPI 的 lifespan。
from contextlib import asynccontextmanager

# FastAPI：Web 框架。
# WebSocket / WebSocketDisconnect：桌面版的事件流端点。
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

# BaseModel / Field：Pydantic 模型与字段声明，
# FastAPI 用它做请求体校验和 OpenAPI 文档。
from pydantic import BaseModel, Field

# RUNTIME：应用运行配置，这里取调度开关和通知渠道。
from agent.config import RUNTIME

# AgentContext：单次调用的运行时上下文。
from agent.context import AgentContext

# get_agent：进程内唯一的 Agent 单例（见 runner.py）。
from .runner import get_agent

# BaseEvent / RunError / RunStart：事件模型。
from .events import BaseEvent, RunError, RunStart

# RunBusyError / RunManager：run 的生命周期管理。
from .runs import RunBusyError, RunManager

# list_jobs / start_scheduler / stop_scheduler：调度器接口。
from .scheduler import list_jobs, start_scheduler, stop_scheduler

# ScheduledTask / add_task / all_tasks / remove_task：任务表接口。
from .tasks import ScheduledTask, add_task, all_tasks, remove_task

logger = logging.getLogger(__name__)
WORKBENCH_HTML = Path(__file__).resolve().parent.parent / "frontend" / "workbench.html"


# ============================================================
# 1. 应用生命周期
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子。

    FastAPI 0.135 起 @app.on_event 已废弃，改用 lifespan 写法。

    结构：
    yield 之前是启动逻辑，
    yield 之后是关闭逻辑。

    启动逻辑：
    按 RUNTIME.scheduler_enabled 决定要不要拉起调度器。
    注意这个开关是 agent/config.py 在模块导入时读配置固化的，
    所以 --no-schedule 必须在导入本模块之前设置好命令行覆盖
    （见 agent/config.py 的 set_cli_override）。

    关闭逻辑：
    停掉调度器。
    Agent 单例本身不用管，
    它的 Checkpointer 连接随进程退出一起释放。
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if RUNTIME.scheduler_enabled:
        start_scheduler()
        logger.info("定时任务已启动")

    # Run Manager 挂在 app.state 上 ——
    # 每个应用实例一个，测试之间不共享状态。
    app.state.run_manager = RunManager()

    yield
    stop_scheduler()


# app：FastAPI 应用实例。
#
# lifespan 通过构造参数传进去，
# 这是 FastAPI 0.135+ 推荐的写法，
# 不要再用 @app.on_event("startup") / @app.on_event("shutdown")。
app = FastAPI(title="Automation Agent", version="0.1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def workbench() -> FileResponse:
    """从共享前端目录返回本地 Agent 调试工作台。"""
    return FileResponse(WORKBENCH_HTML)


# ============================================================
# 2. 请求体模型
# ============================================================

class ScheduledTaskRequest(BaseModel):
    """
    POST /tasks 的请求体：新增一个定时任务。

    注意：
    这个模型描述的是「什么时候跑」，
    不是「跑一次」—— 后者由 runner.run_task() 直接接收，
    不经过 HTTP。
    """

    # 任务名。
    name: str

    # cron 表达式，如 '0 9 * * *'。
    # 标准 5 段格式。
    cron: str = Field(..., description="cron 表达式，如 '0 9 * * *'")

    # 任务指令。
    instruction: str

    # 是否允许写操作。
    allow_write: bool = False


# ============================================================
# 3. 状态查询接口
# ============================================================

@app.get("/health")
def health() -> dict:
    """
    健康检查。

    返回：
        status：固定为 "ok"。
        scheduler：调度器开关是否打开。

    注意：
    这里返回的是「配置上调度器是否启用」，
    不是「调度器此刻是否真的在跑」。
    两者在服务启动失败时可能不一致。
    """
    return {"status": "ok", "scheduler": RUNTIME.scheduler_enabled}


@app.get("/tasks")
def list_scheduled_tasks() -> list[dict]:
    """
    列出定时任务表。

    返回：
        每个元素含 name、cron、instruction、allow_write。

    注意：
    查的是 tasks.py 里的内存字典，
    里面包含 POST /tasks 动态加的任务（进程重启后就没了）。
    想看调度器实际注册了什么，用 /jobs。
    """
    return [
        {
            "name": t.name,
            "cron": t.cron,
            "instruction": t.instruction,
            "allow_write": t.allow_write,
        }
        for t in all_tasks()
    ]


@app.get("/jobs")
def jobs() -> list[dict]:
    """
    列出调度器已注册的 job 和下次运行时间。

    返回：
        每个元素含 id 和 next_run。

    注意：
    调度器没启动时返回空列表，
    不会报错。
    """
    return list_jobs()


# ============================================================
# 4. 定时任务增删接口
# ============================================================

@app.post("/tasks")
def create_task(req: ScheduledTaskRequest) -> dict:
    """
    动态添加一个定时任务。

    返回：
        created：任务名。
        cron：注册进去的 cron 表达式。

    注意：
    add_task() 只改内存字典，
    必须再调一次 start_scheduler() 才能让调度器真正认识它。
    start_scheduler() 内部会先停掉旧调度器再重建，
    所以这里直接调用是安全的。

    注意：
    任务只存在内存里，
    进程重启就没了。要持久化得改 tasks.py 的 TASKS 列表。
    """
    task = ScheduledTask(
        name=req.name,
        cron=req.cron,
        instruction=req.instruction,
        allow_write=req.allow_write,
    )
    add_task(task)

    # 重建调度器，让新任务立刻生效。
    start_scheduler()
    return {"created": task.name, "cron": task.cron}


@app.delete("/tasks/{name}")
def delete_task(name: str) -> dict:
    """
    删除一个定时任务。

    返回：
        removed：被删的任务名。
        ok：是否真的删掉了（任务本来就不存在时为 False）。

    注意：
    同样需要重启调度器才生效，
    这里已经顺手调了 start_scheduler()。
    """
    removed = remove_task(name)
    start_scheduler()
    return {"removed": name, "ok": removed}


# ============================================================
# 6. WebSocket 事件流端点（桌面版）
# ============================================================

# WS_QUEUE_SIZE：每个连接的事件队列容量。
#
# 有界 + 背压：客户端消费不过来时，生产者会停下来
# （见 runtime/consumer.py 的两层上限说明）。
WS_QUEUE_SIZE = 256


@app.websocket("/ws")
async def websocket_events(websocket: WebSocket) -> None:
    """桌面版的事件流端点。

    **客户端 → 服务端**：

    ```json
    {"type": "run.start", "instruction": "列出数据目录下的文件",
     "task_name": "列文件", "thread_id": "可选"}
    {"type": "cancel"}
    {"type": "approval.respond", "run_id": "...", "call_id": "...", "approved": true}
    ```

    浏览器工作台不传 `allow_write`，有副作用的工具会逐次请求批准。
    为兼容旧客户端，显式传 `allow_write: true` 仍代表整轮授权。

    **服务端 → 客户端**：一条一个 JSON 事件。
    `run.start` / `model.text` / `tool.start` / `tool.end` / `tool.error` /
    `approval.request` / `run.progress` / `run.end` / `run.cancelled` / `run.error`

    注意：
    1. **单一发送点** —— Starlette 的 `WebSocket.send` 没有锁，
       并发发帧会交错。所以所有要发给客户端的消息
       （事件的 + 端点自己产生的）都进同一个队列，
       由唯一的推送任务发出去。
    2. **单连接单任务** —— 同一连接已有活跃 run 时，
       新的 run.start 会被拒绝，回一条 run.error，
       而不是静默取消旧的。
    3. **连接断开要清理** —— 断开时取消这个连接上的所有 run，
       否则消费线程会一直跑下去。
    """
    await websocket.accept()

    manager: RunManager = websocket.app.state.run_manager
    conn_id = uuid.uuid4().hex[:8]
    queue: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=WS_QUEUE_SIZE)

    async def pump() -> None:
        """唯一的发送点：把队列里的事件推给客户端。"""
        while True:
            event = await queue.get()
            await websocket.send_json(event.model_dump())

    pump_task = asyncio.create_task(pump())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await queue.put(RunError(run_id="", message="消息不是合法 JSON"))
                continue
            if not isinstance(payload, dict):
                await queue.put(RunError(run_id="", message="消息必须是 JSON 对象"))
                continue
            await _handle_ws_command(
                manager=manager,
                queue=queue,
                conn_id=conn_id,
                payload=payload,
            )
    except WebSocketDisconnect:
        logger.info("WebSocket 断开：%s", conn_id)
    finally:
        # 断开：取消这个连接上的所有 run。
        # 不取消的话，消费线程会一直跑到 agent 结束。
        for handle in manager.active_for(conn_id):
            manager.cancel(handle.run_id)
        pump_task.cancel()


async def _handle_ws_command(
    *,
    manager: RunManager,
    queue: asyncio.Queue,
    conn_id: str,
    payload: dict,
) -> None:
    """处理一条客户端命令。

    注意：
    产生的消息都进 queue，由唯一的推送任务发出去 ——
    端点自己不直接 send。
    """
    command = payload.get("type")

    if command == "run.start":
        await _handle_run_start(manager=manager, queue=queue, conn_id=conn_id, payload=payload)
    elif command == "cancel":
        await _handle_ws_cancel(manager=manager, queue=queue, conn_id=conn_id)
    elif command == "approval.respond":
        await _handle_approval_respond(manager=manager, queue=queue, conn_id=conn_id, payload=payload)
    else:
        await queue.put(RunError(run_id="", message=f"未知命令：{command}"))


async def _handle_approval_respond(*, manager: RunManager, queue: asyncio.Queue, conn_id: str, payload: dict) -> None:
    """校验授权应答，并确保只有发起 run 的连接能完成对应请求。"""
    run_id, call_id, approved = payload.get("run_id"), payload.get("call_id"), payload.get("approved")
    valid_run_id = run_id if isinstance(run_id, str) else ""
    if not isinstance(run_id, str) or not run_id or not isinstance(call_id, str) or not call_id or type(approved) is not bool:
        await queue.put(RunError(run_id=valid_run_id, message="approval.respond 需要非空 run_id、非空 call_id 和布尔值 approved", recoverable=True))
        return
    if not manager.respond_approval(run_id, conn_id, call_id, approved):
        await queue.put(RunError(run_id=run_id, message="授权请求不存在、已处理或不属于当前连接", recoverable=True))
        return
    # 应答确认属于协议控制消息，不另发事件；工具后续会产生 tool.start / tool.error。


async def _handle_run_start(
    *,
    manager: RunManager,
    queue: asyncio.Queue,
    conn_id: str,
    payload: dict,
) -> None:
    """启动一次执行。"""
    instruction = str(payload.get("instruction", "")).strip()
    if not instruction:
        await queue.put(RunError(run_id="", message="缺少 instruction"))
        return

    task_name = str(payload.get("task_name") or instruction[:30])
    thread_id = payload.get("thread_id") or None
    # 写权限默认关闭 —— 只读是默认态，要写必须显式开。
    allow_write = bool(payload.get("allow_write", False))

    ctx = AgentContext(
        task_name=task_name,
        # WebSocket 是 HTTP 层触发的，用 api。
        trigger="api",
        allow_write=allow_write,
        notify_channel=RUNTIME.notify_channel,
    )

    try:
        handle = manager.start(
            owner=conn_id,
            instruction=instruction,
            task_name=task_name,
            thread_id=thread_id,
            agent=get_agent(),
            queue=queue,
            loop=asyncio.get_running_loop(),
            ctx=ctx,
        )
    except RunBusyError:
        # 上一轮还在运行 —— 拒绝，不静默取消旧的。
        await queue.put(RunError(run_id="", message="上一轮还在运行"))
        return

    await queue.put(RunStart(
        run_id=handle.run_id,
        task_name=task_name,
        thread_id=handle.thread_id,
        instruction=instruction,
    ))


async def _handle_ws_cancel(
    *,
    manager: RunManager,
    queue: asyncio.Queue,
    conn_id: str,
) -> None:
    """取消当前连接上的活跃 run。

    注意：
    取消是协作式的 —— 正在跑的那一步会做完，
    取消信号最终表现为一条 run.cancelled 事件（由消费器发）。
    """
    active = manager.active_for(conn_id)
    if not active:
        await queue.put(RunError(run_id="", message="没有正在运行的任务"))
        return
    manager.cancel(active[0].run_id)
