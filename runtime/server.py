
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

# uuid：生成连接 id。
import uuid

# asynccontextmanager：把生成器包装成异步上下文管理器，
# 用于定义 FastAPI 的 lifespan。
from contextlib import asynccontextmanager

# FastAPI：Web 框架。
# WebSocket / WebSocketDisconnect：桌面版的事件流端点。
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

# BaseModel / Field：Pydantic 模型与字段声明，
# FastAPI 用它做请求体校验和 OpenAPI 文档。
from pydantic import BaseModel, Field
from apscheduler.triggers.cron import CronTrigger

# RUNTIME：应用运行配置，这里取调度开关和通知渠道。
from agent.config import RUNTIME

# AgentContext：单次调用的运行时上下文。
from agent.context import AgentContext

# get_agent：进程内唯一的 Agent 单例（见 runner.py）。
from .runner import get_agent

# BaseEvent / RunError / RunStart：事件模型。
from .events import BaseEvent, RunError, RunStart, THINK_RE

# RunBusyError / RunManager：run 的生命周期管理。
from .runs import RunBusyError, RunManager

# list_jobs / start_scheduler / stop_scheduler：调度器接口。
from .scheduler import list_jobs, start_scheduler, stop_scheduler
from .storage import get_runtime_store

# ScheduledTask / add_task / all_tasks / remove_task：任务表接口。
from .tasks import ScheduledTask, add_task, all_tasks, configure_persistence, remove_task

logger = logging.getLogger(__name__)


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
    store = get_runtime_store()
    store.initialize()
    store.mark_interrupted_runs()
    configure_persistence(store)
    app.state.runtime_store = store

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
app.add_middleware(
    CORSMiddleware,
    # Tauri 本地 WebView 用应用源直接访问 loopback HTTP API；普通浏览器工作台已迁入桌面包。
    allow_origins=["tauri://localhost", "http://tauri.localhost", "https://tauri.localhost"],
    allow_methods=["GET", "DELETE"],
    allow_headers=["*"],
)


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

    查询持久化任务注册表。想看调度器实际注册了什么，用 /jobs。
    """
    return [
        {
            "name": t.name,
            "cron": t.cron,
            "instruction": t.instruction,
            "allow_write": t.allow_write,
            "enabled": t.enabled,
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


@app.get("/runs")
def list_agent_runs(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    """分页查询最近的 Agent 执行记录，包括结果、错误和状态。"""
    return app.state.runtime_store.list_runs(limit=limit, offset=offset)


def _checkpoint_message_text(content: object) -> str:
    """将 Checkpointer 消息正文转成可展示的纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if parts:
            return "".join(parts)
        return "[包含非文本内容]" if content else ""
    if content is None:
        return ""
    return str(content)


def _serialize_checkpoint_messages(thread_id: str) -> dict:
    """读取会话最新状态中的对话消息，不返回系统提示或内部思考内容。"""
    try:
        agent = get_agent()
        snapshot = agent.get_state({"configurable": {"thread_id": thread_id}})
    except Exception as exc:
        logger.exception("读取会话检查点失败：%s", thread_id)
        raise HTTPException(status_code=503, detail="暂时无法读取会话内容") from exc

    raw_messages = (snapshot.values or {}).get("messages", []) if snapshot else []
    messages: list[dict] = []
    for message in raw_messages:
        if isinstance(message, dict):
            message_type = message.get("type") or message.get("role")
            content = message.get("content")
            name = message.get("name")
            tool_calls = message.get("tool_calls") or []
        else:
            message_type = getattr(message, "type", "")
            content = getattr(message, "content", None)
            name = getattr(message, "name", None)
            tool_calls = getattr(message, "tool_calls", []) or []

        if message_type in ("human", "user"):
            role = "user"
        elif message_type in ("ai", "assistant"):
            role = "assistant"
        elif message_type == "tool":
            role = "tool"
        else:
            # 不把 system / developer / function 等内部控制消息展示给用户。
            continue

        text = _checkpoint_message_text(content)
        if role == "assistant":
            text = THINK_RE.sub("", text)
            if not text.strip() and tool_calls:
                calls = []
                for call in tool_calls:
                    if isinstance(call, dict):
                        call_name = call.get("name", "工具")
                        arguments = call.get("args", {})
                    else:
                        call_name = getattr(call, "name", "工具")
                        arguments = getattr(call, "args", {})
                    calls.append(f"{call_name}({arguments})")
                text = "调用工具：" + "、".join(calls)
        if not text:
            continue

        item = {"role": role, "content": text}
        if role == "tool" and name:
            item["name"] = str(name)
        messages.append(item)

    limit = 500
    return {
        "thread_id": thread_id,
        "messages": messages[-limit:],
        "message_count": len(messages),
        "truncated": len(messages) > limit,
    }


@app.get("/threads/messages")
def get_thread_messages(
    thread_id: str = Query(..., min_length=1, max_length=256),
) -> dict:
    """从 Checkpointer 读取会话对话；与运行历史列表分开返回。"""
    return _serialize_checkpoint_messages(thread_id)


@app.delete("/runs/{run_id}")
def delete_agent_run(run_id: str) -> dict:
    """删除单条运行记录，不删除其会话记忆。"""
    store = app.state.runtime_store
    if store.is_run_running(run_id):
        raise HTTPException(status_code=409, detail="运行中的记录暂时不能删除")
    deleted = store.delete_run(run_id)
    return {"run_id": run_id, "deleted": deleted}


@app.delete("/runs")
def clear_agent_runs() -> dict:
    """清空运行记录；会话 Checkpointer 与定时任务定义保留。"""
    store = app.state.runtime_store
    if store.has_running_runs():
        raise HTTPException(status_code=409, detail="仍有任务运行，暂时不能清空运行记录")
    deleted = store.clear_runs()
    return {"deleted": deleted}


@app.delete("/threads")
def delete_thread(
    thread_id: str = Query(..., min_length=1, max_length=256),
) -> dict:
    """删除会话检查点以及关联的全部运行记录。"""
    manager: RunManager = app.state.run_manager
    store = app.state.runtime_store
    if (
        any(run.thread_id == thread_id for run in manager.active())
        or store.has_running_run_for_thread(thread_id)
    ):
        raise HTTPException(status_code=409, detail="会话仍有运行中的任务，暂时不能删除")

    try:
        agent = get_agent()
    except Exception as exc:
        logger.exception("初始化会话存储失败：%s", thread_id)
        raise HTTPException(status_code=503, detail="暂时无法访问会话存储") from exc

    checkpointer = getattr(agent, "checkpointer", None)
    delete_checkpoint = getattr(checkpointer, "delete_thread", None)
    if not callable(delete_checkpoint):
        raise HTTPException(status_code=501, detail="当前 Checkpointer 不支持删除会话")
    try:
        delete_checkpoint(thread_id)
    except Exception as exc:
        logger.exception("删除会话检查点失败：%s", thread_id)
        raise HTTPException(status_code=500, detail="删除会话检查点失败") from exc

    deleted_runs = store.delete_runs_for_thread(thread_id)
    return {"thread_id": thread_id, "deleted_runs": deleted_runs}


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

    任务会先持久化，再重建调度器使其立即生效。
    start_scheduler() 内部会先停掉旧调度器再重建，
    所以这里直接调用是安全的。

    """
    try:
        CronTrigger.from_crontab(req.cron, timezone=RUNTIME.timezone)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"无效的 cron 表达式：{exc}") from exc

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

    桌面工作台不传 `allow_write`，有副作用的工具会逐次请求批准。
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

    # Agent 在首次调用时才惰性初始化。配置缺失（例如桌面版尚未填 API Key）
    # 必须转换成协议错误；否则异常会逃出 WebSocket handler，连接关闭后 UI 会一直显示运行中。
    try:
        agent = get_agent()
    except Exception as exc:
        logger.error("Agent 初始化失败（%s）；请检查模型配置和 API Key。", type(exc).__name__)
        await queue.put(RunError(
            run_id="",
            message="Agent 初始化失败。请检查桌面应用支持目录中的 .env，确认 LLM_API_KEY、模型供应商和模型名称。",
        ))
        return

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
            agent=agent,
            queue=queue,
            loop=asyncio.get_running_loop(),
            ctx=ctx,
            on_started=lambda run: app.state.runtime_store.start_run(
                run_id=run.run_id,
                thread_id=run.thread_id,
                task_name=run.task_name,
                trigger=ctx.trigger,
                instruction=run.instruction,
                started_at=run.started_at,
            ),
            on_terminal=lambda event: _record_terminal_event(
                app.state.runtime_store, event
            ),
        )
    except RunBusyError:
        # 上一轮还在运行 —— 拒绝，不静默取消旧的。
        await queue.put(RunError(run_id="", message="上一轮还在运行"))
        return
    except Exception:
        logger.exception("无法登记 Agent 运行记录")
        await queue.put(RunError(run_id="", message="无法保存运行记录；本次任务未启动。"))
        return

    await queue.put(RunStart(
        run_id=handle.run_id,
        task_name=task_name,
        thread_id=handle.thread_id,
        instruction=instruction,
    ))


def _record_terminal_event(store, event: BaseEvent) -> None:
    """把 WebSocket run 的最终事件同步到本地运行记录。"""
    if event.type == "run.end":
        status = "succeeded"
        output = getattr(event, "output", "")
        error = None
    elif event.type == "run.cancelled":
        status = "cancelled"
        output = ""
        error = getattr(event, "reason", "") or None
    else:
        status = "failed"
        output = ""
        error = getattr(event, "message", "运行失败")
    try:
        store.finish_run(
            event.run_id,
            status=status,
            output=output,
            error=error,
            duration_ms=getattr(event, "duration_ms", None),
            message_count=getattr(event, "messages", None),
        )
    except Exception:
        # 运行记录写入失败不能中断事件推送或改变 Agent 的实际结果。
        logger.exception("无法保存 Agent 运行结果：%s", event.run_id)


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
