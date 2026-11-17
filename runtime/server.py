
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
from dataclasses import dataclass
from secrets import compare_digest
from typing import Annotated, Literal

# FastAPI：Web 框架。
# WebSocket / WebSocketDisconnect：桌面版的事件流端点。
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

# BaseModel / Field：Pydantic 模型与字段声明，
# FastAPI 用它做请求体校验和 OpenAPI 文档。
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError
from apscheduler.triggers.cron import CronTrigger

# RUNTIME：应用运行配置，这里取调度开关和通知渠道。
from agent.config import RUNTIME

# AgentContext：单次调用的运行时上下文。
from agent.context import AgentContext

# 长期记忆服务：HTTP 管理入口与 Agent 工具共用同一套校验和 CRUD。
from agent.memory import (
    MAX_MEMORY_CONTENT_CHARS,
    MemoryConflictError,
    MemoryValidationError,
    delete_memory_record,
    list_memory_records,
    save_memory_record,
)

# get_agent：进程内唯一的 Agent 单例（见 runner.py）。
from .runner import get_agent

# BaseEvent / RunError / RunStart：事件模型。
from .events import BaseEvent, RunError, RunStart, RunProgress, THINK_RE

# RunBusyError / RunManager：run 的生命周期管理。
from .runs import RunBusyError, RunManager
from .concurrency import THREAD_RUNS

# list_jobs / start_scheduler / stop_scheduler：调度器接口。
from .scheduler import list_jobs, start_scheduler, stop_scheduler
from .storage import get_runtime_store

# ScheduledTask / add_task / all_tasks / remove_task：任务表接口。
from .tasks import ScheduledTask, add_task, all_tasks, configure_persistence, remove_task

logger = logging.getLogger(__name__)

# 仅托管 sidecar 使用这组 WebView 来源和版本协议。
DESKTOP_ORIGINS = ("tauri://localhost", "http://tauri.localhost", "https://tauri.localhost")
DESKTOP_WS_PROTOCOL = "automation-agent.v1"


@dataclass(frozen=True, repr=False)
class DesktopAccess:
    """保存单次 sidecar 启动的访问凭证和健康检查标记。"""

    token: str
    readiness: str

    def accepts(self, candidate: str) -> bool:
        """以恒定时间比较凭证，非 ASCII 输入直接拒绝。"""
        return candidate.isascii() and compare_digest(candidate, self.token)


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

    if RUNTIME.scheduler_enabled and getattr(app.state, "desktop_access", None) is None:
        start_scheduler()
        logger.info("定时任务已启动")

    # Run Manager 挂在 app.state 上 ——
    # 每个应用实例一个，测试之间不共享状态。
    app.state.run_manager = RunManager()

    yield
    for handle in app.state.run_manager.active():
        app.state.run_manager.cancel(handle.run_id, disconnect=True)
    stop_scheduler()
    # Starlette 的 TestClient 会在测试夹具回收时还原 app.state 属性。
    # 保持这个可选状态存在，避免下一次生命周期启动时读取不到它。
    app.state.desktop_access = None


# app：FastAPI 应用实例。
#
# lifespan 通过构造参数传进去，
# 这是 FastAPI 0.135+ 推荐的写法，
# 不要再用 @app.on_event("startup") / @app.on_event("shutdown")。
app = FastAPI(title="阿序 · Axu", version="0.1.0", lifespan=lifespan)
app.state.desktop_access = None


@app.middleware("http")
async def authorize_desktop_http(request: Request, call_next):
    """保护托管服务的 HTTP 接口，健康检查仅提供启动状态。"""
    access = getattr(request.app.state, "desktop_access", None)
    if access is None or (request.method == "GET" and request.url.path == "/health"):
        return await call_next(request)

    header = request.headers.get("authorization", "")
    scheme, _, credential = header.partition(" ")
    if scheme.lower() != "bearer" or not access.accepts(credential):
        return JSONResponse(
            {"detail": "需要桌面服务凭证"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
        )
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


# CORS 放在凭证校验外层，使 WebView 的预检和 401 响应都能正常返回。
app.add_middleware(
    CORSMiddleware,
    # Tauri 本地 WebView 用应用源直接访问 loopback HTTP API；普通浏览器工作台已迁入桌面包。
    allow_origins=DESKTOP_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
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


class MemoryWriteRequest(BaseModel):
    """POST /memories 与 PUT /memories/{id} 的显式记忆请求。"""

    # 拒绝未声明字段，避免客户端拼错字段后服务端静默忽略。
    model_config = ConfigDict(extra="forbid")

    # user 跨项目共享；project 只属于当前工作目录。
    scope: Literal["user", "project"] = "project"

    # 只允许偏好和事实，不接受完整对话等泛化类别。
    memory_type: Literal["preference", "fact"] = "preference"

    # Pydantic 先做基础长度校验，agent.memory 仍会 strip 后再次校验。
    content: str = Field(min_length=1, max_length=MAX_MEMORY_CONTENT_CHARS)

    # 固定策略让界面无需处理任意日期，同时覆盖临时事实和长期偏好。
    expiration_policy: Literal["never", "30_days", "90_days", "365_days"] = "never"

    # 新建时省略；修改时必须带上列表中看到的修订号，服务端据此拒绝
    # 已经过期的编辑表单，避免多个入口静默互相覆盖。
    expected_revision: int | None = Field(default=None, ge=1)


# ============================================================
# 3. 状态查询接口
# ============================================================

@app.get("/health")
def health(request: Request) -> dict:
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
    access = getattr(request.app.state, "desktop_access", None)
    if access is not None:
        return {"status": "ok", "scheduler": False, "readiness": access.readiness}
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


# ============================================================
# 4. 长期记忆管理接口
# ============================================================

def _agent_memory_store():
    """取得与 Agent 工具共用的 Store，避免 HTTP 与聊天入口数据分叉。"""
    try:
        store = getattr(get_agent(), "store", None)
    except Exception as exc:
        logger.exception("初始化长期记忆 Store 失败")
        raise HTTPException(status_code=503, detail="暂时无法访问长期记忆") from exc
    if store is None:
        raise HTTPException(status_code=503, detail="当前 Agent 未配置长期记忆 Store")
    return store


@app.get("/memories")
def list_memories_api(
    scope: Literal["user", "project"] = Query(default="project"),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """分页查看用户明确保存的当前用户级或项目级长期记忆。"""
    try:
        return list_memory_records(
            _agent_memory_store(),
            scope=scope,
            limit=limit,
            offset=offset,
        )
    except MemoryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("读取长期记忆失败：scope=%s", scope)
        raise HTTPException(status_code=503, detail="暂时无法读取长期记忆") from exc


@app.post("/memories", status_code=201)
def create_memory_api(req: MemoryWriteRequest) -> dict:
    """由用户在管理界面明确提交一条新的长期记忆。"""
    try:
        return save_memory_record(
            _agent_memory_store(),
            scope=req.scope,
            memory_kind=req.memory_type,
            content=req.content,
            source="settings",
            expiration_policy=req.expiration_policy,
            expected_revision=req.expected_revision,
        )
    except MemoryConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MemoryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("创建长期记忆失败：scope=%s", req.scope)
        raise HTTPException(status_code=500, detail="创建长期记忆失败") from exc


@app.put("/memories/{memory_id}")
def update_memory_api(memory_id: str, req: MemoryWriteRequest) -> dict:
    """由用户明确修改当前作用域中一条已存在的长期记忆。"""
    try:
        return save_memory_record(
            _agent_memory_store(),
            scope=req.scope,
            memory_kind=req.memory_type,
            content=req.content,
            memory_id=memory_id,
            source="settings",
            expiration_policy=req.expiration_policy,
            expected_revision=req.expected_revision,
        )
    except MemoryConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MemoryValidationError as exc:
        status = 404 if str(exc) == "要修改的长期记忆不存在" else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("修改长期记忆失败：%s", memory_id)
        raise HTTPException(status_code=500, detail="修改长期记忆失败") from exc


@app.delete("/memories/{memory_id}")
def delete_memory_api(
    memory_id: str,
    scope: Literal["user", "project"] = Query(default="project"),
    expected_revision: int = Query(ge=1),
) -> dict:
    """由用户明确删除当前作用域中的一条长期记忆。"""
    try:
        deleted = delete_memory_record(
            _agent_memory_store(),
            scope=scope,
            memory_id=memory_id,
            expected_revision=expected_revision,
        )
    except MemoryConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MemoryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("删除长期记忆失败：%s", memory_id)
        raise HTTPException(status_code=500, detail="删除长期记忆失败") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="长期记忆不存在")
    return {"id": memory_id, "scope": scope, "deleted": True}


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


def _serialize_checkpoint_message_list(raw_messages: list) -> list[dict]:
    """把 Checkpointer 消息转为工作台可展示的内容，过滤内部控制消息。"""
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
        # ContextInjectionMiddleware 的权限提示用于下一次模型调用，不是 Agent
        # 对用户的回答。新检查点使用 name 标记；两个固定文本用于兼容已保存的旧数据。
        if role == "assistant" and (
            name == "context_hint"
            or text in {
                "（提示：写入或外部操作需要对具体工具调用单独授权；未授权的操作不会执行。）",
                "（提示：本轮为只读模式，任何写操作都会被拒绝。）",
            }
        ):
            continue
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

    return messages


def _serialize_checkpoint_messages(thread_id: str) -> dict:
    """读取会话最新状态中的对话消息，不返回系统提示或内部思考内容。"""
    try:
        agent = get_agent()
        snapshot = agent.get_state({"configurable": {"thread_id": thread_id}})
    except Exception as exc:
        logger.exception("读取会话检查点失败：%s", thread_id)
        raise HTTPException(status_code=503, detail="暂时无法读取会话内容") from exc

    raw_messages = (snapshot.values or {}).get("messages", []) if snapshot else []
    messages = _serialize_checkpoint_message_list(raw_messages)

    limit = 500
    return {
        "thread_id": thread_id,
        "messages": messages[-limit:],
        "message_count": len(messages),
        "truncated": len(messages) > limit,
    }


@app.get("/threads")
def list_checkpoint_threads(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0, le=10000),
) -> dict:
    """分页列出已保存的会话，支持运行记录表尚无条目的旧会话。"""
    try:
        agent = get_agent()
        checkpointer = getattr(agent, "checkpointer", None)
        list_checkpoints = getattr(checkpointer, "list", None)
        if not callable(list_checkpoints):
            raise HTTPException(status_code=501, detail="当前 Checkpointer 不支持列出会话")

        threads: list[dict] = []
        seen_thread_ids: set[str] = set()
        matched = 0
        has_more = False
        # Checkpointer.list(None) 按 checkpoint_id 从新到旧返回。
        # 每个 thread 第一次出现的检查点就是该会话最新状态。
        for checkpoint_tuple in list_checkpoints(None):
            configurable = (checkpoint_tuple.config or {}).get("configurable", {})
            thread_id = configurable.get("thread_id")
            if not isinstance(thread_id, str) or not thread_id or thread_id in seen_thread_ids:
                continue
            seen_thread_ids.add(thread_id)

            checkpoint = checkpoint_tuple.checkpoint or {}
            channel_values = checkpoint.get("channel_values", {})
            messages = _serialize_checkpoint_message_list(channel_values.get("messages", []))
            user_messages = [message for message in messages if message["role"] == "user"]
            if not user_messages:
                continue

            if matched < offset:
                matched += 1
                continue
            if len(threads) >= limit:
                has_more = True
                break

            latest_message = next(
                (message for message in reversed(messages) if message["role"] in ("user", "assistant")),
                user_messages[-1],
            )
            threads.append({
                "thread_id": thread_id,
                "title": user_messages[0]["content"],
                "preview": latest_message["content"],
                "message_count": len(messages),
            })

        return {
            "threads": threads,
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("列出会话检查点失败")
        raise HTTPException(status_code=503, detail="暂时无法读取会话列表") from exc


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
    try:
        lease = THREAD_RUNS.claim(thread_id)
    except RunBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    with lease:
        return _delete_thread_data(thread_id)


def _delete_thread_data(thread_id: str) -> dict:
    """持有会话占用后再检查并删除，避免检查通过后有新任务进入。"""
    store = app.state.runtime_store
    if store.has_running_run_for_thread(thread_id):
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
# 5. 定时任务增删接口
# ============================================================

@app.post("/tasks")
def create_task(req: ScheduledTaskRequest, request: Request) -> dict:
    """
    动态添加一个定时任务。

    返回：
        created：任务名。
        cron：注册进去的 cron 表达式。

    任务会先持久化，再重建调度器使其立即生效。
    start_scheduler() 内部会先停掉旧调度器再重建，
    所以这里直接调用是安全的。

    """
    if getattr(request.app.state, "desktop_access", None) is not None:
        raise HTTPException(status_code=403, detail="桌面托管服务未启用定时任务管理")
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
def delete_task(name: str, request: Request) -> dict:
    """
    删除一个定时任务。

    返回：
        removed：被删的任务名。
        ok：是否真的删掉了（任务本来就不存在时为 False）。

    注意：
    同样需要重启调度器才生效，
    这里已经顺手调了 start_scheduler()。
    """
    if getattr(request.app.state, "desktop_access", None) is not None:
        raise HTTPException(status_code=403, detail="桌面托管服务未启用定时任务管理")
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


class WSCommand(BaseModel):
    """WebSocket 命令使用严格类型，拒绝未声明字段。"""

    model_config = ConfigDict(strict=True, extra="forbid")


class StartCommand(WSCommand):
    type: Literal["run.start"]
    instruction: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50_000)]
    task_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)] | None = None
    thread_id: Annotated[str, StringConstraints(min_length=1, max_length=256)] | None = None
    allow_write: bool = False


class CancelCommand(WSCommand):
    type: Literal["cancel"]


class ApprovalCommand(WSCommand):
    type: Literal["approval.respond"]
    run_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    call_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    approved: bool


WS_COMMAND_TYPES = {
    "run.start": StartCommand,
    "cancel": CancelCommand,
    "approval.respond": ApprovalCommand,
}


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

    桌面托管模式验证握手凭证，并要求有副作用的工具逐次请求批准。
    独立服务保留布尔值 `allow_write: true` 的旧客户端兼容。

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
    access = getattr(websocket.app.state, "desktop_access", None)
    if access is not None:
        origin = websocket.headers.get("origin")
        protocols = websocket.scope.get("subprotocols", [])
        credential = next(
            (protocol.removeprefix("bearer.") for protocol in protocols if protocol.startswith("bearer.")),
            "",
        )
        if (
            (origin is not None and origin not in DESKTOP_ORIGINS)
            or DESKTOP_WS_PROTOCOL not in protocols
            or not access.accepts(credential)
        ):
            await websocket.close(code=1008)
            return
    # 只回传业务协议名，避免把凭证写回响应头。
    await websocket.accept(subprotocol=DESKTOP_WS_PROTOCOL if access is not None else None)

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
            frame = await websocket.receive()
            if frame["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(frame["code"])
            raw = frame.get("text")
            if raw is None or len(raw) > 262_144:
                await queue.put(RunError(run_id="", message="命令必须是长度受限的 JSON 文本", recoverable=True))
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await queue.put(RunError(run_id="", message="消息不是合法 JSON", recoverable=True))
                continue
            if not isinstance(payload, dict):
                await queue.put(RunError(run_id="", message="消息必须是 JSON 对象", recoverable=True))
                continue
            await _handle_ws_command(
                manager=manager,
                queue=queue,
                conn_id=conn_id,
                payload=payload,
                managed=access is not None,
            )
    except WebSocketDisconnect:
        logger.info("WebSocket 断开：%s", conn_id)
    finally:
        # 断开：取消这个连接上的所有 run。
        # 不取消的话，消费线程会一直跑到 agent 结束。
        manager.disconnect(conn_id)
        pump_task.cancel()


async def _handle_ws_command(
    *,
    manager: RunManager,
    queue: asyncio.Queue,
    conn_id: str,
    payload: dict,
    managed: bool,
) -> None:
    """处理一条客户端命令。

    注意：
    产生的消息都进 queue，由唯一的推送任务发出去 ——
    端点自己不直接 send。
    """
    kind = payload.get("type")
    # 同连接重复启动只拒绝新命令，不能让界面把仍运行中的旧任务当作失败。
    if kind == "run.start" and manager.active_for(conn_id):
        await queue.put(RunError(run_id="", message="上一轮还在运行", recoverable=True))
        return
    model = WS_COMMAND_TYPES.get(kind) if isinstance(kind, str) else None
    if model is None:
        await queue.put(RunError(run_id="", message="未知命令", recoverable=True))
        return
    try:
        command = model.model_validate(payload)
    except ValidationError as exc:
        # 不回显客户端输入，错误中只列出已知字段名。
        fields = sorted({
            error["loc"][0] for error in exc.errors(include_input=False)
            if error["loc"] and error["loc"][0] in model.model_fields
        })
        invalid = "、".join(fields) or "存在未知字段"
        await queue.put(RunError(run_id="", message=f"{kind} 命令无效：{invalid}", recoverable=kind != "run.start"))
        return

    if isinstance(command, StartCommand):
        if managed and command.allow_write:
            await queue.put(RunError(run_id="", message="桌面模式不支持整轮授权，请逐次审批"))
            return
        await _handle_run_start(manager=manager, queue=queue, conn_id=conn_id, command=command)
    elif isinstance(command, CancelCommand):
        await _handle_ws_cancel(manager=manager, queue=queue, conn_id=conn_id)
    elif isinstance(command, ApprovalCommand):
        # 只有发起 run 的连接可以应答。成功通过后由工具事件报告执行结果。
        if not manager.respond_approval(command.run_id, conn_id, command.call_id, command.approved):
            await queue.put(RunError(run_id=command.run_id, message="授权请求不存在、已处理或不属于当前连接", recoverable=True))


async def _handle_run_start(
    *,
    manager: RunManager,
    queue: asyncio.Queue,
    conn_id: str,
    command: StartCommand,
) -> None:
    """启动一次执行。"""
    instruction = command.instruction
    task_name = command.task_name or instruction[:30]

    # Agent 在首次调用时才惰性初始化。配置缺失（例如桌面版尚未填 API Key）
    # 必须转换成协议错误；否则异常会逃出 WebSocket handler，连接关闭后 UI 会一直显示运行中。
    try:
        agent = await asyncio.to_thread(get_agent)
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
        allow_write=command.allow_write,
        notify_channel=RUNTIME.notify_channel,
    )

    try:
        handle = manager.start(
            owner=conn_id,
            instruction=instruction,
            task_name=task_name,
            thread_id=command.thread_id,
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
    except RunBusyError as exc:
        # 上一轮还在运行 —— 拒绝，不静默取消旧的。
        await queue.put(RunError(run_id="", message=str(exc)))
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
    # 先确认取消请求，再置标志；消费者的终态不会排到确认消息前面。
    await queue.put(RunProgress(
        run_id=active[0].run_id,
        note="已请求取消，正在等待当前调用退出；已完成的操作不会回滚。",
    ))
    manager.cancel(active[0].run_id)
