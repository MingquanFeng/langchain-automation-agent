"""FastAPI 服务：手动触发 Agent + 查看调度状态 + 流式输出。

课件未涉及部署，本层按 langgraph dev 之外的路径用 FastAPI 实现。
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.config import RUNTIME
from .runner import get_agent, run_task
from .scheduler import list_jobs, start_scheduler, stop_scheduler
from .tasks import ScheduledTask, add_task, all_tasks, remove_task

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子。

    FastAPI 0.135 起 @app.on_event 已废弃，改用 lifespan 写法。
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if RUNTIME.scheduler_enabled:
        start_scheduler()
        logger.info("定时任务已启动")
    yield
    stop_scheduler()


app = FastAPI(title="Automation Agent", version="0.1.0", lifespan=lifespan)


class TaskRequest(BaseModel):
    instruction: str = Field(..., description="给 Agent 的自然语言指令")
    task_name: str | None = Field(None, description="任务名，出现在通知标题里")
    thread_id: str | None = Field(None, description="会话 ID，相同则共享记忆")
    allow_write: bool = Field(False, description="是否允许写操作")


class ScheduledTaskRequest(BaseModel):
    name: str
    cron: str = Field(..., description="cron 表达式，如 '0 9 * * *'")
    instruction: str
    allow_write: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "scheduler": RUNTIME.scheduler_enabled}


@app.get("/tasks")
def list_scheduled_tasks() -> list[dict]:
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
    return list_jobs()


@app.post("/tasks")
def create_task(req: ScheduledTaskRequest) -> dict:
    task = ScheduledTask(
        name=req.name,
        cron=req.cron,
        instruction=req.instruction,
        allow_write=req.allow_write,
    )
    add_task(task)
    start_scheduler()
    return {"created": task.name, "cron": task.cron}


@app.delete("/tasks/{name}")
def delete_task(name: str) -> dict:
    removed = remove_task(name)
    start_scheduler()
    return {"removed": name, "ok": removed}


@app.post("/run")
def run(req: TaskRequest) -> dict:
    """同步执行一次任务。"""
    return run_task(
        instruction=req.instruction,
        task_name=req.task_name,
        trigger="api",
        allow_write=req.allow_write,
        thread_id=req.thread_id,
    )


@app.post("/run/stream")
def run_stream(req: TaskRequest) -> StreamingResponse:
    """流式执行，逐块返回模型输出。"""
    from langchain_core.messages import HumanMessage

    from agent.context import AgentContext

    agent = get_agent()
    ctx = AgentContext(
        task_name=req.task_name or req.instruction[:30],
        trigger="api",
        allow_write=req.allow_write,
        notify_channel=RUNTIME.notify_channel,
    )
    # 显式传 thread_id 才复用会话；默认每次新会话，避免流式请求互相污染记忆
    config = {"configurable": {"thread_id": req.thread_id or f"stream-{uuid.uuid4().hex[:8]}"}}

    def generate():
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=req.instruction)]},
            config,
            context=ctx,
            stream_mode="values",
        ):
            messages = chunk.get("messages", [])
            if messages:
                last = messages[-1]
                content = getattr(last, "content", "")
                if content:
                    yield f"data: {content}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
