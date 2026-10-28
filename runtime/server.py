
"""
FastAPI 服务：手动触发 Agent + 查看调度状态 + 流式输出。

课件未涉及部署，本层按 langgraph dev 之外的路径用 FastAPI 实现。

对应课件：
课件未覆盖 HTTP 服务，需要自行实现。

主要职责：
1. 暴露一组 HTTP 端点，让外部程序能触发 Agent。
2. 在应用启动时拉起调度器，关闭时停掉。
3. 提供同步触发（/run）和流式触发（/run/stream）两种方式。

注意：
FastAPI 的 @app.on_event 已废弃，
本文件统一用 lifespan 上下文管理器写启动/关闭逻辑。
"""

from __future__ import annotations

# logging：标准库的日志模块。
import logging

# uuid：标准库，用来生成随机会话 ID。
import uuid

# asynccontextmanager：把生成器包装成异步上下文管理器，
# 用于定义 FastAPI 的 lifespan。
from contextlib import asynccontextmanager

# FastAPI：Web 框架。
from fastapi import FastAPI

# StreamingResponse：SSE 流式响应。
from fastapi.responses import StreamingResponse

# BaseModel / Field：Pydantic 模型与字段声明，
# FastAPI 用它做请求体校验和 OpenAPI 文档。
from pydantic import BaseModel, Field

# RUNTIME：应用运行配置，这里取通知渠道和调度开关。
from agent.config import RUNTIME

# get_agent / run_task：统一的 Agent 调用入口。
from .runner import get_agent, run_task

# list_jobs / start_scheduler / stop_scheduler：调度器接口。
from .scheduler import list_jobs, start_scheduler, stop_scheduler

# ScheduledTask / add_task / all_tasks / remove_task：任务表接口。
from .tasks import ScheduledTask, add_task, all_tasks, remove_task

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
    if RUNTIME.scheduler_enabled:
        start_scheduler()
        logger.info("定时任务已启动")
    yield
    stop_scheduler()


# app：FastAPI 应用实例。
#
# lifespan 通过构造参数传进去，
# 这是 FastAPI 0.135+ 推荐的写法，
# 不要再用 @app.on_event("startup") / @app.on_event("shutdown")。
app = FastAPI(title="Automation Agent", version="0.1.0", lifespan=lifespan)


# ============================================================
# 2. 请求体模型
# ============================================================

class TaskRequest(BaseModel):
    """
    /run 和 /run/stream 的请求体。

    Field(...) 表示必填；
    Field(默认值, ...) 表示可选并带默认值。

    注意：
    allow_write 默认 False，即接口默认只读。
    要让 Agent 改文件必须显式传 true。
    """

    # 给 Agent 的自然语言指令。
    instruction: str = Field(..., description="给 Agent 的自然语言指令")

    # 任务名，出现在通知标题里。
    # 不传时用指令前 30 个字符兜底。
    task_name: str | None = Field(None, description="任务名，出现在通知标题里")

    # 会话 ID，相同则共享记忆。
    #
    # 注意：
    # /run 不传就是一次性会话；
    # /run/stream 不传也是一次性会话，
    # 两者都要显式传才会复用历史。
    thread_id: str | None = Field(None, description="会话 ID，相同则共享记忆")

    # 是否允许写操作。
    allow_write: bool = Field(False, description="是否允许写操作")


class ScheduledTaskRequest(BaseModel):
    """
    POST /tasks 的请求体：新增一个定时任务。

    注意：
    这个模型和 TaskRequest 分开，
    因为一个描述「跑一次」，一个描述「什么时候跑」。
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
# 5. 触发 Agent
# ============================================================

@app.post("/run")
def run(req: TaskRequest) -> dict:
    """同步执行一次任务。

    返回：
        runner.run_task() 的结果字典，
        含 task_name、thread_id、output、messages。

    注意：
    这是同步接口，
    客户端要一直等到 Agent 跑完才拿到响应。
    长任务建议用 /run/stream。

    注意：
    trigger 固定为 "api"，
    让 Agent 知道自己是被 HTTP 触发的。
    """
    return run_task(
        instruction=req.instruction,
        task_name=req.task_name,
        trigger="api",
        allow_write=req.allow_write,
        thread_id=req.thread_id,
    )


@app.post("/run/stream")
def run_stream(req: TaskRequest) -> StreamingResponse:
    """流式执行，逐块返回模型输出。

    返回：
        text/event-stream 类型的响应。

    注意：
    这里没有复用 runner.run_task()，
    因为流式需要自己控制 agent.stream() 的迭代，
    run_task() 只提供了同步封装。
    但上下文和 thread_id 的构造规则保持一致。
    """
    # 延迟导入：
    # 这两个模块只在真正要流式输出时才用得上，
    # 放在函数内可以省掉模块级的导入开销。
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
        """
        SSE 生成器。

        stream_mode="values" 表示每步产出整个状态字典，
        里面的 messages 是到目前为止的完整消息列表。

        注意：
        这里只取最后一条消息的内容往外发，
        相当于「每次有新内容就把最新的完整答案重发一遍」，
        客户端需要自己处理覆盖式更新。

        注意：
        SSE 的数据帧格式是 `data: {内容}`，
        帧之间用两个换行分隔。
        """
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=req.instruction)]},
            config,
            context=ctx,
            stream_mode="values",
        ):
            messages = chunk.get("messages", [])
            if messages:
                last = messages[-1]

                # content 可能是字符串，也可能是多模态的内容块列表。
                # 这里只处理字符串形态，其他形态直接跳过。
                content = getattr(last, "content", "")
                if content:
                    yield f"data: {content}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
