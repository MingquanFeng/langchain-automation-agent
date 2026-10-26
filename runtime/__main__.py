"""统一入口：一条命令同时起 HTTP 服务和调度器。

用法：
    python -m runtime              # 起服务（含定时任务）
    python -m runtime --no-schedule # 只起服务
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from agent.config import RUNTIME


def main() -> None:
    parser = argparse.ArgumentParser(description="自动化 Agent 运行时")
    parser.add_argument("--host", default=RUNTIME.api_host)
    parser.add_argument("--port", type=int, default=RUNTIME.api_port)
    parser.add_argument("--no-schedule", action="store_true", help="不启动定时任务")
    parser.add_argument("--reload", action="store_true", help="代码热重载（开发用）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # 必须在导入 runtime.server 之前设置：server.py 在模块导入时就读了 RUNTIME
    if args.no_schedule:
        import os

        os.environ["SCHEDULER_ENABLED"] = "0"

    import importlib

    import agent.config

    importlib.reload(agent.config)
    from runtime import server

    importlib.reload(server)

    uvicorn.run(
        server.app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
