
"""
统一入口：一条命令同时起 HTTP 服务和调度器。

对应课件：
课件未覆盖部署与命令行入口，需要自行实现。

主要职责：
1. 解析命令行参数（子命令 + 监听地址 + 端口 + 调度开关）。
2. 决定启动 HTTP 服务还是交互式终端。
3. 在导入服务模块之前把调度开关生效到配置上。

两个子命令的区别：
    serve（默认）：
        启动 FastAPI 服务，同时按需带上定时调度器。
        无界面，通过 HTTP 接口触发 Agent。
        对应 runtime/server.py。

    chat：
        启动交互式终端，直接在命令行里像聊天一样给 Agent 下指令。
        对应 runtime/repl.py。
        不起 HTTP 服务，也不启动调度器。

用法：
    python -m runtime              # 起服务（含定时任务）
    python -m runtime --no-schedule # 只起服务
    python -m runtime chat         # 交互式终端
"""

from __future__ import annotations

# argparse：标准库的命令行参数解析模块。
import argparse

# logging：标准库的日志模块。
import logging

# os：标准库，用来改写环境变量。
import os

# uvicorn：ASGI 服务器，用来承载 FastAPI 应用。
import uvicorn

# RUNTIME：应用运行配置，提供 host/port 默认值。
from agent.config import RUNTIME


def main() -> None:
    """
    命令行入口。

    职责：
    1. 解析参数。
    2. 配置日志。
    3. 让调度开关在服务模块被导入前生效。
    4. 分派到 chat 或 serve 两条路径。

    注意：
    这里的执行顺序有强依赖，不要随意调整。
    具体见下面「调度开关的生效时机」一节的注释。
    """
    parser = argparse.ArgumentParser(description="自动化 Agent 运行时")

    # --------------------------------------------------------
    # 1.1 位置参数：子命令
    # --------------------------------------------------------

    # nargs="?" 表示这个参数可以省略。
    # default="serve" 表示省略时走 serve。
    #
    # choices 限定取值范围，
    # 传了别的值 argparse 会直接报错退出。
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["serve", "chat"],
        help="serve=起 HTTP 服务（默认），chat=交互式终端",
    )

    # --------------------------------------------------------
    # 1.2 可选参数：监听地址、端口、开关
    # --------------------------------------------------------

    # host/port 的默认值来自 RUNTIME，
    # 即 agent/config.py 读的环境变量
    # API_HOST / API_PORT。
    parser.add_argument("--host", default=RUNTIME.api_host)
    parser.add_argument("--port", type=int, default=RUNTIME.api_port)

    # action="store_true"：只要出现这个参数就是 True，
    # 不出现就是 False，不需要再写默认值。
    parser.add_argument("--no-schedule", action="store_true", help="不启动定时任务")
    parser.add_argument("--reload", action="store_true", help="代码热重载（开发用）")
    args = parser.parse_args()

    # --------------------------------------------------------
    # 1.3 配置日志
    # --------------------------------------------------------

    # basicConfig 只需要调用一次，多余的调用不会有额外效果。
    # 格式里带上 logger 名，方便区分是调度器还是 HTTP 服务打的日志。
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # --------------------------------------------------------
    # 2. 调度开关的生效时机
    # --------------------------------------------------------

    # 必须在导入 runtime.server 之前设置：server.py 在模块导入时就读了 RUNTIME
    #
    # 这里的因果链不能调换顺序：
    #
    # 1. agent/config.py 在「模块导入时」读一次环境变量，
    #    读完就固化成 MODEL / RUNTIME 两个对象。
    #    之后再改 os.environ 不会自动反映到 RUNTIME 上。
    #
    # 2. 所以 --no-schedule 要先写环境变量，
    #    再 reload(agent.config) 让它重新读一遍。
    #
    # 3. runtime/server.py 用 `from agent.config import RUNTIME` 导入，
    #    这同样是导入时绑定。
    #    因此 server 必须在 reload 之后才导入，
    #    它拿到的才是开关已生效的那份 RUNTIME。
    #
    # 顺序反了（先 import server 再设环境变量），
    # --no-schedule 就会失效，调度器照常启动。
    #
    # 注意（已知坑，未修）：
    # agent/config.py 用 load_dotenv(override=True) 加载 .env，
    # override=True 的含义是「.env 覆盖已有环境变量」。
    # 所以下面这行设进去的 "0"，
    # 会被 reload(agent.config) 时重新读 .env 的动作顶掉
    # （.env.example 里默认就带着 SCHEDULER_ENABLED=1）。
    #
    # 实测：
    # .env 里存在 SCHEDULER_ENABLED=1 时，
    # python -m runtime --no-schedule 依然会启动调度器，
    # /health 返回的 scheduler 也是 true。
    #
    # 规避办法（不改代码）：
    # 用 SCHEDULER_ENABLED=0 python -m runtime 之外的环境变量也没用
    # （同样会被 .env 覆盖），
    # 只能临时把 .env 里那行改掉，或者先 mv .env .env.bak。
    # 真正的修法是让 config.py 改用 load_dotenv(override=False)，
    # 或者把 --no-schedule 的开关写成不经过环境变量的独立路径。
    if args.no_schedule:
        os.environ["SCHEDULER_ENABLED"] = "0"

    # importlib：标准库模块，用来重新加载模块。
    #
    # reload() 会重新执行模块代码，
    # 对 config.py 来说就是重新读一遍环境变量。
    import importlib

    # import agent.config
    #
    # 必须先 import 再 reload：
    # reload 要求模块对象已经存在于 sys.modules 里。
    import agent.config

    importlib.reload(agent.config)

    # --------------------------------------------------------
    # 3. 分派子命令
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 3.1 chat：交互式终端
    # --------------------------------------------------------

    # 延迟导入：只有真的走 chat 分支时才加载 repl 及其依赖。
    #
    # 注意：
    # chat 分支在 return 之前就结束了，
    # 所以既不会起 HTTP 服务，也不会起调度器。
    #
    # 它和 serve 唯一的交集是共用了
    # runtime/runner.py 里的 Agent 单例逻辑。
    if args.command == "chat":
        from runtime.repl import run_repl

        run_repl()
        return

    # --------------------------------------------------------
    # 3.2 serve：HTTP 服务
    # --------------------------------------------------------

    # 同样延迟导入。
    #
    # 放在 reload(agent.config) 之后，
    # 保证 server 绑定到的是开关生效后的 RUNTIME。
    from runtime import server

    importlib.reload(server)

    # uvicorn.run() 直接接收 app 对象并阻塞运行，
    # 直到进程被 Ctrl-C 打断。
    #
    # 注意：
    # --reload 目前不可用。
    # uvicorn 要求开启 reload 时应用必须以「导入字符串」形式传入
    # （如 "runtime.server:app"），
    # 传对象它会直接 sys.exit(1) 退出。
    # 这是开发期想用热重载时需要注意的坑。
    uvicorn.run(
        server.app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


# 作为脚本直接运行时（python -m runtime）才调用 main()，
# 被别的模块导入时不触发。
if __name__ == "__main__":
    main()
