
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
    # 2. 所以 --no-schedule 要先写覆盖，
    #    再 reload(agent.config) 让它重新读一遍。
    #
    # 3. runtime/server.py 用 `from agent.config import RUNTIME` 导入，
    #    这同样是导入时绑定。
    #    因此 server 必须在 reload 之后才导入，
    #    它拿到的才是开关已生效的那份 RUNTIME。
    #
    # 顺序反了（先 import server 再设开关），
    # --no-schedule 就会失效，调度器照常启动。
    #
    # 注意：
    # 这里用的是 agent.config.set_cli_override() 而不是
    # os.environ["SCHEDULER_ENABLED"] = "0"。
    #
    # 因为 agent/config.py 用 load_dotenv(override=True) 加载 .env，
    # 含义是「.env 覆盖已有环境变量」——
    # 直接设进 os.environ 的那个 "0"，
    # 会在 reload 时重新读 .env 的动作里被顶掉
    # （.env.example 里默认就带着 SCHEDULER_ENABLED=1），
    # --no-schedule 形同虚设。
    #
    # set_cli_override() 写的是另一个独立前缀的键，
    # .env 里不会有它，reload 也不会清掉它，
    # 所以开关能真正生效。
    #
    # 下面的 import agent.config 必须在这一行之前 ——
    # set_cli_override 是 agent.config 的函数，
    # 模块还没导入就调不到。
    import agent.config

    if args.no_schedule:
        agent.config.set_cli_override("SCHEDULER_ENABLED", "0")

    # importlib：标准库模块，用来重新加载模块。
    #
    # reload() 会重新执行模块代码，
    # 对 config.py 来说就是重新读一遍环境变量。
    import importlib

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
    # 注意 --reload 要传导入字符串，不能传对象：
    # uvicorn 开启 reload（或 workers）时要求应用以
    # 「模块:属性」的形式传入，比如 "runtime.server:app"，
    # 传对象它会直接 sys.exit(1)。
    #
    # 传字符串之后，uvicorn 的监视进程会另起一个子进程，
    # 由子进程重新导入 runtime.server:app 并真正提供服务。
    #
    # 好处是子进程继承当前进程的 os.environ，
    # 所以 --no-schedule 写的命令行覆盖在子进程里同样生效。
    #
    # 前提是项目根目录在 sys.path 上，
    # 用 python -m runtime 启动时 cwd 就是项目根，没问题。
    target = "runtime.server:app" if args.reload else server.app

    uvicorn.run(
        target,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


# 作为脚本直接运行时（python -m runtime）才调用 main()，
# 被别的模块导入时不触发。
if __name__ == "__main__":
    main()
