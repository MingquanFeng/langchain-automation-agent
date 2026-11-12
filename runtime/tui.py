"""Textual 终端界面的启动入口与服务连通性检查。

本模块由 `python -m runtime tui` 间接调用，当前负责 TUI 的启动准备工作：

1. 延迟导入 `textual` 与 `websockets`，让只启动 HTTP 服务的 `serve` 命令
   不必在模块导入时依赖终端界面专用包。
2. 缺少可选依赖时打印安装指引，而不是把底层 `ModuleNotFoundError` 原样抛给用户。
3. 打开一个轻量 Textual 页面，在后台尝试连接本地 `/ws`，显示服务是否可达。

这里的 WebSocket 连接只用于启动检查，检查完成后立即关闭。聊天消息、运行事件、
工具卡片和逐次授权交互会在后续 TUI 任务中接入；届时传输层也会迁移到独立模块。
"""

from __future__ import annotations


# ============================================================
# 1. 默认连接地址
# ============================================================

# 默认地址必须与 `runtime.__main__` 的 `--ws-url` 参数使用同一来源。
# 本地服务如运行在其他端口，命令行调用方可以传入自定义地址覆盖它。
DEFAULT_WS_URL = "ws://127.0.0.1:8000/ws"


# ============================================================
# 2. TUI 启动入口
# ============================================================

def main(ws_url: str = DEFAULT_WS_URL) -> None:
    """检查界面依赖并打开启动页，显示 WebSocket 服务的连通结果。

    参数：
        ws_url：TUI 要连接的 WebSocket 地址。默认指向本机 8000 端口的 `/ws`；
            也可由命令行 `--ws-url` 指定其他地址。

    退出方式：
        页面底部显示 `q` 快捷键；按下后由 Textual 正常关闭应用。

    注意：
        服务检查只确认 WebSocket 握手可以完成，不会启动 Agent run，也不会调用模型。
    """
    # 2.1 延迟加载 TUI 专用依赖。
    # 将导入放在 main() 内，这样只运行 `serve` 时不会因为缺少 Textual 而启动失败；
    # 同时可以捕获缺失模块名，并给出一条可直接执行的安装命令。
    try:
        import websockets
        from textual.app import App, ComposeResult
        from textual.containers import Center, Vertical
        from textual.widgets import Footer, Header, Static
    except ModuleNotFoundError as exc:
        # exc.name 会指出本次导入失败的包；兜底文本兼容缺少模块名的异常情况。
        missing = exc.name or "textual / websockets"
        raise SystemExit(
            f"启动 TUI 缺少依赖：{missing}。请在项目环境中安装："
            "pip install textual websockets"
        ) from exc

    # 2.2 在函数内定义 App，避免把可选的 Textual 类型放到模块级导入中。
    class TuiEntryApp(App):
        """显示服务连接状态，作为后续聊天与事件组件的应用外壳。"""

        # 终端标题栏中的应用名与副标题。
        TITLE = "Automation Agent"
        SUB_TITLE = "Textual TUI"

        # 首版启动页采用居中状态卡片；后续聊天视图会在独立任务中扩展布局。
        CSS = """
        Screen { align: center middle; }
        #card { width: 76; height: auto; padding: 2 3; border: round $primary; }
        #heading { text-style: bold; color: $text; margin-bottom: 1; }
        #status { margin-bottom: 1; }
        #endpoint { color: $text-muted; }
        """

        # `q` 是退出快捷键；Textual 会把说明显示在底部 Footer 中。
        BINDINGS = [("q", "quit", "退出")]

        def compose(self) -> ComposeResult:
            """按从上到下的顺序创建标题栏、连接状态卡片和快捷键栏。

            `#status` 会在后台握手完成后更新；`#endpoint` 保留用户输入的地址，
            便于连接失败时确认当前尝试连接的是哪个服务。
            """
            yield Header()
            with Center():
                with Vertical(id="card"):
                    yield Static("Agent 终端工作台", id="heading")
                    yield Static("正在检查本地 Agent 服务…", id="status")
                    yield Static(f"连接地址：{ws_url}", id="endpoint")
                    yield Static(
                        "聊天记录、工具卡片和逐次授权将在后续任务中接入。"
                    )
            yield Footer()

        def on_mount(self) -> None:
            """页面挂载后启动后台连通性检查。

            不在事件处理器里同步等待网络握手，避免服务响应慢时终端界面冻结。
            `exclusive=True` 确保同一检查任务不会被重复启动。
            """
            self.run_worker(self._check_connection(), name="websocket-check", exclusive=True)

        async def _check_connection(self) -> None:
            """尝试完成一次 WebSocket 握手，并把结果写回状态组件。

            成功时马上离开 async with，关闭这条临时连接；失败时把异常类型和信息
            显示在页面中，并提示用户先启动 FastAPI 服务。此检查不发送任何命令。
            """
            status = self.query_one("#status", Static)
            try:
                # 设置短握手超时，让地址无效或服务未启动时尽快显示指导信息。
                async with websockets.connect(ws_url, open_timeout=3):
                    status.update("服务已连接。终端界面基础入口已就绪。")
            except Exception as exc:
                # 网络错误、服务未监听和握手失败都显示在同一处，方便用户据此排查。
                status.update(
                    "无法连接 Agent 服务。请先在另一个终端运行：\n"
                    "python -m runtime --no-schedule\n"
                    f"连接错误：{type(exc).__name__}: {exc}"
                )

    # 启动 Textual 的主事件循环；页面关闭后 main() 随之返回。
    TuiEntryApp().run()
