# 阿序 · Axu 桌面应用

这是独立的 Tauri 桌面项目。桌面入口会启动随应用打包的 Python sidecar，并加载应用包内的工作台页面；聊天界面随桌面应用分发，Agent 执行、事件归一化和授权仍由 Python 服务负责。

## 环境要求

- macOS 及 Xcode Command Line Tools
- Node.js 与 npm
- Rust stable 工具链

## 开发与打包

桌面端把 Python 环境构建为 macOS sidecar。构建时需要一个已安装项目运行依赖的 Python 环境，以及 PyInstaller：

```bash
/opt/miniconda3/envs/langchain1.2/bin/python -m pip install -r desktop/requirements-build.txt
```

在 `desktop/` 目录运行开发版或构建 `.app`：

```bash
cd desktop
AUTOMATION_AGENT_PYTHON=/opt/miniconda3/envs/langchain1.2/bin/python npm run dev
AUTOMATION_AGENT_PYTHON=/opt/miniconda3/envs/langchain1.2/bin/python npm run build
```

`npm run dev` 和 `npm run build` 会先调用 `scripts/build_runtime.py`，生成当前 macOS 架构的 PyInstaller sidecar，再启动 Tauri。`npm run build` 成功后会自动把最新的 `阿序 · Axu.app` 移到桌面，并替换桌面上的旧版本。首次需要联网安装构建依赖。sidecar 会从当前 Python 环境收集运行时依赖；桌面应用使用时不要求用户安装 Python。当前只配置 Apple Silicon 与 Intel Mac 的构建目标，签名和公证尚未配置。

桌面应用启动时会选择一个空闲端口，只绑定 `127.0.0.1`。宿主通过 `/health` 核对本次启动标记，再把连接地址和仅本次启动有效的凭证交给工作台。工作台通过 `Authorization: Bearer` 访问 HTTP API，并在 WebSocket 子协议握手中传递凭证。重启服务会更换凭证，工作台只向宿主提供的地址发送凭证。

托管服务要求逐次审批写入和外部通知，并关闭定时任务管理。多个工具同时请求审批时，输入框上方每次只显示一项，可以展开参数或文件差异。每项从展示时起有 60 秒可供决定，取消任务会撤销待批请求。独立运行的 `python -m runtime` 保留原有接口用于开发。应用关闭窗口或退出时会向托管服务发送 `SIGTERM`，超时后再回收整个 sidecar 进程组。

点击取消后会显示“正在取消”，直到当前模型或工具调用退出；已发生的写入不会回滚。模型网络 I/O 默认超时为 60 秒，可在应用支持目录的 `.env` 中设置 `LLM_REQUEST_TIMEOUT` 并重启服务。取消期间同一会话保持占用，另一连接无法启动新任务或删除该会话；连接断开会清理事件队列，最终状态仍写入运行历史。

API Key 和模型配置保存在 macOS 应用支持目录的 `.env`；会话记忆数据库保存在同目录的 `data/memory.sqlite`，定时任务配置与 Agent 运行记录保存在 `data/runtime.sqlite3`，启动日志在 `logs/runtime.log`。运行记录包含任务指令、最终结果或错误信息，时间按 UTC 保存。文件工具默认使用这个 `data/` 目录，也可点击工作台输入框左侧的文件夹按钮选择其他工作目录。更换目录会保存到应用支持目录并重启 Agent 服务，文件工具仍受目录沙箱限制；会话记忆和运行数据不会移入所选目录。首次启动默认使用 MiniMax-M3（`LLM_PROVIDER`、`LLM_MODEL`、`LLM_BASE_URL`），模型设置只录入 `LLM_API_KEY`。

左栏“添加目录”会记住选过的文件夹，点击已有目录可切回。最近会话和运行历史按完整目录路径归档。窗口较窄时，可从输入框左侧的文件夹按钮切换。升级前的历史保存在“查看历史会话”的“未归档（旧版会话）”中。打开旧会话后，发送下一条消息即可将它归入当前目录。

恢复会话只读取已保存的 Checkpointer 消息。删除单条记录保留会话记忆，删除整个会话会清理关联记录和 Checkpointer 数据。清空运行记录仅作用于历史窗口中选定的范围，保存的会话仍可继续。

窗口载入工作台后，聊天页顶部也提供“连接服务 / 重试连接”按钮。它只重新打开 WebSocket，不会自动重发之前的任务；连接中断时任务最终状态可能未确认，需先查看服务端或日志再决定是否手动重发。

## 项目边界

- `web/index.html` 是随应用打包的本机连接页，负责模型 API Key 录入、工作目录选择、服务地址设置、loopback 校验、工作台握手和重试；它加载同目录的 `workbench.html`，再通过受限 `postMessage` 传入 sidecar 地址。
- `web/workbench.html` 是随 Tauri 静态资源打包的聊天与运行历史界面；直接连接 loopback API 和 `/ws`，不从 Python sidecar 下载 HTML。
- `src-tauri/tauri.conf.json` 配置应用名称、桌面页面资源、loopback HTTP/WebSocket CSP、图标和 Python sidecar。
- `src-tauri/src/lib.rs` 在 Rust 侧启动、监控并停止 sidecar，并提供 API Key 保存和工作目录更新命令；WebView 没有通用 Shell 或文件系统权限。
- `src-tauri/capabilities/default.json` 仅匹配主窗口，并只允许系统目录选择对话框。
- `src-tauri/icons/automation-agent-logo-source.png` 是应用图标源；替换后运行 `npm run icon -- src-tauri/icons/automation-agent-logo-source.png` 更新平台图标。
- `runtime_sidecar.py` 与 `scripts/build_runtime.py` 将后端打成带目标架构后缀的 PyInstaller 单文件 sidecar，供 Tauri `externalBin` 打包。
