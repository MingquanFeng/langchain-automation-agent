# Automation Agent 桌面壳

这是独立的 Tauri 桌面项目。桌面入口会启动随应用打包的 Python sidecar，并嵌入它提供的浏览器工作台；聊天、事件归一化和授权逻辑仍由服务端页面负责。

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

`npm run dev` 和 `npm run build` 会先调用 `scripts/build_runtime.py`，生成当前 macOS 架构的 PyInstaller sidecar，再启动 Tauri。`npm run build` 成功后会自动把最新的 `Automation Agent.app` 移到桌面，并替换桌面上的旧版本。首次需要联网安装构建依赖。sidecar 会从当前 Python 环境收集运行时依赖；桌面应用使用时不要求用户安装 Python。当前只配置 Apple Silicon 与 Intel Mac 的构建目标，签名和公证尚未配置。

桌面应用启动时会选择一个空闲端口，只绑定 `127.0.0.1`，并在服务通过 `/health` 检查后载入工作台。端口由操作系统动态分配，因此不会占用浏览器版默认的 8000 端口。应用关闭窗口或退出时会向服务发送 `SIGTERM`，超时后再回收整个 sidecar 进程组。

API Key 和模型配置保存在 macOS 应用支持目录的 `.env`；会话记忆数据库保存在同目录的 `data/`，启动日志在 `logs/runtime.log`。文件工具默认使用这个 `data/` 目录，也可点击工作台输入框左侧的文件夹按钮选择其他工作目录。更换目录会保存到应用支持目录并重启 Agent 服务，文件工具仍受目录沙箱限制；会话记忆不会移入所选目录。首次启动默认使用 MiniMax-M3（`LLM_PROVIDER`、`LLM_MODEL`、`LLM_BASE_URL`），模型设置只录入 `LLM_API_KEY`。

窗口载入工作台后，聊天页顶部也提供“连接服务 / 重试连接”按钮。它只重新打开 WebSocket，不会自动重发之前的任务；连接中断时任务最终状态可能未确认，需先查看服务端或日志再决定是否手动重发。

## 项目边界

- `web/index.html` 是随应用打包的本机连接页，负责模型 API Key 录入、工作目录选择、服务地址设置、loopback 校验、工作台握手和重试；它嵌入由 FastAPI 提供的 `frontend/workbench.html`，iframe 仅允许脚本与原服务来源能力，不允许顶层导航。
- `src-tauri/tauri.conf.json` 配置应用名称、连接页资源、loopback CSP、图标和 Python sidecar。
- `src-tauri/src/lib.rs` 在 Rust 侧启动、监控并停止 sidecar，并提供 API Key 保存和工作目录更新命令；WebView 没有通用 Shell 或文件系统权限。
- `src-tauri/capabilities/default.json` 仅匹配主窗口，并只允许系统目录选择对话框。
- `src-tauri/icons/automation-agent-logo-source.png` 是应用图标源；替换后运行 `npm run icon -- src-tauri/icons/automation-agent-logo-source.png` 更新平台图标。
- `runtime_sidecar.py` 与 `scripts/build_runtime.py` 将后端打成带目标架构后缀的 PyInstaller 单文件 sidecar，供 Tauri `externalBin` 打包。
