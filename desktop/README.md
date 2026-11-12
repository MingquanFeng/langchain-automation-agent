# Automation Agent 桌面壳

这是独立的 Tauri 桌面项目。桌面入口先打开随应用打包的本机连接页，连接后嵌入根目录 FastAPI 服务提供的浏览器工作台；聊天、事件归一化和授权逻辑仍由服务端页面负责。

## 环境要求

- macOS 及 Xcode Command Line Tools
- Node.js 与 npm
- Rust stable 工具链

## 开发启动

在仓库根目录启动 Agent 服务：

```bash
/opt/miniconda3/envs/langchain1.2/bin/python -m runtime --no-schedule
```

在另一个终端启动桌面应用：

```bash
cd desktop
npm install
npm run dev
```

桌面窗口会载入 `http://127.0.0.1:8000/`，并等待工作台页面握手。若服务没有启动或页面未载入，连接页会保留在窗口中，可先启动服务再点重试。地址设置保存在当前桌面应用的 WebView 存储中；首版只接受 `http://127.0.0.1` 或 `http://localhost`，端口可以修改。

窗口载入工作台后，聊天页顶部也提供“连接服务 / 重试连接”按钮。它只重新打开 WebSocket，不会自动重发之前的任务；连接中断时任务最终状态可能未确认，需先查看服务端或日志再决定是否手动重发。

## 项目边界

- `web/index.html` 是随应用打包的本机连接页，负责服务地址设置、loopback 校验、工作台握手和重试；它嵌入由 FastAPI 提供的 `frontend/workbench.html`，iframe 仅允许脚本与原服务来源能力，不允许顶层导航。
- `src-tauri/tauri.conf.json` 配置应用名称、连接页资源、仅允许本机来源的 CSP 和 macOS `.app` 打包图标。
- `src-tauri/src/lib.rs` 仅创建 Tauri 窗口，不注册系统命令或插件。
- `src-tauri/capabilities/default.json` 不授予文件系统、Shell 或其他本机权限。
- `src-tauri/icons/app-icon.svg` 是临时图标源；需要时运行 `npm run icon -- src-tauri/icons/app-icon.svg` 更新平台图标。
- Python 服务目前需要单独启动；自动托管、API Key 配置界面和可分发安装包属于后续阶段。
