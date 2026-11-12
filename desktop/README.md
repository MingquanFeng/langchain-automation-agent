# Automation Agent 桌面壳

这是独立的 Tauri 桌面项目。第一阶段复用根目录 FastAPI 服务提供的浏览器工作台，不复制聊天、事件归一化或授权逻辑。

## 环境要求

- macOS 及 Xcode Command Line Tools
- Node.js 与 npm
- Rust stable 工具链

## 开发启动

先从仓库根目录启动 Agent 服务：

```bash
/opt/miniconda3/envs/langchain1.2/bin/python -m runtime --no-schedule
```

再在另一个终端进入本目录安装依赖并启动 Tauri：

```bash
cd desktop
npm install
npm run dev
```

桌面窗口默认加载 `http://127.0.0.1:8000/`。服务没有启动时，Tauri 开发服务器无法连接到工作台。

## 项目边界

- `src-tauri/tauri.conf.json` 配置应用名称、主窗口、loopback 页面地址和 macOS `.app` 打包图标。
- `src-tauri/src/lib.rs` 仅创建 Tauri 窗口，不注册系统命令或插件。
- `src-tauri/capabilities/default.json` 不授予文件系统、Shell 或其他本机权限。
- `src-tauri/icons/app-icon.svg` 是临时图标源；需要时运行 `npm run icon -- src-tauri/icons/app-icon.svg` 更新平台图标。
- Python 服务目前需要单独启动；自动托管、API Key 配置界面和可分发安装包属于后续阶段。
