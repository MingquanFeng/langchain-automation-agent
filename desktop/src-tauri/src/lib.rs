//! Automation Agent 桌面壳的 Tauri 应用入口。
//!
//! 第一阶段不在 Rust 侧执行 Agent 任务，也不调用本机文件系统或 Shell 能力。
//! WebView 直接打开单独运行的 FastAPI 工作台，并通过页面自身的同源 WebSocket
//! 协议与 Agent 服务通信。

/// 创建并运行 Tauri 应用。
pub fn run() {
    // 不注册命令或插件：桌面 MVP 暂时只需要显示本机 FastAPI 页面。
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("启动 Automation Agent 桌面应用失败");
}
