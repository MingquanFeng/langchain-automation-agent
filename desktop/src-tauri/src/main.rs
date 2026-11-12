//! macOS 桌面应用的二进制入口。
//!
//! 实际 Tauri builder 放在 `lib.rs`，这样同一套应用初始化代码也可供平台构建流程
//! 使用；本文件只负责转交启动控制权。

// Windows 发布版本使用 GUI 子系统，避免额外弹出控制台窗口。
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // 启动桌面应用库中的 Tauri builder。
    automation_agent_desktop_lib::run();
}
