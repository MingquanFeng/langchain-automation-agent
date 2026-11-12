//! Tauri 编译期入口。
//!
//! 该脚本让 `tauri-build` 读取 `tauri.conf.json` 并生成应用上下文；窗口配置与
//! 前端 URL 都由配置文件维护，因此此处不放运行时业务逻辑。

fn main() {
    // tauri_build 会在编译时校验并处理 Tauri 应用配置。
    tauri_build::build()
}
