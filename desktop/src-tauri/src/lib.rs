//! 阿序 · Axu 桌面应用入口与 Python sidecar 生命周期管理。

use serde::Serialize;
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};
use tauri::{Manager, State};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};

// Tauri places externalBin sidecars next to the main executable inside the app bundle.
const SIDECAR_NAME: &str = "automation-agent-runtime";
// PyInstaller one-file sidecars can take longer to unpack on a cold launch.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(90);

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeInfo {
    service_url: String,
    access_token: String,
    #[serde(skip)]
    readiness: String,
    config_path: String,
    log_path: String,
    working_directory: PathBuf,
    working_directories: Vec<PathBuf>,
    status: String,
    error: Option<String>,
}

struct RuntimeState {
    info: Mutex<RuntimeInfo>,
    child: Mutex<Option<CommandChild>>,
    log_file: Mutex<File>,
    app_home: PathBuf,
    started_at: Mutex<Instant>,
    exited: AtomicBool,
    stopping: AtomicBool,
}

impl RuntimeState {
    fn info(&self) -> RuntimeInfo {
        self.info.lock().expect("runtime info mutex poisoned").clone()
    }

    fn log(&self, line: &str) {
        if let Ok(mut file) = self.log_file.lock() {
            let _ = writeln!(file, "{line}");
            let _ = file.flush();
        }
    }

    fn log_output(&self, bytes: &[u8]) {
        if let Ok(mut file) = self.log_file.lock() {
            let _ = file.write_all(bytes);
            if !bytes.ends_with(b"\n") {
                let _ = file.write_all(b"\n");
            }
            let _ = file.flush();
        }
    }

    fn fail(&self, message: String) {
        let mut info = self.info.lock().expect("runtime info mutex poisoned");
        info.status = "failed".into();
        info.error = Some(message.clone());
        self.log(&format!("桌面运行时启动失败：{message}"));
    }
}

#[tauri::command]
fn runtime_info(state: State<'_, Arc<RuntimeState>>) -> RuntimeInfo {
    let current = state.info();
    if current.status != "starting" {
        return current;
    }

    if !state.exited.load(Ordering::Acquire)
        && health_check(&current.service_url, &current.readiness)
    {
        let mut info = state.info.lock().expect("runtime info mutex poisoned");
        if info.status == "starting" && info.readiness == current.readiness {
            info.status = "ready".into();
            info.error = None;
            state.log(&format!("Agent 服务已就绪：{}", info.service_url));
        }
        return info.clone();
    }

    let started_at = *state
        .started_at
        .lock()
        .expect("runtime start time mutex poisoned");
    if started_at.elapsed() > HEALTH_TIMEOUT && !state.exited.load(Ordering::Acquire) {
        state.fail(format!(
            "等待本机服务超过 {} 秒；请查看启动日志。",
            HEALTH_TIMEOUT.as_secs()
        ));
    }
    state.info()
}

#[tauri::command]
fn read_runtime_log(state: State<'_, Arc<RuntimeState>>) -> Result<String, String> {
    let path = PathBuf::from(&state.info().log_path);
    let mut file =
        File::open(&path).map_err(|error| format!("无法读取启动日志：{error}"))?;
    let length = file
        .metadata()
        .map_err(|error| format!("无法读取日志信息：{error}"))?
        .len();
    let offset = length.saturating_sub(24 * 1024);
    use std::io::{Seek, SeekFrom};
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("无法定位日志：{error}"))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|error| format!("无法读取启动日志：{error}"))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

#[tauri::command]
fn restart_runtime(
    app: tauri::AppHandle,
    state: State<'_, Arc<RuntimeState>>,
) -> Result<(), String> {
    stop_runtime(state.inner());
    start_runtime(&app, state.inner())
}

// 首次启动或升级时兼容旧的一行配置。返回活动目录以及侧栏应保留的完整清单。
fn read_working_directories(
    settings_path: &std::path::Path,
    default_working_directory: &std::path::Path,
) -> Result<(PathBuf, Vec<PathBuf>), String> {
    let mut working_directories = Vec::new();
    let saved = fs::read_to_string(settings_path).unwrap_or_default();
    for line in saved.lines().filter(|line| !line.is_empty()) {
        let path = PathBuf::from(line);
        // 暂未挂载的外置目录仍保留入口；启动时选第一个可用目录。
        let path = fs::canonicalize(&path).unwrap_or(path);
        if !working_directories.contains(&path) {
            working_directories.push(path);
        }
    }
    let working_directory = working_directories
        .iter()
        .find(|path| path.is_dir())
        .cloned()
        .unwrap_or(fs::canonicalize(default_working_directory).map_err(|error| {
            format!("无法确定默认工作目录：{error}")
        })?);
    if !working_directories.contains(&working_directory) {
        working_directories.push(working_directory.clone());
    }
    Ok((working_directory, working_directories))
}

// 把最近选中的目录置顶，保存成功后宿主才切换内存状态和文件工具的目录。
fn save_working_directory(
    settings_path: &std::path::Path,
    previous: &[PathBuf],
    path: &std::path::Path,
) -> Result<Vec<PathBuf>, String> {
    let mut directories = previous.to_vec();
    directories.retain(|saved| saved != path);
    directories.insert(0, path.to_path_buf());
    // 旧配置只有一行。继续每行保存一个规范路径，首行代表上次使用的目录。
    // 临时文件写完再替换，退出或磁盘写满时仍可读取原清单。
    let temporary = settings_path.with_extension(format!("{}.tmp", std::process::id()));
    let save = || -> std::io::Result<()> {
        let mut options = OpenOptions::new();
        options.create(true).truncate(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temporary)?;
        for directory in &directories {
            writeln!(file, "{}", directory.display())?;
        }
        file.sync_all()?;
        fs::rename(&temporary, settings_path)
    };
    if let Err(error) = save() {
        let _ = fs::remove_file(&temporary);
        return Err(format!("无法保存工作目录：{error}"));
    }
    Ok(directories)
}

#[cfg(test)]
mod workspace_tests {
    use super::*;

    #[test]
    fn old_selection_and_new_directories_survive_restart() {
        // 用两个真实存在的目录模拟系统选择框，只在临时目录写一份配置。
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let a = fs::canonicalize(root.join("src")).unwrap();
        let b = fs::canonicalize(root.join("icons")).unwrap();
        let settings = std::env::temp_dir().join(format!("axu-directory-test-{}.txt", std::process::id()));
        fs::write(&settings, a.to_str().unwrap()).unwrap();
        let (old_active, old_list) = read_working_directories(&settings, &root).unwrap();
        save_working_directory(&settings, &old_list, &b).unwrap();
        let (after_restart, saved) = read_working_directories(&settings, &root).unwrap();
        save_working_directory(&settings, &saved, &a).unwrap();
        let (switched_back, final_list) = read_working_directories(&settings, &root).unwrap();
        fs::remove_file(settings).unwrap();

        assert_eq!(old_active, a);
        assert_eq!(after_restart, b);
        assert_eq!(switched_back, a);
        assert_eq!(final_list, [a, b]);
    }
}

#[tauri::command]
fn set_working_directory(
    path: String,
    app: tauri::AppHandle,
    state: State<'_, Arc<RuntimeState>>,
) -> Result<bool, String> {
    if path.trim().is_empty() || path.chars().any(char::is_control) {
        return Err("请选择有效的工作目录。".into());
    }
    let path = fs::canonicalize(path).map_err(|error| format!("无法访问所选目录：{error}"))?;
    if !path.is_dir() {
        return Err("所选路径不是文件夹。".into());
    }
    if path.to_str().is_none_or(|value| value.chars().any(char::is_control)) {
        return Err("目录路径无法保存，请选择名称可显示且不含控制字符的文件夹。".into());
    }

    {
        let mut info = state.info.lock().expect("runtime info mutex poisoned");
        if path == info.working_directory {
            return Ok(false);
        }
        let settings_path = state.app_home.join("working-directory.txt");
        info.working_directories = save_working_directory(&settings_path, &info.working_directories, &path)?;
        info.working_directory = path;
    }

    stop_runtime(state.inner());
    if let Err(error) = start_runtime(&app, state.inner()) {
        state.fail(error.clone());
        return Err(error);
    }
    Ok(true)
}

#[tauri::command]
fn save_model_api_key(api_key: String, state: State<'_, Arc<RuntimeState>>) -> Result<(), String> {
    let api_key = api_key.trim();
    if api_key.is_empty() {
        return Err("请输入 MiniMax API Key。".into());
    }
    if api_key.chars().any(|character| character.is_control()) {
        return Err("API Key 不能包含换行或控制字符。".into());
    }

    let config_path = PathBuf::from(&state.info().config_path);
    let contents = fs::read_to_string(&config_path)
        .map_err(|error| format!("无法读取模型配置文件：{error}"))?;
    let mut updated = String::new();
    let mut found = false;
    for line in contents.lines() {
        if line.trim_start().starts_with("LLM_API_KEY=") {
            if !found {
                updated.push_str("LLM_API_KEY=");
                updated.push_str(api_key);
                updated.push('\n');
                found = true;
            }
        } else {
            updated.push_str(line);
            updated.push('\n');
        }
    }
    if !found {
        updated.push_str("LLM_API_KEY=");
        updated.push_str(api_key);
        updated.push('\n');
    }
    fs::write(&config_path, updated)
        .map_err(|error| format!("无法保存模型 API Key：{error}"))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&config_path, fs::Permissions::from_mode(0o600))
            .map_err(|error| format!("无法限制配置文件权限：{error}"))?;
    }
    Ok(())
}

fn migrate_legacy_model_defaults(config_path: &PathBuf) -> Result<(), String> {
    let contents = fs::read_to_string(config_path)
        .map_err(|error| format!("无法读取桌面配置文件：{error}"))?;
    let value = |name: &str| {
        contents.lines().find_map(|line| {
            let (key, value) = line.split_once('=')?;
            (key.trim() == name).then_some(value.trim())
        })
    };
    if value("LLM_PROVIDER") != Some("deepseek")
        || value("LLM_MODEL") != Some("deepseek-chat")
        || value("LLM_BASE_URL").is_some_and(|base_url| !base_url.is_empty())
    {
        return Ok(());
    }

    let mut updated = String::new();
    let mut provider_found = false;
    let mut model_found = false;
    let mut base_url_found = false;
    for line in contents.lines() {
        match line.split_once('=').map(|(name, _)| name.trim()) {
            Some("LLM_PROVIDER") => {
                updated.push_str("LLM_PROVIDER=openai\n");
                provider_found = true;
            }
            Some("LLM_MODEL") => {
                updated.push_str("LLM_MODEL=MiniMax-M3\n");
                model_found = true;
            }
            Some("LLM_BASE_URL") => {
                updated.push_str("LLM_BASE_URL=https://api.minimaxi.com/v1\n");
                base_url_found = true;
            }
            _ => {
                updated.push_str(line);
                updated.push('\n');
            }
        }
    }
    if !provider_found {
        updated.push_str("LLM_PROVIDER=openai\n");
    }
    if !model_found {
        updated.push_str("LLM_MODEL=MiniMax-M3\n");
    }
    if !base_url_found {
        updated.push_str("LLM_BASE_URL=https://api.minimaxi.com/v1\n");
    }
    fs::write(config_path, updated)
        .map_err(|error| format!("无法更新桌面默认模型配置：{error}"))
}

fn create_runtime_state(app: &tauri::AppHandle) -> Result<Arc<RuntimeState>, String> {
    let app_home = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("无法确定应用支持目录：{error}"))?;
    let logs_dir = app_home.join("logs");
    fs::create_dir_all(&logs_dir)
        .map_err(|error| format!("无法创建应用数据目录：{error}"))?;

    let default_working_directory = app_home.join("data");
    fs::create_dir_all(&default_working_directory)
        .map_err(|error| format!("无法创建默认工作目录：{error}"))?;
    let (working_directory, working_directories) = read_working_directories(
        &app_home.join("working-directory.txt"), &default_working_directory,
    )?;

    let config_path = app_home.join(".env");
    if !config_path.exists() {
        fs::write(
            &config_path,
            "# 阿序 · Axu 桌面运行时配置。修改后重启应用生效。\n\
             LLM_PROVIDER=openai\n\
             LLM_MODEL=MiniMax-M3\n\
             LLM_BASE_URL=https://api.minimaxi.com/v1\n\
             LLM_TEMPERATURE=0.3\n\
             LLM_API_KEY=\n",
        )
        .map_err(|error| format!("无法创建桌面配置文件：{error}"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&config_path, fs::Permissions::from_mode(0o600))
                .map_err(|error| format!("无法限制配置文件权限：{error}"))?;
        }
    }
    migrate_legacy_model_defaults(&config_path)?;

    let log_path = logs_dir.join("runtime.log");
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|error| format!("无法打开运行日志：{error}"))?;

    Ok(Arc::new(RuntimeState {
        info: Mutex::new(RuntimeInfo {
            service_url: String::new(),
            access_token: String::new(),
            readiness: String::new(),
            config_path: config_path.to_string_lossy().into_owned(),
            log_path: log_path.to_string_lossy().into_owned(),
            working_directory,
            working_directories,
            status: "starting".into(),
            error: None,
        }),
        child: Mutex::new(None),
        log_file: Mutex::new(log_file),
        app_home,
        started_at: Mutex::new(Instant::now()),
        exited: AtomicBool::new(false),
        stopping: AtomicBool::new(false),
    }))
}

fn start_runtime(app: &tauri::AppHandle, state: &Arc<RuntimeState>) -> Result<(), String> {
    // 每次启动轮换凭证。健康检查验证独立标记，不向探测端口发送访问凭证。
    let mut startup_random = [0_u8; 64];
    getrandom::fill(&mut startup_random)
        .map_err(|error| format!("无法生成桌面服务凭证：{error}"))?;
    let encode = |bytes: &[u8]| -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    };
    let access_token = encode(&startup_random[..32]);
    let readiness = encode(&startup_random[32..]);

    let port = TcpListener::bind(("127.0.0.1", 0))
        .and_then(|listener| listener.local_addr())
        .map_err(|error| format!("无法分配本机服务端口：{error}"))?
        .port();
    let service_url = format!("http://127.0.0.1:{port}");
    let port_arg = port.to_string();
    let working_directory = state
        .info
        .lock()
        .map_err(|_| "无法读取 Agent 工作目录。".to_string())?
        .working_directory
        .clone();

    let command = app
        .shell()
        .sidecar(SIDECAR_NAME)
        .map_err(|error| format!("找不到已打包的 Python sidecar：{error}"))?
        .args(["--host", "127.0.0.1", "--port", &port_arg, "--no-schedule"])
        .env("AUTOMATION_AGENT_HOME", &state.app_home)
        .env("__CLI_DATA_DIR__", &working_directory)
        .env("AUTOMATION_AGENT_SIDECAR_TOKEN", &access_token)
        .env("AUTOMATION_AGENT_SIDECAR_READINESS", &readiness)
        .set_process_group(true)
        .kill_on_drop(true);
    let (mut receiver, child) = command
        .spawn()
        .map_err(|error| format!("无法启动 Python sidecar：{error}"))?;

    {
        let mut info = state.info.lock().expect("runtime info mutex poisoned");
        info.service_url = service_url.clone();
        info.access_token = access_token;
        info.readiness = readiness;
        info.status = "starting".into();
        info.error = None;
    }
    *state.started_at.lock().expect("runtime start time mutex poisoned") = Instant::now();
    state.exited.store(false, Ordering::Release);
    state.stopping.store(false, Ordering::Release);
    *state.child.lock().expect("runtime child mutex poisoned") = Some(child);
    state.log(&format!("启动 Agent 服务：{service_url}"));

    let state_for_events = Arc::clone(state);
    tauri::async_runtime::spawn(async move {
        while let Some(event) = receiver.recv().await {
            match event {
                CommandEvent::Stdout(bytes) | CommandEvent::Stderr(bytes) => {
                    state_for_events.log_output(&bytes);
                }
                CommandEvent::Error(error) => {
                    state_for_events.log(&format!("sidecar 通道错误：{error}"));
                }
                CommandEvent::Terminated(payload) => {
                    state_for_events.exited.store(true, Ordering::Release);
                    let mut info = state_for_events
                        .info
                        .lock()
                        .expect("runtime info mutex poisoned");
                    if state_for_events.stopping.load(Ordering::Acquire) {
                        info.status = "stopped".into();
                        info.error = None;
                    } else {
                        info.status = "failed".into();
                        info.error = Some(format!(
                            "Python 服务已退出（code={:?}, signal={:?}）。",
                            payload.code, payload.signal
                        ));
                    }
                    state_for_events.log(&format!(
                        "Python sidecar 退出：code={:?}, signal={:?}",
                        payload.code, payload.signal
                    ));
                }
                _ => {}
            }
        }
    });

    Ok(())
}

fn health_check(service_url: &str, readiness: &str) -> bool {
    let Some(port) = service_url
        .strip_prefix("http://127.0.0.1:")
        .and_then(|value| value.parse::<u16>().ok())
    else {
        return false;
    };
    let address = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(250)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let request = format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = BufReader::new(stream);
    let mut status_line = String::new();
    if response.read_line(&mut status_line).is_err()
        || !status_line.starts_with("HTTP/1.1 200 ")
    {
        return false;
    }

    let mut content_length = None;
    loop {
        let mut header = String::new();
        if response.read_line(&mut header).is_err() || header.is_empty() {
            return false;
        }
        if header == "\r\n" {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse::<usize>().ok();
            }
        }
    }

    let Some(content_length) = content_length.filter(|length| *length <= 4096) else {
        return false;
    };
    let mut body = vec![0; content_length];
    if response.read_exact(&mut body).is_err() {
        return false;
    }
    let body = String::from_utf8_lossy(&body);
    body.contains("\"status\":\"ok\"")
        && body.contains(&format!("\"readiness\":\"{readiness}\""))
}

fn stop_runtime(state: &RuntimeState) {
    let child = state.child.lock().ok().and_then(|mut child| child.take());
    let Some(child) = child else {
        return;
    };

    let runtime = state.info();
    let service_running = health_check(&runtime.service_url, &runtime.readiness);
    if state.exited.load(Ordering::Acquire) && !service_running {
        return;
    }

    state.stopping.store(true, Ordering::Release);
    {
        let mut info = state.info.lock().expect("runtime info mutex poisoned");
        info.status = "stopping".into();
        info.error = None;
    }
    state.log("正在有序停止 Agent 服务。");

    let pid = child.pid() as libc::pid_t;
    // PyInstaller one-file mode can keep the server in a child process, so
    // signal the dedicated process group created for this sidecar.
    unsafe {
        libc::kill(-pid, libc::SIGTERM);
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    while (!state.exited.load(Ordering::Acquire)
        || health_check(&runtime.service_url, &runtime.readiness))
        && Instant::now() < deadline
    {
        thread::sleep(Duration::from_millis(50));
    }
    if !state.exited.load(Ordering::Acquire)
        || health_check(&runtime.service_url, &runtime.readiness)
    {
        state.log("Agent 服务未在 3 秒内退出，正在终止 sidecar 进程组。");
        let _ = child.kill();
    }
}

/// 创建并运行 Tauri 应用。
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            runtime_info,
            read_runtime_log,
            restart_runtime,
            save_model_api_key,
            set_working_directory
        ])
        .setup(|app| {
            let state = create_runtime_state(app.handle())
                .map_err(std::io::Error::other)?;
            app.manage(Arc::clone(&state));
            if let Err(error) = start_runtime(app.handle(), &state) {
                state.fail(error);
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                if let Some(state) = window.try_state::<Arc<RuntimeState>>() {
                    stop_runtime(state.inner());
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("启动阿序 · Axu 桌面应用失败");

    app.run(|app_handle, event| {
        if matches!(event, tauri::RunEvent::Exit) {
            if let Some(state) = app_handle.try_state::<Arc<RuntimeState>>() {
                stop_runtime(state.inner());
            }
        }
    });
}
