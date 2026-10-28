
"""
集中管理 Agent 的模型配置和运行时配置。

主要职责：
1. 从 .env 或系统环境变量中读取配置。
2. 使用 dataclass 将配置封装成对象。
3. 提供 MODEL.build() 方法创建大模型实例。
4. 统一管理调度器、HTTP、记忆、通知等参数。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# 1. 加载环境变量
# ============================================================

# 将 .env 文件中的变量加载到当前进程环境中。
# override=True 表示 .env 中的同名变量覆盖已有环境变量。
load_dotenv(override=True)


# ============================================================
# 2. 定义项目目录
# ============================================================

# __file__：当前 config.py 文件路径。
# resolve()：转换为规范化的绝对路径。
# parent.parent：向上查找两级目录。
# 假设文件位于 project/agent/config.py，
# 则 ROOT 指向 project 目录。
ROOT = Path(__file__).resolve().parent.parent

# 使用 Path 的 / 运算符拼接路径。
# 最终得到 project/data。
DATA_DIR = ROOT / "data"

# 创建数据目录。
# exist_ok=True：目录已存在时不报错。
DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# 3. 环境变量读取辅助函数
# ============================================================

def _env(name: str, default: str = "") -> str:
    """
    读取环境变量并去除首尾空白字符。

    参数：
        name：环境变量名称。
        default：环境变量不存在时使用的默认值。

    返回：
        字符串形式的配置值。

    注意：
        如果环境变量存在但为空字符串，
        不会自动使用 default。
    """
    return os.getenv(name, default).strip()


# ============================================================
# 4. 模型配置
# ============================================================

@dataclass(frozen=True)
class ModelConfig:
    """
    大模型接入配置。

    @dataclass：
        自动生成 __init__ 等方法。

    frozen=True：
        禁止通过普通属性赋值修改实例字段。

    default_factory：
        在创建实例且未显式传入字段时，
        调用指定函数生成默认值。
    """

    # 模型名称。
    # 默认使用 DeepSeek 的 deepseek-chat。
    model: str = field(
        default_factory=lambda: _env(
            "LLM_MODEL",
            "deepseek-chat"
        )
    )

    # 模型供应商。
    # 用于告诉 LangChain 使用哪个供应商集成。
    provider: str = field(
        default_factory=lambda: _env(
            "LLM_PROVIDER",
            "deepseek"
        )
    )

    # 模型 API 密钥。
    # 未配置时返回空字符串。
    api_key: str = field(
        default_factory=lambda: _env(
            "LLM_API_KEY"
        )
    )

    # 自定义模型 API 地址。
    # 为空时不显式传入 base_url，
    # 由模型供应商集成决定默认地址。
    base_url: str = field(
        default_factory=lambda: _env(
            "LLM_BASE_URL"
        )
    )

    # 模型采样温度。
    # float() 将环境变量字符串转换成浮点数。
    # 默认值为 0.3。
    temperature: float = field(
        default_factory=lambda: float(
            _env("LLM_TEMPERATURE", "0.3")
        )
    )

    def build(self):
        """
        根据当前配置创建 LangChain 聊天模型。

        返回：
            LangChain ChatModel 实例。

        说明：
            这里只创建模型对象，
            不负责执行 Agent 任务。
        """

        # 在方法内部导入，只有调用 build() 时才加载。
        from langchain.chat_models import init_chat_model

        # 创建模型初始化参数字典。
        # self 表示当前 ModelConfig 实例。
        kwargs: dict = {
            "model": self.model,
            "model_provider": self.provider,
            "temperature": self.temperature,
        }

        # 仅在配置了 API Key 时传入。
        if self.api_key:
            kwargs["api_key"] = self.api_key

        # 仅在配置了自定义地址时传入。
        if self.base_url:
            kwargs["base_url"] = self.base_url

        # **kwargs：将字典解包成关键字参数。
        #
        # 例如：
        # {"model": "deepseek-chat"}
        #
        # 等价于：
        # init_chat_model(model="deepseek-chat")
        #
        # 注意：部分模型或供应商可能不支持 temperature。
        return init_chat_model(**kwargs)


# ============================================================
# 5. Agent 运行时配置
# ============================================================

@dataclass(frozen=True)
class RuntimeConfig:
    """
    Agent 应用级运行配置。

    与 AgentContext 的区别：
        RuntimeConfig：整个应用的默认运行配置。
        AgentContext：某一次 Agent 调用的上下文。
    """

    # ------------------------
    # 调度器配置
    # ------------------------

    # 是否启用定时调度器。
    # 环境变量为字符串 "1" 时返回 True，
    # 其他值返回 False。
    scheduler_enabled: bool = field(
        default_factory=lambda: (
            _env("SCHEDULER_ENABLED", "1") == "1"
        )
    )

    # 调度器时区。
    timezone: str = field(
        default_factory=lambda: _env(
            "SCHEDULER_TIMEZONE",
            "Asia/Shanghai"
        )
    )

    # ------------------------
    # HTTP 服务配置
    # ------------------------

    # 服务监听地址。
    # 127.0.0.1 表示监听本机回环接口。
    api_host: str = field(
        default_factory=lambda: _env(
            "API_HOST",
            "127.0.0.1"
        )
    )

    # HTTP 服务端口。
    # int() 将环境变量转换成整数。
    api_port: int = field(
        default_factory=lambda: int(
            _env("API_PORT", "8000")
        )
    )

    # ------------------------
    # 记忆配置
    # ------------------------

    # Checkpointer 存储后端。
    # sqlite：使用 SQLite 持久化。
    # 其他值：当前 graph.py 会使用内存存储。
    memory_backend: str = field(
        default_factory=lambda: _env(
            "MEMORY_BACKEND",
            "sqlite"
        )
    )

    # SQLite 数据库文件路径。
    # 默认存放在项目 data 目录下。
    # 支持通过 SQLITE_PATH 环境变量覆盖。
    sqlite_path: Path = field(
        default_factory=lambda: Path(
            _env(
                "SQLITE_PATH",
                str(DATA_DIR / "memory.sqlite")
            )
        )
    )

    # ------------------------
    # 通知配置
    # ------------------------

    # Webhook 通知地址。
    # 为空时由通知模块决定如何处理。
    notify_webhook: str = field(
        default_factory=lambda: _env(
            "NOTIFY_WEBHOOK"
        )
    )

    # 默认通知渠道。
    # terminal：终端输出。
    notify_channel: str = field(
        default_factory=lambda: _env(
            "NOTIFY_CHANNEL",
            "terminal"
        )
    )

    # ------------------------
    # 模型调用预算
    # ------------------------

    # 模型调用次数上限。
    # 默认 30 次。
    #
    # 注意：
    # graph.py 中使用 thread_limit 时，
    # 限制的是同一 thread 的累计调用次数，
    # 并不是单次 invoke 的调用次数。
    max_model_calls: int = field(
        default_factory=lambda: int(
            _env("MAX_MODEL_CALLS", "30")
        )
    )


# ============================================================
# 6. 创建全局配置对象
# ============================================================

# 创建模型配置实例。
# 在实例化时执行各字段的 default_factory。
MODEL = ModelConfig()

# 创建应用运行配置实例。
RUNTIME = RuntimeConfig()
