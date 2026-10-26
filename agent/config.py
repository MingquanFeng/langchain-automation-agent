"""集中管理运行时配置。所有可调项都从环境变量读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class ModelConfig:
    """模型接入配置。base_url 为空时走供应商官方端点。"""

    model: str = field(default_factory=lambda: _env("LLM_MODEL", "deepseek-chat"))
    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "deepseek"))
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL"))
    temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.3")))

    def build(self):
        from langchain.chat_models import init_chat_model

        kwargs: dict = {"model": self.model, "model_provider": self.provider}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return init_chat_model(**kwargs)


@dataclass(frozen=True)
class RuntimeConfig:
    """运行时外壳配置。"""

    # 调度器
    scheduler_enabled: bool = field(default_factory=lambda: _env("SCHEDULER_ENABLED", "1") == "1")
    timezone: str = field(default_factory=lambda: _env("SCHEDULER_TIMEZONE", "Asia/Shanghai"))

    # HTTP 服务
    api_host: str = field(default_factory=lambda: _env("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(_env("API_PORT", "8000")))

    # 记忆
    memory_backend: str = field(default_factory=lambda: _env("MEMORY_BACKEND", "sqlite"))
    sqlite_path: Path = field(default_factory=lambda: DATA_DIR / "memory.sqlite")

    # 通知出口：webhook 留空则只写终端
    notify_webhook: str = field(default_factory=lambda: _env("NOTIFY_WEBHOOK"))
    notify_channel: str = field(default_factory=lambda: _env("NOTIFY_CHANNEL", "terminal"))

    # 预算守卫：单次任务最大模型调用次数
    max_model_calls: int = field(default_factory=lambda: int(_env("MAX_MODEL_CALLS", "30")))


MODEL = ModelConfig()
RUNTIME = RuntimeConfig()
