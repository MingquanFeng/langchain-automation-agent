"""运行记录与客户端错误文本的最小凭据脱敏。"""

from __future__ import annotations

import re


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_ -]?key|access[_ -]?token|auth(?:orization)?|bearer|"
    r"client[_ -]?secret|password|secret)\b\s*([=:])\s*([^\s,;]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
_URL_PASSWORD = re.compile(r"(https?://[^\s:/@]+:)([^@\s/]+)(@)", re.I)


def redact_sensitive_text(value: object) -> str:
    """保留可诊断的错误上下文，同时替换常见凭据值。"""
    text = str(value)
    text = _BEARER_TOKEN.sub("Bearer [已隐藏]", text)
    text = _URL_PASSWORD.sub(r"\1[已隐藏]\3", text)
    return _SECRET_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[已隐藏]", text
    )
