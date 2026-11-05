"""
PII 脱敏中间件。

对应课件：
- 08 章 2.3（courseware/langChain/尚硅谷-08-中间件.pdf）：内置中间件。
- 课件未覆盖 PII 脱敏，本模块按项目需要自行配置。

为什么需要：
提示词与工具结果都会发给模型供应商，还可能因 LangSmith 追踪上云。
用户指令、文件内容里的手机号 / 邮箱 / 身份证号不该原样出去。

脱敏哪些内容：
- **用户指令** —— 最后一条 HumanMessage
- **工具结果** —— 最后一条 AIMessage 之后的全部 ToolMessage。
  文件工具读出来的内容走这条，是本项目最主要的泄露通道。

**不**脱敏模型输出：
模型输出多半来自用户自己的输入或工具结果，用户自己知道那是什么。
脱敏输出会让「提醒我自己邮箱是多少」这种场景失效。

占位符格式（strategy="redact"）：
    邮箱    → [REDACTED_EMAIL]
    信用卡  → [REDACTED_CREDIT_CARD]
    手机号  → [REDACTED_PHONE_CN]
    身份证  → [REDACTED_ID_CARD_CN]

为什么用 redact 而不是 mask：
mask 只保留末尾几位（手机号是 `****5678`）—— 那几位仍然是真实个人信息，
照样会进供应商和 LangSmith。redact 整个替换掉，一个字符都不漏。

注意：
1. PIIMiddleware 是 langchain 的**内置**中间件，这里只负责配置它。
2. 构造函数一次只接一种 PII 类型，要脱敏多种就得挂多个实例。
3. 脱敏是改写 state 里的消息，所以**脱敏后的内容才会落进 checkpointer**，
   原始 PII 不入库。
"""

from __future__ import annotations

# PIIMiddleware：langchain 内置的 PII 脱敏中间件。
from langchain.agents.middleware.pii import PIIMiddleware

# RUNTIME：应用运行配置，这里读脱敏开关。
from ..config import RUNTIME


# ============================================================
# 1. 自定义检测器
# ============================================================

# 中国大陆手机号：11 位，1 开头，第二位是 3-9。
#
# 前后加 (?<!\d) (?!\d) —— 防止把更长数字串里的一段当手机号。
# 实测「订单号 20261103，共 1234567890 件」不会被误伤。
PHONE_CN_PATTERN = r"(?<!\d)1[3-9]\d{9}(?!\d)"

# 中国大陆居民身份证号：18 位，前 17 位数字，最后一位是数字或 X。
#
# 同样的边界断言，防止匹配更长数字串的片段。
ID_CARD_CN_PATTERN = r"(?<!\d)\d{17}[\dXx](?!\d)"


# ============================================================
# 2. 中间件列表
# ============================================================

def build_pii_middleware() -> list[PIIMiddleware]:
    """按配置构造 PII 脱敏中间件列表。

    返回：
        中间件实例列表。脱敏开关关闭时返回空列表 ——
        graph.py 用 `*build_pii_middleware()` 展开，空列表展开后什么都没有，
        等于没挂这个中间件。

    注意：
        每种 PII 一个实例 —— PIIMiddleware 的构造函数一次只接一种类型。
        顺序无所谓（各自的正则互不重叠），但**全部要在模型看到之前跑完**。
    """
    if not RUNTIME.pii_masking:
        return []

    return [
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
            apply_to_tool_results=True,
        ),
        PIIMiddleware(
            "credit_card",
            strategy="redact",
            apply_to_input=True,
            apply_to_tool_results=True,
        ),
        PIIMiddleware(
            "phone_cn",
            strategy="redact",
            detector=PHONE_CN_PATTERN,
            apply_to_input=True,
            apply_to_tool_results=True,
        ),
        PIIMiddleware(
            "id_card_cn",
            strategy="redact",
            detector=ID_CARD_CN_PATTERN,
            apply_to_input=True,
            apply_to_tool_results=True,
        ),
    ]
