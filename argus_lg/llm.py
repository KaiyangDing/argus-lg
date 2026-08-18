"""模型接入与价目的单一事实源（撞墙记录①路线：ChatOpenAI + 阿里官方 OpenAI 兼容端点）。"""

import os
from decimal import Decimal

from langchain_openai import ChatOpenAI

DASHSCOPE_COMPAT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT_MODEL = "qwen-flash"
EMBED_MODEL = "text-embedding-v4"

# 价目：百炼官网价目页 2026-08-04 查询、经旧仓控制台对账核实；变价改这里。
PRICES_PER_1K: dict[str, tuple[Decimal, Decimal]] = {
    "qwen-flash": (Decimal("0.0012"), Decimal("0.0072")),
    "qwen-plus": (Decimal("0.002"), Decimal("0.008")),
    "text-embedding-v4": (Decimal("0.0005"), Decimal(0)),
}


def make_chat(model: str = CHAT_MODEL, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=DASHSCOPE_COMPAT_BASE,
        api_key=os.environ["DASHSCOPE_API_KEY"],
        temperature=temperature,
        # 黑洞请求事故（v0.2 首跑 30 分钟单连接空转）后钉死：默认 600s 超时会把
        # 服务端拖死的请求放大成半小时黑洞——快速失败，错误可见（devlog 工程问题账）。
        timeout=120.0,
        max_retries=2,
    )


def estimate_yuan(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    price_in, price_out = PRICES_PER_1K[model]
    return (Decimal(input_tokens) * price_in + Decimal(output_tokens) * price_out) / 1000
