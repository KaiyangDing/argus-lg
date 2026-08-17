"""冒烟：真调用一次 qwen-flash，验证 ChatOpenAI(阿里兼容端点) 集成与用量读数。

S1 撞墙记录：首版走 ChatTongyi(langchain-community)，实测不回填标准 usage_metadata
且 community 整包日落——2026-08-17 拍板换 OpenAI 兼容端点（PLAN 撞墙与替换记录①）。

花钱命令（≈¥0.001/次），由用户手动触发：
    uv run python scripts/smoke_llm.py
"""

import os
import time
from decimal import Decimal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

DASHSCOPE_COMPAT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 价目：百炼官网价目页 2026-08-04 查询、经旧仓控制台对账核实；变价改这里。
PRICES_PER_1K: dict[str, tuple[Decimal, Decimal]] = {
    "qwen-flash": (Decimal("0.0012"), Decimal("0.0072")),
    "qwen-plus": (Decimal("0.002"), Decimal("0.008")),
    "text-embedding-v4": (Decimal("0.0005"), Decimal(0)),
}

MODEL = "qwen-flash"


def estimate_yuan(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    price_in, price_out = PRICES_PER_1K[model]
    return (Decimal(input_tokens) * price_in + Decimal(output_tokens) * price_out) / 1000


def main() -> None:
    load_dotenv()
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("缺 DASHSCOPE_API_KEY（.env），冒烟中止。")

    llm = ChatOpenAI(
        model=MODEL,
        base_url=DASHSCOPE_COMPAT_BASE,
        api_key=os.environ["DASHSCOPE_API_KEY"],
    )
    t0 = time.perf_counter()
    reply = llm.invoke("用一句话说明你是什么模型。")
    elapsed = time.perf_counter() - t0

    print(f"模型: {MODEL}")
    print(f"回复: {reply.content}")
    print(f"耗时: {elapsed:.1f}s")

    usage = reply.usage_metadata
    if usage is None:
        # 撞墙点候选：社区集成未回填标准 usage_metadata 时，退回原始元数据人工读数。
        print(f"usage_metadata 为空，response_metadata={reply.response_metadata}")
        return

    cost = estimate_yuan(MODEL, usage["input_tokens"], usage["output_tokens"])
    print(f"用量: 入 {usage['input_tokens']} tok / 出 {usage['output_tokens']} tok")
    print(f"折算: ≈¥{cost:.6f}")


if __name__ == "__main__":
    main()
