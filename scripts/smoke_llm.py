"""冒烟：真调用一次 qwen-flash，验证 ChatOpenAI(阿里兼容端点) 集成与用量读数。

S1 撞墙记录：首版走 ChatTongyi(langchain-community)，实测不回填标准 usage_metadata
且 community 整包日落——2026-08-17 拍板换 OpenAI 兼容端点（PLAN 撞墙与替换记录①）。

花钱命令（≈¥0.001/次），由用户手动触发：
    uv run python scripts/smoke_llm.py
"""

import os
import time

from dotenv import load_dotenv

from argus_lg.llm import CHAT_MODEL as MODEL
from argus_lg.llm import estimate_yuan, make_chat


def main() -> None:
    load_dotenv()
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("缺 DASHSCOPE_API_KEY（.env），冒烟中止。")

    llm = make_chat()
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
