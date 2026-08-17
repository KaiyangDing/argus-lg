"""向量化（花钱步骤 ≈¥0.76 一次性，用户手动触发；分段落盘可断点续跑）：

    uv run python scripts/embed.py

corpus/derived/documents.jsonl → corpus/derived/vectors.json（InMemoryVectorStore.dump，
gitignore 不入库）。估价按 字符×0.75 折 token × ¥0.0005/1k（text-embedding-v4），
精确金额以百炼控制台为准。
"""

import time
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore

from argus_lg.corpus import read_jsonl
from argus_lg.llm import EMBED_MODEL, PRICES_PER_1K
from argus_lg.retrieval import make_embeddings, rows_to_documents

REPO = Path(__file__).resolve().parent.parent
VECTORS = REPO / "corpus" / "derived" / "vectors.json"
SEGMENT = 500  # 每 500 块 dump 一次：崩溃最多重花一段的钱

PRICE_PER_1K = PRICES_PER_1K[EMBED_MODEL][0]
TOKENS_PER_CHAR = Decimal("0.75")


def est_yuan(chars: int) -> Decimal:
    return Decimal(chars) * TOKENS_PER_CHAR / 1000 * PRICE_PER_1K


def main() -> None:
    load_dotenv()
    rows = read_jsonl(REPO / "corpus" / "derived" / "documents.jsonl")
    docs = rows_to_documents(rows)
    embeddings = make_embeddings()

    if VECTORS.exists():
        store = InMemoryVectorStore.load(str(VECTORS), embeddings)
        done = set(store.store)  # store.store: {chunk_id: 向量记录}，键集即已完成块
        print(f"续跑：vectors.json 已有 {len(done)} 块")
    else:
        store = InMemoryVectorStore(embeddings)
        done: set[str] = set()

    pending = [d for d in docs if d.id not in done]
    total_chars = sum(len(d.page_content) for d in pending)
    print(
        f"待向量化 {len(pending)}/{len(docs)} 块 / {total_chars} 字符 / 预估 ≈¥{est_yuan(total_chars):.4f}"
    )
    if not pending:
        print("无待办，退出。")
        return

    t0 = time.perf_counter()
    for i in range(0, len(pending), SEGMENT):
        batch = pending[i : i + SEGMENT]
        store.add_documents(batch, ids=[str(d.id) for d in batch])
        store.dump(str(VECTORS))
        n = min(i + SEGMENT, len(pending))
        print(f"  {n}/{len(pending)} 已落盘（{time.perf_counter() - t0:.0f}s）")

    print(f"完成：store 共 {len(store.store)} 块 → {VECTORS.name}（精确花费以百炼控制台为准）")


if __name__ == "__main__":
    main()
