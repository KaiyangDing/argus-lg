"""检索层 recall@k 三路对比（BM25 / 向量 / 混合）。

口径：对每条 must 金标要点，以要点原文为查询、公司域内检索 top-k；
锚点组全部命中于检出块集合 = 该要点召回（组间允许不同块，同 presence）。
BM25 零成本；向量/混合每查询产生 query embedding（≈¥0.00002/条，真调用）。

    uv run python scripts/eval_retrieval.py --modes bm25      # 零成本
    uv run python scripts/eval_retrieval.py                   # 三路，需先 scripts/embed.py
    uv run python scripts/eval_retrieval.py --sub-k 12        # 扩融合池：子检索器各出 12，融合后取 top-k
    uv run python scripts/eval_retrieval.py --diag            # 对 miss 要点打印锚点组真实名次（裁决 排位差/嵌入错配）
    uv run python scripts/eval_retrieval.py --sweep 4,8,12,16 # 召回曲线：深池检索一次，免费切出各 k 的召回
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from argus_lg.corpus import read_jsonl
from argus_lg.presence import Anchor, check_kp, load_anchors, load_cases, normalize
from argus_lg.retrieval import (
    build_bm25,
    build_hybrid,
    make_embeddings,
    rows_to_documents,
    vector_retriever,
)

REPO = Path(__file__).resolve().parent.parent
VECTORS = REPO / "corpus" / "derived" / "vectors.json"
DIAG_DEPTH = 200


def group_rank(docs: list[Document], group: list[str]) -> int | None:
    """锚点组在排序里首个命中块的名次（1 起）；None=前 DIAG_DEPTH 名不见。"""
    forms = [normalize(f) for f in group]
    for i, doc in enumerate(docs, start=1):
        text = normalize(doc.page_content)
        if any(form in text for form in forms):
            return i
    return None


def depth_rankings(
    bm25: object, store: InMemoryVectorStore | None, company: str, query: str, modes: list[str]
) -> dict[str, list[Document]]:
    """各路深度 DIAG_DEPTH 的完整排序（bm25.k 现场调为深度）。"""
    bm25.k = DIAG_DEPTH  # type: ignore[attr-defined]
    rankings: dict[str, list[Document]] = {}
    vec = vector_retriever(store, company, DIAG_DEPTH) if store is not None else None
    if "bm25" in modes:
        rankings["bm25"] = bm25.invoke(query)  # type: ignore[attr-defined]
    if "vector" in modes and vec is not None:
        rankings["vector"] = vec.invoke(query)
    if "hybrid" in modes and vec is not None:
        rankings["hybrid"] = build_hybrid(bm25, vec).invoke(query)
    return rankings


def run_sweep(
    modes: list[str],
    ks: list[int],
    anchors: list[Anchor],
    kp_text: dict[tuple[str, str], str],
    bm25_by_company: dict[str, object],
    store: InMemoryVectorStore | None,
) -> None:
    totals: dict[str, dict[int, int]] = {m: dict.fromkeys(ks, 0) for m in modes}
    needed_hybrid: list[tuple[str, int | None]] = []
    for anchor in anchors:
        company = anchor["case_id"]
        query = kp_text[(company, anchor["kp_id"])]
        rankings = depth_rankings(bm25_by_company[company], store, company, query, modes)
        for mode, docs in rankings.items():
            texts = [normalize(d.page_content) for d in docs]
            for k in ks:
                totals[mode][k] += int(check_kp(anchor, texts[:k])["hit"])
        if "hybrid" in rankings:
            ranks = [group_rank(rankings["hybrid"], g) for g in anchor["groups"]]
            needed = None if any(r is None for r in ranks) else max(r for r in ranks if r)
            needed_hybrid.append((f"{anchor['case_id']}/{anchor['kp_id']}", needed))

    print(f"召回曲线（子检索深度 {DIAG_DEPTH}，融合后切 top-k）：")
    header = "  mode    " + "".join(f"@{k:<6}" for k in ks)
    print(header)
    for mode in modes:
        cells = "".join(f"{totals[mode][k]}/{len(anchors):<5}" for k in ks)
        print(f"  {mode:<8}{cells}")
    if needed_hybrid:
        worst = sorted(needed_hybrid, key=lambda x: (x[1] is None, x[1] or 0), reverse=True)[:6]
        shown = ", ".join(f"{name}→{n if n else '>' + str(DIAG_DEPTH)}" for name, n in worst)
        print(f"  混合补齐单条要点所需最深 k（最费力前 6）：{shown}")


def diag_kp(anchor: Anchor, rankings: dict[str, list[Document]]) -> None:
    print(f"  [diag] {anchor['case_id']}/{anchor['kp_id']}")
    for mode, docs in rankings.items():
        parts = []
        for gi, group in enumerate(anchor["groups"], start=1):
            rank = group_rank(docs, group)
            shown = group[0] if len(group[0]) <= 16 else group[0][:16] + "…"
            parts.append(f"组{gi}({shown}): {'rank ' + str(rank) if rank else f'>{DIAG_DEPTH}'}")
        print(f"    {mode:<7}" + "   ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="bm25,vector,hybrid")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--sub-k", type=int, default=None, help="子检索器候选数（默认=k）")
    parser.add_argument("--diag", action="store_true", help="对 miss 要点打印锚点组真实名次")
    parser.add_argument("--sweep", default=None, help="逗号分隔的 k 列表：深池检索一次切出召回曲线")
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    sub_k = args.sub_k if args.sub_k is not None else args.k

    load_dotenv()
    rows = read_jsonl(REPO / "corpus" / "derived" / "documents.jsonl")
    docs = rows_to_documents(rows)
    cases = load_cases(REPO / "eval" / "cases.jsonl")
    anchors = load_anchors(REPO / "eval" / "presence_anchors.json", cases)
    kp_text = {
        (case["case_id"], kp["kp_id"]): kp["text"] for case in cases for kp in case["keypoints"]
    }

    store = None
    if {"vector", "hybrid"} & set(modes):
        if not VECTORS.exists():
            raise SystemExit(
                "缺 corpus/derived/vectors.json——先跑 scripts/embed.py，或 --modes bm25"
            )
        store = InMemoryVectorStore.load(str(VECTORS), make_embeddings())

    by_company: dict[str, list[Document]] = {}
    for d in docs:
        by_company.setdefault(str(d.metadata["company"]), []).append(d)
    # BM25 索引每公司建一次（jieba 分词是大头）；k 是查询期属性，按用途现调
    bm25_by_company = {c: build_bm25(cdocs, sub_k) for c, cdocs in by_company.items()}

    if args.sweep:
        ks = sorted({int(x) for x in args.sweep.split(",") if x.strip()})
        run_sweep(modes, ks, anchors, kp_text, bm25_by_company, store)
        return

    totals = dict.fromkeys(modes, 0)
    missed: list[Anchor] = []
    print(f"k={args.k} sub_k={sub_k}")
    for anchor in anchors:
        company = anchor["case_id"]
        query = kp_text[(company, anchor["kp_id"])]
        bm25 = bm25_by_company[company]
        marks = []
        kp_missed = False
        for mode in modes:
            bm25.k = sub_k
            if mode == "bm25":
                retriever = bm25
            elif mode == "vector":
                retriever = vector_retriever(store, company, sub_k)
            else:
                retriever = build_hybrid(bm25, vector_retriever(store, company, sub_k))
            out = retriever.invoke(query)[: args.k]
            texts = [normalize(d.page_content) for d in out]
            hit = check_kp(anchor, texts)["hit"]
            totals[mode] += int(hit)
            kp_missed = kp_missed or not hit
            marks.append(f"{mode}:{'✓' if hit else '✗'}")
        print(f"  {anchor['case_id']}/{anchor['kp_id']}  " + "  ".join(marks))
        if kp_missed:
            missed.append(anchor)

    line = " / ".join(f"{m} {totals[m]}/{len(anchors)}" for m in modes)
    print(f"\nrecall@{args.k} (sub_k={sub_k}): {line}")

    if args.diag and missed:
        print(
            f"\n诊断（完整排序前 {DIAG_DEPTH} 名，rank 小=排位差一口气，>{DIAG_DEPTH}=嵌入/词面根本够不着）："
        )
        for anchor in missed:
            company = anchor["case_id"]
            query = kp_text[(company, anchor["kp_id"])]
            rankings = depth_rankings(
                bm25_by_company[company], store, company, query, ["bm25", "vector", "hybrid"]
            )
            diag_kp(anchor, rankings)


if __name__ == "__main__":
    main()
