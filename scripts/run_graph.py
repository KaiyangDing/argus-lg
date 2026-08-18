"""整图真跑（花钱 ≈¥0.1/次，用户手动触发）：

    uv run python scripts/run_graph.py lpz

产出：stdout 报告 + runtime/report_<slug>.md（gitignore）+ 用量折 ¥ 汇总。
"""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import MemorySaver

from argus_lg.corpus import COMPANIES, corpus_profile, read_jsonl
from argus_lg.graph import build_graph
from argus_lg.llm import estimate_yuan, make_chat
from argus_lg.retrieval import make_embeddings, make_hybrid_search

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("company", help="lpz / yh / szss 或中文名")
    args = parser.parse_args()
    by_name = {v: k for k, v in COMPANIES.items()}
    slug = by_name.get(args.company, args.company)
    if slug not in COMPANIES:
        raise SystemExit(f"未知公司：{args.company}（可选 {list(COMPANIES)}）")

    load_dotenv()
    rows = read_jsonl(REPO / "corpus" / "derived" / "documents.jsonl")
    store = InMemoryVectorStore.load(
        str(REPO / "corpus" / "derived" / "vectors.json"), make_embeddings()
    )
    graph = build_graph(
        make_chat(),
        make_hybrid_search(rows, store),
        profile_fn=lambda s: corpus_profile(rows, s),
    ).compile(checkpointer=MemorySaver())

    usage_cb = UsageMetadataCallbackHandler()
    config = {"configurable": {"thread_id": slug}, "callbacks": [usage_cb]}
    t0 = time.perf_counter()
    # 流式节点进度：黑洞事故后 CLI 必须实时可观测（print 全部 flush）
    print("执行进度：", flush=True)
    for chunk in graph.stream(
        {"company": COMPANIES[slug], "slug": slug}, config=config, stream_mode="updates"
    ):
        for node in chunk:
            print(f"  [{time.perf_counter() - t0:5.0f}s] {node} 完成", flush=True)
    state = graph.get_state(config).values
    elapsed = time.perf_counter() - t0

    print(f"\n方面：{[a['name'] for a in state['aspects']]}")
    for s in state["sections"]:
        print(f"  {s['aspect_id']} {s['name']}: 证据 {len(set(s['refs']))} 条")
    print(f"全局证据 {len(state['evidence'])} 条 / 耗时 {elapsed:.0f}s\n")
    print(state["report"])

    out = REPO / "runtime" / f"report_{slug}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(state["report"] + "\n", encoding="utf-8")
    # 证据随报告落盘：S5 忠实度裁判需按编号取被引证据原文
    (REPO / "runtime" / f"evidence_{slug}.json").write_text(
        json.dumps(state["evidence"], ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"\n已存 {out}（含 evidence_{slug}.json）")

    total = None
    for model, usage in usage_cb.usage_metadata.items():
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        line = f"用量[{model}]: 入 {tokens_in} tok / 出 {tokens_out} tok"
        if model in {"qwen-flash", "qwen-plus", "text-embedding-v4"}:
            cost = estimate_yuan(model, tokens_in, tokens_out)
            total = cost if total is None else total + cost
            line += f" / ≈¥{cost:.4f}"
        print(line)
    if total is not None:
        print(f"本次合计 ≈¥{total:.4f}（精确以百炼控制台为准）")
    else:
        print("未采集到 usage_metadata（撞墙点候选，报给 AI）")


if __name__ == "__main__":
    main()
