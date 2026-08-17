"""端到端评测 CLI（花钱 ≈¥0.15/轮，用户手动触发）：

    uv run python scripts/eval_reports.py

前置：三家各真跑一次图（runtime/report_<slug>.md + evidence_<slug>.json）：
    uv run python scripts/run_graph.py lpz   # yh / szss 同
缺哪家就跳过哪家（会提示）。产出 runtime/eval_latest.json。
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler

from argus_lg.corpus import COMPANIES
from argus_lg.eval import aggregate, eval_case
from argus_lg.llm import estimate_yuan, make_chat
from argus_lg.presence import load_cases

REPO = Path(__file__).resolve().parent.parent
RUNTIME = REPO / "runtime"


def main() -> None:
    load_dotenv()
    cases = {c["case_id"]: c for c in load_cases(REPO / "eval" / "cases.jsonl")}
    chat = make_chat()
    usage_cb = UsageMetadataCallbackHandler()

    def judge(model_cls: type, msgs: object) -> object:
        return chat.with_structured_output(model_cls).invoke(msgs, config={"callbacks": [usage_cb]})

    scores = []
    for slug in COMPANIES:
        report_path = RUNTIME / f"report_{slug}.md"
        evidence_path = RUNTIME / f"evidence_{slug}.json"
        if not report_path.exists() or not evidence_path.exists():
            print(f"[跳过] {slug}：缺 {report_path.name} 或 {evidence_path.name}（先 run_graph）")
            continue
        report = report_path.read_text(encoding="utf-8")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        score = eval_case(cases[slug], report, evidence, judge)
        scores.append(score)
        detail = " ".join(
            f"{d['kp_id']}{'✓' if d['covered'] else '✗'}" for d in score["kp_details"]
        )
        print(
            f"[{slug}] 覆盖 {score['covered']}/{score['musts']}（{detail}）"
            f" 陷阱泄漏 {score['trap_leaks']}/{score['traps']}"
            f" 引用率 {score['citation_rate']:.1%}（聚合句 {score['agg_sentences']}，"
            f"越界 {score['invalid_refs']}）"
            f" 忠实度 {score['faith_supported']}/{score['faith_sampled']}"
            f" quote纠错 {score['quote_errors']}"
        )

    if not scores:
        raise SystemExit("无可评报告。")

    agg = aggregate(scores)
    print(f"\n汇总：{json.dumps(agg, ensure_ascii=False)}")
    print(
        "对照（argus v3 端到端，语料与管线不同仅作方向参考）：覆盖 7/18 · 忠实度 100% · 引用率 76.1% · 陷阱 0 泄漏"
    )

    (RUNTIME / "eval_latest.json").write_text(
        json.dumps({"cases": scores, "aggregate": agg}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"已存 {RUNTIME / 'eval_latest.json'}")

    total = None
    for model, usage in usage_cb.usage_metadata.items():
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        if model in {"qwen-flash", "qwen-plus"}:
            cost = estimate_yuan(model, tokens_in, tokens_out)
            total = cost if total is None else total + cost
        print(f"用量[{model}]: 入 {tokens_in} tok / 出 {tokens_out} tok")
    if total is not None:
        print(f"本轮评测合计 ≈¥{total:.4f}（精确以百炼控制台为准）")


if __name__ == "__main__":
    main()
