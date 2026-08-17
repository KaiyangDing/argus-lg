"""金标在场率 CLI（零成本）：documents.jsonl 对照 18 条 must 锚点。

uv run python scripts/check_presence.py
"""

from pathlib import Path

from argus_lg.corpus import read_jsonl
from argus_lg.presence import check_presence, load_anchors, load_cases

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    cases = load_cases(REPO / "eval" / "cases.jsonl")
    anchors = load_anchors(REPO / "eval" / "presence_anchors.json", cases)
    rows = read_jsonl(REPO / "corpus" / "derived" / "documents.jsonl")
    results = check_presence(anchors, rows)

    by_case: dict[str, list] = {}
    for r in results:
        by_case.setdefault(r["case_id"], []).append(r)

    total_hit = 0
    for case_id, kps in by_case.items():
        hits = sum(1 for k in kps if k["hit"])
        total_hit += hits
        print(f"\n[{case_id}] {hits}/{len(kps)}")
        for k in kps:
            mark = "✓" if k["hit"] else "✗"
            missed = "" if k["hit"] else f"  未命中组: {k['missed_groups']}"
            print(f"  {mark} {k['kp_id']} [{k['source']}]{missed}")

    print(f"\n金标在场率: {total_hit}/{len(results)}（argus v3 对照基准: 18/18）")


if __name__ == "__main__":
    main()
