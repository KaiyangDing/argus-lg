"""语料摄取 CLI（零成本，无 LLM 调用）：

corpus/raw/**/*.pdf → corpus/manifest.json（sha256 指纹，入 git）
                    → corpus/derived/documents.jsonl（块，gitignore）
                    → corpus/derived/ingest_report.json（逐文档记账，入 git 由报告引用）

    uv run python scripts/ingest.py
"""

import json
import sys
import time
from pathlib import Path

from argus_lg.corpus import build_manifest, chunks_to_rows, load_pdf_pages, split_pages, write_jsonl

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "corpus" / "raw"
DERIVED = REPO / "corpus" / "derived"


def main() -> None:
    t0 = time.perf_counter()
    manifest = build_manifest(RAW)
    (REPO / "corpus" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    all_rows: list[dict[str, object]] = []
    report: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for source_id, entry in manifest.items():
        path = RAW / str(entry["file"])
        try:
            pages = load_pdf_pages(path)
        except Exception as exc:  # noqa: BLE001 逐文档记账不断批，坏文档入 failures
            failures.append(source_id)
            report[source_id] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[FAIL] {source_id}: {exc}", file=sys.stderr)
            continue
        chunks = split_pages(pages)
        rows = chunks_to_rows(chunks)
        all_rows.extend(rows)
        chars = sum(len(str(r["text"])) for r in rows)
        report[source_id] = {"pages": len(pages), "chunks": len(rows), "chars": chars}
        print(f"[ok] {source_id}: {len(pages)} 页 → {len(rows)} 块 / {chars} 字符")

    write_jsonl(all_rows, DERIVED / "documents.jsonl")
    summary = {
        "docs_ok": len(manifest) - len(failures),
        "docs_failed": failures,
        "chunks": len(all_rows),
        "chars": sum(len(str(r["text"])) for r in all_rows),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    (DERIVED / "ingest_report.json").write_text(
        json.dumps({"summary": summary, "docs": report}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(
        f"\n合计: {summary['docs_ok']}/{len(manifest)} 文档"
        f" / {summary['chunks']} 块 / {summary['chars']} 字符 / {summary['elapsed_s']}s"
    )
    if failures:
        raise SystemExit(f"失败文档: {failures}")


if __name__ == "__main__":
    main()
