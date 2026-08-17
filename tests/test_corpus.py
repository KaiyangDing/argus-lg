"""corpus.py 纯函数面：manifest 指纹 / 切块契约 / jsonl 往返。

PyPDFLoader 的真 PDF 解析不进单测（无合成 PDF 设施），由 ingest 实跑 +
在场率读数验收——见 PLAN S2。
"""

import hashlib
from pathlib import Path

import pytest
from langchain_core.documents import Document

from argus_lg.corpus import (
    CHUNK_SIZE,
    build_manifest,
    chunks_to_rows,
    read_jsonl,
    sha256_file,
    split_pages,
    write_jsonl,
)


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert sha256_file(f) == hashlib.sha256(b"abc").hexdigest()


def test_build_manifest_sorted_relative_and_fingerprinted(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "b" / "z-doc.pdf").write_bytes(b"zz")
    (tmp_path / "a" / "a-doc.pdf").write_bytes(b"aa")
    manifest = build_manifest(tmp_path)
    assert list(manifest) == ["a-doc", "z-doc"]
    assert manifest["a-doc"]["file"] == "a/a-doc.pdf"
    assert manifest["a-doc"]["sha256"] == hashlib.sha256(b"aa").hexdigest()
    assert manifest["z-doc"]["bytes"] == 2
    assert build_manifest(tmp_path) == manifest  # 确定性


def test_build_manifest_rejects_duplicate_source_id(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "dup.pdf").write_bytes(b"1")
    (tmp_path / "b" / "dup.pdf").write_bytes(b"2")
    with pytest.raises(ValueError, match="重复"):
        build_manifest(tmp_path)


def _page(text: str, page: int) -> Document:
    return Document(
        page_content=text,
        metadata={"source_id": "t-doc", "company": "t", "page": page},
    )


def test_split_pages_contract() -> None:
    long_text = ("甲乙丙丁" * 30 + "。") * 10  # 10 句 × 121 字符
    chunks = split_pages([_page(long_text, 1), _page("尾页短文。", 2)])

    assert len(chunks) >= 3
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)
    assert all(c.page_content.endswith("。") for c in chunks)  # keep_separator="end"
    for seq, c in enumerate(chunks):
        assert c.metadata["seq"] == seq
        assert c.metadata["chunk_id"] == f"t-doc:{seq}"
        assert c.metadata["company"] == "t"
    assert chunks[0].metadata["page"] == 1
    assert chunks[-1].metadata["page"] == 2


def test_jsonl_round_trip(tmp_path: Path) -> None:
    rows = chunks_to_rows(split_pages([_page("三只松鼠营收百亿。" * 20, 7)]))
    out = tmp_path / "d" / "rows.jsonl"
    write_jsonl(rows, out)
    assert read_jsonl(out) == rows
    assert rows[0]["page"] == 7
    assert isinstance(rows[0]["text"], str)
