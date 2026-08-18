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
    corpus_profile,
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


def test_annotate_page_sections_carries_headers_across_pages() -> None:
    from argus_lg.corpus import annotate_page_sections

    p1 = Document(page_content="某年报\n母公司资产负债表\n资产：", metadata={})
    p2 = Document(page_content="存货 10,751,508.64", metadata={})  # 表头在前页
    p3 = Document(page_content="5、应收账款\n(1).按账龄披露", metadata={})
    p4 = Document(page_content="合计 259,209,498.08", metadata={})
    annotate_page_sections([p1, p2, p3, p4])
    assert p1.metadata["section"] == ""  # 进入首页前无运行头
    assert p2.metadata["section"] == "母公司资产负债表"
    assert p3.metadata["section"] == "母公司资产负债表"
    assert p4.metadata["section"].startswith("5、应收账款")


def test_corpus_profile_data_driven() -> None:
    rows = [
        {"company": "t", "source_id": "t-ar-2024", "text": "x"},
        {"company": "t", "source_id": "t-ar-2024", "text": "y"},
        {"company": "t", "source_id": "t-news-20231104-src", "text": "z"},
        {"company": "other", "source_id": "o-ar-2025", "text": "w"},  # 他司不入
    ]
    profile = corpus_profile(rows, "t")
    assert "可用文档 2 份" in profile
    assert "t-ar-2024（年报，2024）" in profile
    assert "t-news-20231104-src（新闻，2023）" in profile
    assert "覆盖年份：2023、2024" in profile
    assert "最新年份：2024" in profile
    assert "2025" not in profile  # 年份来自本司语料实况，不带先验

    assert corpus_profile(rows, "ghost") == "该公司暂无语料。"

    no_year = corpus_profile([{"company": "t", "source_id": "merger_deck", "text": "x"}], "t")
    assert "未知" in no_year  # 无法解析年份时提示先探明，不瞎猜


def test_jsonl_round_trip(tmp_path: Path) -> None:
    rows = chunks_to_rows(split_pages([_page("三只松鼠营收百亿。" * 20, 7)]))
    out = tmp_path / "d" / "rows.jsonl"
    write_jsonl(rows, out)
    assert read_jsonl(out) == rows
    assert rows[0]["page"] == 7
    assert isinstance(rows[0]["text"], str)
