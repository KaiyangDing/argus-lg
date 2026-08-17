"""检索层：jieba 接线 / BM25 排序 / 向量过滤与落盘往返 / 混合融合。全 Fake 零网络。"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

from argus_lg.retrieval import (
    build_bm25,
    build_hybrid,
    jieba_tokenize,
    rows_to_documents,
    vector_retriever,
)


def _docs() -> list[Document]:
    rows = [
        {
            "chunk_id": "a:0",
            "source_id": "a",
            "company": "lpz",
            "page": 1,
            "text": "良品铺子营业收入下降",
        },
        {
            "chunk_id": "a:1",
            "source_id": "a",
            "company": "lpz",
            "page": 2,
            "text": "三只松鼠坚果业务增长",
        },
        {
            "chunk_id": "b:0",
            "source_id": "b",
            "company": "yh",
            "page": 1,
            "text": "永辉超市门店调改",
        },
    ]
    return rows_to_documents(rows)


def _store() -> InMemoryVectorStore:
    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=16))
    docs = _docs()
    store.add_documents(docs, ids=[str(d.id) for d in docs])
    return store


def test_jieba_tokenize_segments_chinese() -> None:
    toks = jieba_tokenize("良品铺子营业收入下降")
    assert "营业收入" in toks or ("营业" in toks and "收入" in toks)
    assert all(t.strip() for t in toks)


def test_rows_to_documents_contract() -> None:
    docs = _docs()
    assert docs[0].id == "a:0"
    assert docs[0].metadata == {"source_id": "a", "company": "lpz", "page": 1, "chunk_id": "a:0"}


def test_bm25_ranks_term_match_first() -> None:
    top = build_bm25(_docs(), k=1).invoke("永辉超市 调改")
    assert top[0].metadata["chunk_id"] == "b:0"


def test_vector_store_filter_and_dump_load(tmp_path: Path) -> None:
    store = _store()
    lpz_only = vector_retriever(store, "lpz", k=3).invoke("随便查点什么")
    assert lpz_only
    assert all(d.metadata["company"] == "lpz" for d in lpz_only)

    path = tmp_path / "v.json"
    store.dump(str(path))
    restored = InMemoryVectorStore.load(str(path), DeterministicFakeEmbedding(size=16))
    again = vector_retriever(restored, "lpz", k=3).invoke("随便查点什么")
    assert [d.id for d in again] == [d.id for d in lpz_only]


def test_hybrid_fuses_and_dedups() -> None:
    docs = _docs()
    hybrid = build_hybrid(build_bm25(docs, k=2), vector_retriever(_store(), None, 2))
    out = hybrid.invoke("良品铺子 收入")
    assert out
    assert all(isinstance(d, Document) for d in out)
    ids = [d.id for d in out]
    assert len(ids) == len(set(ids))
