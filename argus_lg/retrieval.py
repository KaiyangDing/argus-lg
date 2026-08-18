"""检索层：BM25(jieba 分词) + 向量(InMemoryVectorStore) + EnsembleRetriever 混合。

argus 停在门口的混合检索，LC 生态开箱组合：
- BM25Retriever（community / rank_bm25 内核）：preprocess_func 必须挂 jieba——
  默认按空白切词，中文整句成一个 token，检索直接退化（社区件中文隐坑）。
- InMemoryVectorStore（core / numpy 余弦）：5134 块 ×1024 维内存足够；dump/load 落盘。
- EnsembleRetriever（classic / 加权 RRF 融合）。

embedding 沿 S1 拍板同路（撞墙与替换记录①）：OpenAIEmbeddings + 阿里兼容端点。
check_embedding_ctx_length=False 必设——OpenAI 集成默认用 tiktoken 预编码按 token
数组送 /embeddings，非 OpenAI 端点不认这种输入。
"""

import os
from collections.abc import Callable

import jieba
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from argus_lg.llm import DASHSCOPE_COMPAT_BASE, EMBED_MODEL

EMBED_BATCH = 10  # 百炼 embedding 端点单请求条数上限
HYBRID_POOL = 200  # S3 结论：深池检索再切 top-k，避免 RRF 稀释单路命中


def jieba_tokenize(text: str) -> list[str]:
    return [tok for tok in jieba.lcut(text) if tok.strip()]


def rows_to_documents(rows: list[dict]) -> list[Document]:
    """documents.jsonl 行 → LC Document（id=chunk_id，metadata 取检索所需子集）。"""
    return [
        Document(
            id=str(row["chunk_id"]),
            page_content=str(row["text"]),
            metadata={k: row[k] for k in ("source_id", "company", "page", "chunk_id")},
        )
        for row in rows
    ]


def make_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        base_url=DASHSCOPE_COMPAT_BASE,
        api_key=os.environ["DASHSCOPE_API_KEY"],
        check_embedding_ctx_length=False,
        chunk_size=EMBED_BATCH,
    )


def build_bm25(docs: list[Document], k: int) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(docs, preprocess_func=jieba_tokenize)
    retriever.k = k
    return retriever


def _company_filter(company: str) -> Callable[[Document], bool]:
    def keep(doc: Document) -> bool:
        return doc.metadata.get("company") == company

    return keep


def vector_retriever(
    store: InMemoryVectorStore, company: str | None, k: int
) -> VectorStoreRetriever:
    kwargs: dict[str, object] = {"k": k}
    if company is not None:
        kwargs["filter"] = _company_filter(company)
    return store.as_retriever(search_kwargs=kwargs)


def build_hybrid(
    bm25: BM25Retriever, vector: BaseRetriever, weights: tuple[float, float] = (0.5, 0.5)
) -> EnsembleRetriever:
    return EnsembleRetriever(retrievers=[bm25, vector], weights=list(weights))


def make_hybrid_search(
    rows: list[dict], store: InMemoryVectorStore, pool: int = HYBRID_POOL
) -> Callable[[str, str, int], list[Document]]:
    """图用 SearchFn：(query, company_slug, k) -> 深池混合排序切 top-k。

    BM25 索引每公司懒建一次（jieba 分词是大头）；向量与融合按查询现组（轻量）。
    """
    docs = rows_to_documents(rows)
    by_company: dict[str, list[Document]] = {}
    for d in docs:
        by_company.setdefault(str(d.metadata["company"]), []).append(d)
    bm25_cache: dict[str, BM25Retriever] = {}
    # 章节面包屑来自最新 documents.jsonl；向量库 dump 的 metadata 是旧快照，
    # 检索结果统一回贴 section（免重嵌的关键：面包屑不进 embedding 文本）
    section_map = {str(r["chunk_id"]): str(r.get("section", "")) for r in rows}

    def search(query: str, company: str, k: int) -> list[Document]:
        if company not in by_company:
            raise ValueError(f"未知公司域：{company}（可选 {sorted(by_company)}）")
        if company not in bm25_cache:
            bm25_cache[company] = build_bm25(by_company[company], pool)
        bm25 = bm25_cache[company]
        bm25.k = pool
        hybrid = build_hybrid(bm25, vector_retriever(store, company, pool))
        out = hybrid.invoke(query)[:k]
        for doc in out:
            cid = str(doc.metadata.get("chunk_id", ""))
            doc.metadata["section"] = section_map.get(cid, doc.metadata.get("section", ""))
        return out

    return search
