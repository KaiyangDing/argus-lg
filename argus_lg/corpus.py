"""语料摄取：corpus/raw 的 PDF → LC 文档 → 切块 → documents.jsonl。

LC 原生件：PyPDFLoader（页粒度）+ RecursiveCharacterTextSplitter（中文分隔符）。
chunk 契约（S3 检索与金标在场检查共用）：
    source_id  文件名去后缀（lpz-ar-2024）
    company    source_id 首段（lpz/yh/szss）
    page       1 起页码（跨页块记起始页）
    seq        文档内块序号（0 起）
    chunk_id   "{source_id}:{seq}"
    text       块文本
"""

import hashlib
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

COMPANIES = {"lpz": "良品铺子", "yh": "永辉超市", "szss": "三只松鼠"}

# 中文财报语料的分隔符优先级：段落 > 换行 > 句读 > 空格 > 硬切
SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(raw_dir: Path) -> dict[str, dict[str, object]]:
    """{source_id: {file, sha256, bytes}}，按 source_id 排序保证确定性。"""
    entries: dict[str, dict[str, object]] = {}
    for pdf in sorted(raw_dir.rglob("*.pdf")):
        source_id = pdf.stem
        if source_id in entries:
            raise ValueError(f"source_id 重复：{source_id}")
        entries[source_id] = {
            "file": pdf.relative_to(raw_dir).as_posix(),
            "sha256": sha256_file(pdf),
            "bytes": pdf.stat().st_size,
        }
    return entries


def load_pdf_pages(path: Path) -> list[Document]:
    """一页一 Document；metadata 统一为本仓契约字段。"""
    source_id = path.stem
    company = source_id.split("-", 1)[0]
    pages = PyPDFLoader(str(path)).load()
    for doc in pages:
        doc.metadata = {
            "source_id": source_id,
            "company": company,
            "page": int(doc.metadata.get("page", 0)) + 1,
        }
    return pages


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        keep_separator="end",
    )


def split_pages(pages: list[Document]) -> list[Document]:
    """切块并补 seq/chunk_id；splitter 自带 metadata 继承。"""
    chunks = make_splitter().split_documents(pages)
    for seq, doc in enumerate(chunks):
        doc.metadata["seq"] = seq
        doc.metadata["chunk_id"] = f"{doc.metadata['source_id']}:{seq}"
    return chunks


def chunks_to_rows(chunks: list[Document]) -> list[dict[str, object]]:
    return [{**doc.metadata, "text": doc.page_content} for doc in chunks]


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
