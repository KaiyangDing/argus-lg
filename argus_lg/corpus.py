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
import re
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


# 章节头模式：编号标题（一、/5、/（1）…）与报表名（合并/母公司×四表）——
# iteration-2 五连语义错配的根因是块不带表格语境，此处以跨页运行头修复（v02_devlog）
_HEADING_RE = re.compile(
    r"^\s*(?:第[一二三四五六七八九十百]+节|[一二三四五六七八九十]+、|"
    r"（[一二三四五六七八九十]+）|\([一二三四五六七八九十]+\)|"
    r"\d{1,2}、|（\d{1,2}）\.?|\(\d{1,2}\)\.?)\s*\S{2,}"
)
_STATEMENT_RE = re.compile(r"^\s*(合并|母公司)(资产负债表|利润表|现金流量表|所有者权益变动表)")
_TABLE_ROW_RE = re.compile(r"\d{1,3}(?:,\d{3})+")  # 含千分位数字的行是表行不是标题


def annotate_page_sections(pages: list) -> None:
    """跨页运行章节头：每页 section=进入本页前最近的标题（表头在前页、数字在后页的场景）。

    面包屑只进 metadata 与证据渲染，不改块文本——embedding/BM25/在场锚全部零影响，免重嵌。
    """
    major = ""
    minor = ""
    for page in pages:
        page.metadata["section"] = f"{major} / {minor}" if major and minor else major or minor
        for line in page.page_content.splitlines():
            stripped = line.strip()
            if _STATEMENT_RE.match(stripped):
                major, minor = stripped[:40], ""
            elif _HEADING_RE.match(stripped) and not _TABLE_ROW_RE.search(stripped):
                if stripped.startswith(("（", "(")):
                    minor = stripped[:40]  # 次级标题不覆盖主标题（5、应收账款 / (1).按账龄披露）
                else:
                    major, minor = stripped[:40], ""


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


_YEAR_RE = re.compile(r"(20\d{2})")
_KIND_LABELS = {"ar": "年报", "news": "新闻", "notice": "公告"}


def corpus_profile(rows: list[dict[str, object]], company: str) -> str:
    """按语料实况生成公司文档概况，供 prompt 注入。

    产品场景语料由用户任意上传，年份/文档类型不可预设——时间锚等先验一律从这里
    数据驱动生成，不写死在 prompt 里（v0.2 工程问题账「硬编码年份」条）。
    """
    per_source: dict[str, int] = {}
    for row in rows:
        if row.get("company") == company:
            sid = str(row["source_id"])
            per_source[sid] = per_source.get(sid, 0) + 1
    if not per_source:
        return "该公司暂无语料。"
    years: set[str] = set()
    entries: list[str] = []
    for sid in sorted(per_source):
        found = _YEAR_RE.findall(sid)
        years.update(found)
        parts = sid.split("-")
        kind = _KIND_LABELS.get(parts[1], "文档") if len(parts) > 1 else "文档"
        year_tag = f"，{found[0]}" if found else ""
        entries.append(f"{sid}（{kind}{year_tag}）")
    year_line = "、".join(sorted(years)) if years else "未知（先以宽泛查询探明时间范围）"
    latest = max(years) if years else "未知"
    return (
        f"可用文档 {len(per_source)} 份：" + "；".join(entries) + f"。"
        f"覆盖年份：{year_line}；最新年份：{latest}。"
    )
