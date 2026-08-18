"""langgraph dev（Studio）入口：加载真语料与向量库，导出编译图。

启动（起服务本身零 LLM 调用；在 Studio 里跑图=真调用，费用与 run_graph 同量级 ≈¥0.09/次）：
    uv run langgraph dev

Studio 打开后 Input 填：{"company": "良品铺子", "slug": "lpz"}（yh/szss 同理）。
checkpointer 不在此编译——langgraph dev 自带持久层（graph 里 MemorySaver 仅 CLI 用）。
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore

from argus_lg.corpus import corpus_profile, read_jsonl
from argus_lg.graph import build_graph
from argus_lg.llm import make_chat
from argus_lg.retrieval import make_embeddings, make_hybrid_search

_REPO = Path(__file__).resolve().parent.parent

load_dotenv(_REPO / ".env")
_rows = read_jsonl(_REPO / "corpus" / "derived" / "documents.jsonl")
_store = InMemoryVectorStore.load(
    str(_REPO / "corpus" / "derived" / "vectors.json"), make_embeddings()
)

graph = build_graph(
    make_chat(),
    make_hybrid_search(_rows, _store),
    profile_fn=lambda s: corpus_profile(_rows, s),
).compile()
