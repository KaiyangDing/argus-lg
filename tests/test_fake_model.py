"""S1 测试骨架：LC 自带测试替身，零网络零成本。

本仓不带 cassette 录放，单测一律用 Fake 系列（PLAN 设计题 2）。
"""

from typing import TypedDict

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph


def test_fake_chat_model_follows_script() -> None:
    llm = FakeListChatModel(responses=["第一答", "第二答"])
    assert llm.invoke("任意输入").content == "第一答"
    assert llm.invoke("任意输入").content == "第二答"


def test_lcel_chain_composes() -> None:
    prompt = ChatPromptTemplate.from_template("向{name}问好")
    llm = FakeListChatModel(responses=["你好，良品铺子。"])
    chain = prompt | llm | StrOutputParser()
    assert chain.invoke({"name": "良品铺子"}) == "你好，良品铺子。"


def test_fake_embedding_is_deterministic() -> None:
    emb = DeterministicFakeEmbedding(size=8)
    v1 = emb.embed_query("同一句话")
    v2 = emb.embed_query("同一句话")
    other = emb.embed_query("另一句话")
    assert len(v1) == 8
    assert v1 == v2
    assert v1 != other


def test_langgraph_minimal_graph() -> None:
    class S(TypedDict):
        text: str

    def shout(state: S) -> dict[str, str]:
        return {"text": state["text"] + "!"}

    g = StateGraph(S)
    g.add_node("shout", shout)
    g.add_edge(START, "shout")
    g.add_edge("shout", END)
    graph = g.compile()

    assert graph.invoke({"text": "argus-lg"})["text"] == "argus-lg!"
