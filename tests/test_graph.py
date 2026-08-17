"""研究图：纯函数面 + 全 Fake 端到端（零网络零成本）。

结构化出口经 scripted_struct_factory 注入脚本答案；自由文本走 FakeListChatModel；
检索走合成 SearchFn。真实模型行为由 scripts/run_graph.py 人工冒烟验收（PLAN 设计题 2）。
"""

from collections import deque

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel
from pydantic import BaseModel

from argus_lg.graph import (
    AspectPlan,
    AspectSpec,
    QueryList,
    assign_aspect_ids,
    build_graph,
    merge_findings,
    render_evidence_lines,
    union_evidence,
)


def scripted_struct_factory(script: dict[type[BaseModel], list[BaseModel]]):
    """按模型类排队吐预设实例；跨节点调用共享队列。"""
    queues = {cls: deque(items) for cls, items in script.items()}

    class Scripted:
        def __init__(self, cls: type[BaseModel]) -> None:
            self._cls = cls

        def invoke(self, _input: object) -> BaseModel:
            return queues[self._cls].popleft()

    return Scripted


def _doc(cid: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={"chunk_id": cid, "source_id": "src", "page": 3, "company": "t"},
    )


def test_assign_aspect_ids_clamps_and_numbers() -> None:
    plan = AspectPlan(aspects=[AspectSpec(name=f"a{i}", focus="f") for i in range(7)])
    aspects = assign_aspect_ids(plan)
    assert [a["aspect_id"] for a in aspects] == ["r1", "r2", "r3", "r4", "r5"]

    with pytest.raises(ValueError, match="不足"):
        assign_aspect_ids(AspectPlan(aspects=[AspectSpec(name="唯一", focus="f")]))


def test_union_evidence_dedups_and_caps() -> None:
    q1 = [_doc("c1", "一"), _doc("c2", "二")]
    q2 = [_doc("c2", "二"), _doc("c3", "三"), _doc("c4", "四")]
    out = union_evidence([q1, q2], cap=3)
    assert [e["chunk_id"] for e in out] == ["c1", "c2", "c3"]  # 去重保序 + 帽截断


def test_merge_findings_orders_dedups_numbers() -> None:
    f_r2 = {
        "aspect_id": "r2",
        "name": "乙",
        "summary": "s2",
        "evidence": [
            {"chunk_id": "c9", "source_id": "s", "page": 1, "text": "九"},
            {"chunk_id": "c1", "source_id": "s", "page": 1, "text": "一"},
        ],
    }
    f_r1 = {
        "aspect_id": "r1",
        "name": "甲",
        "summary": "s1",
        "evidence": [
            {"chunk_id": "c1", "source_id": "s", "page": 1, "text": "一"},
        ],
    }
    evidence, sections = merge_findings([f_r2, f_r1])  # 乱序输入
    assert [s["aspect_id"] for s in sections] == ["r1", "r2"]
    assert [e["chunk_id"] for e in evidence] == ["c1", "c9"]  # r1 先编号，跨方面去重
    assert sections[0]["refs"] == [1]
    assert sections[1]["refs"] == [2, 1]


def test_render_evidence_lines_format() -> None:
    lines = render_evidence_lines(
        [{"chunk_id": "c", "source_id": "lpz-ar-2024", "page": 8, "text": "营收"}]
    )
    assert lines == "[1] (lpz-ar-2024 p8) 营收"


def test_graph_end_to_end_with_fakes() -> None:
    chat = FakeListChatModel(
        responses=["小结甲。", "小结乙。", "# 测试公司 尽调报告\n营收下降 [1]。"]
    )
    factory = scripted_struct_factory(
        {
            AspectPlan: [
                AspectPlan(
                    aspects=[
                        AspectSpec(name="财务", focus="营收变化"),
                        AspectSpec(name="事件", focus="重大舆情"),
                    ]
                )
            ],
            QueryList: [QueryList(queries=["查A1", "查A2"]), QueryList(queries=["查B1"])],
        }
    )

    def search(query: str, slug: str, k: int) -> list[Document]:
        assert slug == "t"
        assert k == 12
        return [_doc(f"c-{query}", f"{query} 的证据")]

    graph = build_graph(chat, search, struct_factory=factory).compile()
    out = graph.invoke({"company": "测试公司", "slug": "t"})

    assert [a["aspect_id"] for a in out["aspects"]] == ["r1", "r2"]
    assert len(out["findings"]) == 2
    assert {f["summary"] for f in out["findings"]} == {"小结甲。", "小结乙。"}
    assert len(out["evidence"]) == 3  # 3 条查询 3 个不同 chunk，全局编号 1..3
    assert [s["aspect_id"] for s in out["sections"]] == ["r1", "r2"]
    assert out["report"].startswith("# 测试公司 尽调报告")


def test_graph_zero_evidence_skips_summarize_llm() -> None:
    chat = FakeListChatModel(responses=["# 空报告"])  # 只留 write 一条，summarize 不许消费
    factory = scripted_struct_factory(
        {
            AspectPlan: [
                AspectPlan(
                    aspects=[
                        AspectSpec(name="甲", focus="f"),
                        AspectSpec(name="乙", focus="f"),
                    ]
                )
            ],
            QueryList: [QueryList(queries=["空1"]), QueryList(queries=["空2"])],
        }
    )

    graph = build_graph(chat, lambda q, s, k: [], struct_factory=factory).compile()
    out = graph.invoke({"company": "测试公司", "slug": "t"})

    assert all("证据不足" in f["summary"] for f in out["findings"])
    assert out["evidence"] == []
    assert out["report"] == "# 空报告"  # write 拿到的正是唯一剩下的 Fake 响应
