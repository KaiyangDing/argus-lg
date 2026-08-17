"""研究图：supervisor → Send 并行 researcher → merge → write（引用报告）。

设计承自 argus S2 的图思想（代码零复制）：
- supervisor 结构化拆解 2~5 个研究方面，aspect_id 由代码按序钉 r1..rN（不信 LLM 编号）；
- fan_out 条件边返回 Send 列表，researcher 单节点并行执行（查询生成→混合检索→小结），
  findings 经 operator.add reducer 汇入父状态，避免并行分支撞 LastValue 通道；
- merge 按 aspect_id 排序 + 跨方面证据按 chunk_id 去重 + 全局编号；
- write 只见分节小结与编号证据表，产出 [n] 引用报告。

检索配置承 S3 曲线结论：每查询 k=12（K_PER_QUERY）、每方面 2~3 查询取并集、
并集帽 16 条（EVIDENCE_PER_ASPECT）；深池在 SearchFn 内部（retrieval.make_hybrid_search）。
结构化出口经 struct_factory 注入（默认 chat.with_structured_output），测试用脚本替身零网络。
零证据方面不调小结 LLM（省钱防幻觉，承 argus 教训）。
"""

import operator
from collections.abc import Callable, Sequence
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

SearchFn = Callable[[str, str, int], list[Document]]  # (query, company_slug, k) -> docs

K_PER_QUERY = 12
EVIDENCE_PER_ASPECT = 16
MIN_ASPECTS = 2
MAX_ASPECTS = 5


class AspectSpec(BaseModel):
    name: str = Field(description="方面名，如：财务表现")
    focus: str = Field(description="该方面要回答的具体问题，一句话")


class AspectPlan(BaseModel):
    aspects: list[AspectSpec] = Field(description="2~5 个研究方面")


class QueryList(BaseModel):
    queries: list[str] = Field(description="2~3 条中文检索查询")


class Aspect(TypedDict):
    aspect_id: str
    name: str
    focus: str


class EvidenceRef(TypedDict):
    chunk_id: str
    source_id: str
    page: int
    text: str


class Finding(TypedDict):
    aspect_id: str
    name: str
    summary: str
    evidence: list[EvidenceRef]


class Section(TypedDict):
    aspect_id: str
    name: str
    summary: str
    refs: list[int]


class ResearchState(TypedDict, total=False):
    company: str  # 中文名，进 prompt
    slug: str  # 语料 company 域（lpz/yh/szss），进检索
    aspects: list[Aspect]
    findings: Annotated[list[Finding], operator.add]
    evidence: list[EvidenceRef]
    sections: list[Section]
    report: str


class ResearcherInput(TypedDict):
    company: str
    slug: str
    aspect: Aspect


SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是企业尽调研究主管，只做研究拆解，不回答内容。"),
        (
            "human",
            (
                "研究对象：{company}（A 股上市公司）。把一份企业尽调研究拆解为 2~5 个研究方面，"
                "在财务表现、重大事件与舆情、股权与治理、战略与经营等维度中选最有信息量的组合；"
                "每个方面给出 name（方面名）与 focus（该方面要回答的具体问题，一句话）。"
            ),
        ),
    ]
)

QUERIES_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你为检索系统生成查询，只输出查询，不回答问题。"),
        (
            "human",
            (
                "研究对象：{company}。研究方面：{name}（关注：{focus}）。"
                "生成 2~3 条中文检索查询，角度互补；涉财务的方面至少一条包含具体指标词"
                "（如 营业收入、净利润、同比增减）。"
            ),
        ),
    ]
)

SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是尽调研究员。只依据给定证据写作，禁止使用证据之外的知识；"
                "数字必须与证据逐字一致；证据不足就直说。"
            ),
        ),
        (
            "human",
            "方面：{name}（关注：{focus}）\n\n证据列表：\n{evidence}\n\n据此写 3~6 句中文小结。",
        ),
    ]
)

WRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是尽调报告写手。只使用给定证据；每个事实性论断在其句末标注证据编号（如 [3]，"
                "可多个 [1][4]）；编号必须存在于证据表；证据之外的内容不得出现。"
                "引用编号只允许出现在论断句的句末——禁止在章节标题、段落开头或任何位置聚合罗列编号。"
            ),
        ),
        (
            "human",
            (
                "公司：{company}\n\n各方面小结：\n{sections}\n\n"
                "全局证据表（[编号] (来源 p页码) 内容）：\n{evidence}\n\n"
                "写一份 markdown 尽调报告：一级标题《{company} 尽调报告》，按方面分节，"
                "结尾加「证据不足与边界」小节如实说明缺口。"
            ),
        ),
    ]
)


def assign_aspect_ids(plan: AspectPlan) -> list[Aspect]:
    specs = plan.aspects[:MAX_ASPECTS]
    if len(specs) < MIN_ASPECTS:
        raise ValueError(f"supervisor 拆解不足 {MIN_ASPECTS} 个方面：{len(specs)}")
    return [
        {"aspect_id": f"r{i}", "name": s.name, "focus": s.focus}
        for i, s in enumerate(specs, start=1)
    ]


def union_evidence(results_per_query: Sequence[Sequence[Document]], cap: int) -> list[EvidenceRef]:
    """多查询结果并集：按查询序与各自排名序去重合并，帽内截断。"""
    seen: set[str] = set()
    out: list[EvidenceRef] = []
    for results in results_per_query:
        for doc in results:
            cid = str(doc.metadata["chunk_id"])
            if cid in seen:
                continue
            seen.add(cid)
            out.append(
                {
                    "chunk_id": cid,
                    "source_id": str(doc.metadata["source_id"]),
                    "page": int(doc.metadata["page"]),
                    "text": doc.page_content,
                }
            )
    return out[:cap]


def merge_findings(findings: list[Finding]) -> tuple[list[EvidenceRef], list[Section]]:
    """按 aspect_id 排序（并行到达序不定），跨方面按 chunk_id 去重并全局编号。"""
    ordered = sorted(findings, key=lambda f: f["aspect_id"])
    numbers: dict[str, int] = {}
    evidence: list[EvidenceRef] = []
    sections: list[Section] = []
    for f in ordered:
        refs: list[int] = []
        for ev in f["evidence"]:
            cid = ev["chunk_id"]
            if cid not in numbers:
                evidence.append(ev)
                numbers[cid] = len(evidence)
            refs.append(numbers[cid])
        sections.append(
            {
                "aspect_id": f["aspect_id"],
                "name": f["name"],
                "summary": f["summary"],
                "refs": refs,
            }
        )
    return evidence, sections


def render_evidence_lines(evidence: Sequence[EvidenceRef], start: int = 1) -> str:
    return "\n".join(
        f"[{i}] ({ev['source_id']} p{ev['page']}) {ev['text']}"
        for i, ev in enumerate(evidence, start=start)
    )


def render_sections(sections: Sequence[Section]) -> str:
    # 候选编号放独立提示行、不进标题——放进标题会被 writer 模仿成"章节级聚合引用"
    # （首次 lpz 真跑实测：r1 标题罗列 [1]..[55]，句末反而无引用）。
    parts = []
    for s in sections:
        refs = "".join(f"[{n}]" for n in s["refs"]) or "（无）"
        parts.append(f"### {s['name']}\n{s['summary']}\n（本方面候选证据编号：{refs}）")
    return "\n\n".join(parts)


def build_graph(
    chat: BaseChatModel,
    search: SearchFn,
    struct_factory: Callable[[type[BaseModel]], object] | None = None,
) -> StateGraph:
    def _struct(model_cls: type[BaseModel]) -> object:
        if struct_factory is not None:
            return struct_factory(model_cls)
        return chat.with_structured_output(model_cls)

    def supervisor(state: ResearchState) -> dict[str, object]:
        msgs = SUPERVISOR_PROMPT.invoke({"company": state["company"]})
        plan = _struct(AspectPlan).invoke(msgs)
        return {"aspects": assign_aspect_ids(plan)}

    def fan_out(state: ResearchState) -> list[Send]:
        return [
            Send("researcher", {"company": state["company"], "slug": state["slug"], "aspect": a})
            for a in state["aspects"]
        ]

    def researcher(state: ResearcherInput) -> dict[str, object]:
        aspect = state["aspect"]
        msgs = QUERIES_PROMPT.invoke(
            {
                "company": state["company"],
                "name": aspect["name"],
                "focus": aspect["focus"],
            }
        )
        queries = _struct(QueryList).invoke(msgs).queries[:3]
        results = [search(q, state["slug"], K_PER_QUERY) for q in queries]
        evidence = union_evidence(results, EVIDENCE_PER_ASPECT)
        if not evidence:
            summary = "证据不足：检索未命中相关语料。"
        else:
            msgs = SUMMARIZE_PROMPT.invoke(
                {
                    "name": aspect["name"],
                    "focus": aspect["focus"],
                    "evidence": render_evidence_lines(evidence),
                }
            )
            summary = str(chat.invoke(msgs).content)
        finding: Finding = {
            "aspect_id": aspect["aspect_id"],
            "name": aspect["name"],
            "summary": summary,
            "evidence": evidence,
        }
        return {"findings": [finding]}

    def merge(state: ResearchState) -> dict[str, object]:
        evidence, sections = merge_findings(state["findings"])
        return {"evidence": evidence, "sections": sections}

    def write(state: ResearchState) -> dict[str, object]:
        msgs = WRITE_PROMPT.invoke(
            {
                "company": state["company"],
                "sections": render_sections(state["sections"]),
                "evidence": render_evidence_lines(state["evidence"]),
            }
        )
        return {"report": str(chat.invoke(msgs).content)}

    g = StateGraph(ResearchState)
    g.add_node("supervisor", supervisor)
    g.add_node("researcher", researcher)
    g.add_node("merge", merge)
    g.add_node("write", write)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", fan_out, ["researcher"])
    g.add_edge("researcher", "merge")
    g.add_edge("merge", "write")
    g.add_edge("write", END)
    return g
