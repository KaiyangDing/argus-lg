"""端到端评测：金标要点覆盖 / 陷阱泄漏 / 引用质量 / 忠实度抽样。

承 argus 防作弊设计（金标先行 / 输入白名单 / quote 在场校验 / 陷阱要点），judge 换
LC 原生 structured output——本仓实验点：旧仓在 judge 任务上被 JSON 转义炸出行协议，
这里改走工具调用参数通道重考一次。

指标口径：
- keypoint_coverage：judge 只见 报告+要点（不见语料）；判覆盖必须逐字摘录报告原句，
  程序验在场（归一化子串），摘句不在场 → 计 quote_errors 且不算覆盖（宁严勿放）。
- trap_leaks：报告把陷阱说法当事实陈述才算泄漏（明确否认/驳斥不算）；同样 quote 校验，
  摘句不在场 → 不认定泄漏但计 quote_errors。
- 引用质标（纯正则零 LLM）：citation_rate（论断句带 [n] 比例）/ invalid_refs（编号越界）
  / agg_sentences（>3 引用的聚合句计数——S4 撞墙记录③的仪表化）。
- faithfulness：只在 1~3 引用句中确定性等距抽样 ≤6 句/case；judge 只见 句子+被引证据；
  「证据不足/未披露」类否定性论断，证据确实缺该信息即算支持（承 argus 口径教训）。
"""

import re
from collections.abc import Callable
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from argus_lg.presence import normalize

MAX_FAITH_SAMPLES = 6
AGG_REF_THRESHOLD = 3  # 句末引用超过此数视为聚合句：不计入忠实度抽样池，单独计数
MIN_SENTENCE_CHARS = 8

_REF_RE = re.compile(r"\[(\d+)\]")


class CoverageVerdict(BaseModel):
    covered: bool = Field(description="报告是否覆盖该要点所述事实（含义等价即可）")
    quote: str = Field(
        default="", description="若覆盖：逐字摘录报告中支持该要点的一句原文；否则留空"
    )


class TrapVerdict(BaseModel):
    asserted: bool = Field(description="报告是否把该说法当作事实陈述（明确否认/驳斥不算）")
    quote: str = Field(default="", description="若陈述：逐字摘录该句；否则留空")


class FaithVerdict(BaseModel):
    supported: bool = Field(description="句子中的事实是否被所引证据支持")


JudgeFn = Callable[[type[BaseModel], object], BaseModel]  # (verdict 模型类, prompt 消息) -> 实例


COVERAGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ("你是评测裁判。只依据给定报告判断，禁止使用任何外部知识或你自己对该公司的了解。"),
        ),
        (
            "human",
            (
                "报告全文：\n{report}\n\n待判定要点：{keypoint}\n\n"
                "判断报告是否覆盖了该要点所述的事实（含义等价即可，不要求措辞相同；"
                "数字经四舍五入或单位换算视为等价）。若覆盖，quote 必须逐字摘录报告中"
                "支持该要点的一句原文，一字不改。"
            ),
        ),
    ]
)

TRAP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ("你是评测裁判。只依据给定报告判断，禁止使用任何外部知识或你自己对该公司的了解。"),
        ),
        (
            "human",
            (
                "报告全文：\n{report}\n\n待判定说法：{keypoint}\n\n"
                "该说法是错误信息。判断报告是否把它当作事实陈述了"
                "（报告明确否认、驳斥或说明其不成立的，不算陈述）。"
                "若陈述了，quote 逐字摘录该句，一字不改。"
            ),
        ),
    ]
)

FAITH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是忠实度裁判。只依据给定证据判断句子，禁止外部知识。"
                "注意：若句子是「证据不足/未披露/无法确认」类否定性论断，"
                "只要证据确实缺少该信息，就算被支持。"
            ),
        ),
        ("human", "句子：{sentence}\n\n该句所引证据：\n{evidence}\n\n句子中的事实是否被证据支持？"),
    ]
)


class KpOutcome(TypedDict):
    kp_id: str
    covered: bool
    quote_ok: bool


class CaseScore(TypedDict):
    case_id: str
    covered: int
    musts: int
    kp_details: list[KpOutcome]
    trap_leaks: int
    traps: int
    quote_errors: int
    sentences: int
    cited: int
    citation_rate: float
    invalid_refs: int
    agg_sentences: int
    faith_supported: int
    faith_sampled: int


def split_sentences(report: str) -> list[str]:
    """标题行剔除，按句读切，短碎（<8 字）丢弃。"""
    sentences: list[str] = []
    for raw_line in report.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for part in re.split(r"(?<=[。！？])", line):
            part = part.strip(" \t-　")
            if len(part) >= MIN_SENTENCE_CHARS:
                sentences.append(part)
    return sentences


def extract_refs(sentence: str) -> list[int]:
    return [int(m) for m in _REF_RE.findall(sentence)]


def verify_quote(quote: str, report: str) -> bool:
    return bool(quote.strip()) and normalize(quote) in normalize(report)


def citation_stats(report: str, evidence_count: int) -> dict[str, object]:
    sentences = split_sentences(report)
    cited = 0
    invalid = 0
    agg = 0
    for s in sentences:
        refs = extract_refs(s)
        if refs:
            cited += 1
        if len(refs) > AGG_REF_THRESHOLD:
            agg += 1
        invalid += sum(1 for r in refs if r < 1 or r > evidence_count)
    rate = round(cited / len(sentences), 3) if sentences else 0.0
    return {
        "sentences": len(sentences),
        "cited": cited,
        "citation_rate": rate,
        "invalid_refs": invalid,
        "agg_sentences": agg,
    }


def sample_faith_sentences(report: str, max_n: int = MAX_FAITH_SAMPLES) -> list[str]:
    """1~3 引用句为抽样池（聚合句与无引用句排除），确定性等距抽样。"""
    pool = [s for s in split_sentences(report) if 1 <= len(extract_refs(s)) <= AGG_REF_THRESHOLD]
    if len(pool) <= max_n:
        return pool
    if max_n == 1:
        return [pool[0]]
    idx = sorted({round(i * (len(pool) - 1) / (max_n - 1)) for i in range(max_n)})
    return [pool[i] for i in idx]


def eval_case(case: dict, report: str, evidence: list[dict], judge: JudgeFn) -> CaseScore:
    musts = [kp for kp in case["keypoints"] if kp["kind"] == "must"]
    traps = [kp for kp in case["keypoints"] if kp["kind"] == "trap"]

    quote_errors = 0
    kp_details: list[KpOutcome] = []
    covered = 0
    for kp in musts:
        msgs = COVERAGE_PROMPT.invoke({"report": report, "keypoint": kp["text"]})
        v = judge(CoverageVerdict, msgs)
        quote_ok = verify_quote(v.quote, report)
        ok = bool(v.covered and quote_ok)
        if v.covered and not quote_ok:
            quote_errors += 1  # judge 判覆盖但摘句不在场：降级不算覆盖
        covered += int(ok)
        kp_details.append({"kp_id": kp["kp_id"], "covered": ok, "quote_ok": quote_ok})

    trap_leaks = 0
    for kp in traps:
        msgs = TRAP_PROMPT.invoke({"report": report, "keypoint": kp["text"]})
        t = judge(TrapVerdict, msgs)
        if t.asserted:
            if verify_quote(t.quote, report):
                trap_leaks += 1
            else:
                quote_errors += 1  # 摘句不在场：不认定泄漏

    faith_supported = 0
    samples = sample_faith_sentences(report)
    judged = 0
    for sentence in samples:
        refs = [r for r in extract_refs(sentence) if 1 <= r <= len(evidence)]
        if not refs:
            continue
        ev_lines = "\n".join(
            f"[{r}] ({evidence[r - 1]['source_id']} p{evidence[r - 1]['page']}"
            f"{' · ' + str(evidence[r - 1].get('section', '')) if evidence[r - 1].get('section') else ''}) "
            f"{evidence[r - 1]['text']}"
            for r in refs
        )
        msgs = FAITH_PROMPT.invoke({"sentence": sentence, "evidence": ev_lines})
        f = judge(FaithVerdict, msgs)
        judged += 1
        faith_supported += int(f.supported)

    stats = citation_stats(report, len(evidence))
    return {
        "case_id": case["case_id"],
        "covered": covered,
        "musts": len(musts),
        "kp_details": kp_details,
        "trap_leaks": trap_leaks,
        "traps": len(traps),
        "quote_errors": quote_errors,
        "sentences": int(stats["sentences"]),
        "cited": int(stats["cited"]),
        "citation_rate": float(stats["citation_rate"]),
        "invalid_refs": int(stats["invalid_refs"]),
        "agg_sentences": int(stats["agg_sentences"]),
        "faith_supported": faith_supported,
        "faith_sampled": judged,
    }


def aggregate(scores: list[CaseScore]) -> dict[str, object]:
    covered = sum(s["covered"] for s in scores)
    musts = sum(s["musts"] for s in scores)
    sampled = sum(s["faith_sampled"] for s in scores)
    supported = sum(s["faith_supported"] for s in scores)
    sentences = sum(s["sentences"] for s in scores)
    cited = sum(s["cited"] for s in scores)
    return {
        "coverage": f"{covered}/{musts}",
        "trap_leaks": sum(s["trap_leaks"] for s in scores),
        "quote_errors": sum(s["quote_errors"] for s in scores),
        "citation_rate": round(cited / sentences, 3) if sentences else 0.0,
        "invalid_refs": sum(s["invalid_refs"] for s in scores),
        "agg_sentences": sum(s["agg_sentences"] for s in scores),
        "faithfulness": f"{supported}/{sampled}",
    }
