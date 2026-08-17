"""评测器：句切/引用统计/quote 校验/抽样 纯函数面 + eval_case 脚本裁判端到端。零网络。"""

from collections import deque

from pydantic import BaseModel

from argus_lg.eval import (
    CoverageVerdict,
    FaithVerdict,
    TrapVerdict,
    citation_stats,
    eval_case,
    extract_refs,
    sample_faith_sentences,
    split_sentences,
    verify_quote,
)

REPORT = """# 测试报告

## 财务
营收下降百分之十一 [1]。利润由盈转亏并持续承压 [2][3]。
总体来看各方面情况一般 [1][2][3][4][5]。
短句。
这句话没有引用编号但足够长。
"""


def test_split_sentences_skips_headings_and_shorts() -> None:
    sentences = split_sentences(REPORT)
    assert len(sentences) == 4
    assert all(not s.startswith("#") for s in sentences)
    assert all(len(s) >= 8 for s in sentences)


def test_extract_refs() -> None:
    assert extract_refs("利润由盈转亏 [2][3]。") == [2, 3]
    assert extract_refs("无引用句。") == []


def test_citation_stats() -> None:
    stats = citation_stats(REPORT, evidence_count=3)
    assert stats["sentences"] == 4
    assert stats["cited"] == 3
    assert stats["citation_rate"] == 0.75
    assert stats["invalid_refs"] == 2  # [4][5] 越界
    assert stats["agg_sentences"] == 1  # 5 引用聚合句


def test_verify_quote_normalized() -> None:
    assert verify_quote("营收下降 百分之十一", REPORT)  # 空白差异归一化后在场
    assert not verify_quote("这句不在报告里", REPORT)
    assert not verify_quote("  ", REPORT)


def test_sample_faith_pool_excludes_agg_and_uncited() -> None:
    samples = sample_faith_sentences(REPORT)
    assert len(samples) == 2  # 只剩 1~3 引用的两句
    assert all(1 <= len(extract_refs(s)) <= 3 for s in samples)
    long_pool = [f"第{i}句足够长的引用句子 [1]。" for i in range(20)]
    picked = sample_faith_sentences("\n".join(long_pool), max_n=6)
    assert len(picked) == 6
    assert picked == sample_faith_sentences("\n".join(long_pool), max_n=6)  # 确定性


def scripted_judge(script: dict[type[BaseModel], list[BaseModel]]):
    queues = {cls: deque(items) for cls, items in script.items()}

    def judge(model_cls: type[BaseModel], _msgs: object) -> BaseModel:
        return queues[model_cls].popleft()

    return judge


_CASE = {
    "case_id": "t",
    "company": "测试",
    "keypoints": [
        {"kp_id": "m1", "text": "营收下降约 11%", "kind": "must", "source": "s"},
        {"kp_id": "m2", "text": "现金流充裕", "kind": "must", "source": "s"},
        {"kp_id": "t1", "text": "营收大幅增长", "kind": "trap"},
    ],
}

_EVIDENCE = [
    {"chunk_id": f"c{i}", "source_id": "src", "page": i, "text": f"证据{i}"} for i in (1, 2, 3)
]


def test_eval_case_full_flow() -> None:
    judge = scripted_judge(
        {
            CoverageVerdict: [
                CoverageVerdict(covered=True, quote="营收下降百分之十一"),
                CoverageVerdict(covered=True, quote="这句不在报告里"),  # 摘句不在场→降级
            ],
            TrapVerdict: [TrapVerdict(asserted=False)],
            FaithVerdict: [FaithVerdict(supported=True), FaithVerdict(supported=False)],
        }
    )
    score = eval_case(_CASE, REPORT, _EVIDENCE, judge)
    assert score["covered"] == 1
    assert score["musts"] == 2
    assert score["quote_errors"] == 1
    assert score["trap_leaks"] == 0
    assert score["faith_sampled"] == 2
    assert score["faith_supported"] == 1
    assert score["agg_sentences"] == 1


def test_eval_case_trap_requires_verified_quote() -> None:
    base = {
        CoverageVerdict: [
            CoverageVerdict(covered=False),
            CoverageVerdict(covered=False),
        ],
        FaithVerdict: [FaithVerdict(supported=True), FaithVerdict(supported=True)],
    }
    leak = eval_case(
        _CASE,
        REPORT,
        _EVIDENCE,
        scripted_judge(
            {**base, TrapVerdict: [TrapVerdict(asserted=True, quote="总体来看各方面情况一般")]}
        ),
    )
    assert leak["trap_leaks"] == 1

    fabricated = eval_case(
        _CASE,
        REPORT,
        _EVIDENCE,
        scripted_judge({**base, TrapVerdict: [TrapVerdict(asserted=True, quote="编造的句子")]}),
    )
    assert fabricated["trap_leaks"] == 0
    assert fabricated["quote_errors"] == 1
