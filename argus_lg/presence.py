"""金标在场率：must 要点的字面锚点是否在对应 source 文档的块里找得到。

口径（S2 验收指标，纯确定性零 LLM）：
- 每条 must 要点 = 若干锚点组；组内任一形态命中即组命中（OR），
  所有组各自在该 source 的任一块命中才算要点在场（组间 AND，允许不同块）。
- 匹配 = 归一化子串：小写 + 去全部空白 + 去 ASCII 逗号——对 PDF 抽取的
  空格噪声与千分位断裂免疫；数字锚取全精度长串，天然抗误命中。
- 锚点是数据资产（eval/presence_anchors.json），2026-08-17 从 argus v3
  已知好语料（MinerU 解析，18/18 在场）人工挖掘的真实表面形态。
"""

import json
import re
from pathlib import Path
from typing import TypedDict

_NORM_RE = re.compile(r"[\s,]+")


class Anchor(TypedDict):
    case_id: str
    kp_id: str
    source: str
    groups: list[list[str]]


class KpResult(TypedDict):
    case_id: str
    kp_id: str
    source: str
    hit: bool
    missed_groups: list[list[str]]


def normalize(text: str) -> str:
    return _NORM_RE.sub("", text).lower()


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_anchors(path: Path, cases: list[dict]) -> list[Anchor]:
    """载入并校验：每条 must 要点恰有一条锚点，source 与金标一致。"""
    anchors: list[Anchor] = json.loads(path.read_text(encoding="utf-8"))["anchors"]
    by_key = {(a["case_id"], a["kp_id"]): a for a in anchors}
    if len(by_key) != len(anchors):
        raise ValueError("presence_anchors 存在重复 (case_id, kp_id)")
    for case in cases:
        for kp in case["keypoints"]:
            if kp["kind"] != "must":
                continue
            key = (case["case_id"], kp["kp_id"])
            anchor = by_key.pop(key, None)
            if anchor is None:
                raise ValueError(f"must 要点缺锚点：{key}")
            if anchor["source"] != kp["source"]:
                raise ValueError(f"锚点 source 与金标不符：{key}")
            if not anchor["groups"] or any(not g for g in anchor["groups"]):
                raise ValueError(f"锚点组为空：{key}")
    if by_key:
        raise ValueError(f"锚点指向不存在的要点：{sorted(by_key)}")
    return anchors


def check_kp(anchor: Anchor, norm_texts: list[str]) -> KpResult:
    missed: list[list[str]] = []
    for group in anchor["groups"]:
        forms = [normalize(f) for f in group]
        if not any(form in text for text in norm_texts for form in forms):
            missed.append(group)
    return {
        "case_id": anchor["case_id"],
        "kp_id": anchor["kp_id"],
        "source": anchor["source"],
        "hit": not missed,
        "missed_groups": missed,
    }


def check_presence(anchors: list[Anchor], rows: list[dict]) -> list[KpResult]:
    """rows = documents.jsonl 行（需 source_id/text）；归一化按 source 预计算。"""
    norm_by_source: dict[str, list[str]] = {}
    for row in rows:
        norm_by_source.setdefault(str(row["source_id"]), []).append(normalize(str(row["text"])))
    return [check_kp(a, norm_by_source.get(a["source"], [])) for a in anchors]
