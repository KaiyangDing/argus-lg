"""presence.py：归一化匹配语义 + 锚点数据校验（含仓内真实资产自检）。"""

from pathlib import Path

import pytest

from argus_lg.presence import Anchor, check_kp, check_presence, load_anchors, load_cases, normalize

REPO = Path(__file__).resolve().parent.parent


def test_normalize_strips_space_comma_case() -> None:
    assert normalize("7,159,201,563.03") == "7159201563.03"
    assert normalize(" H 股 ") == "h股"
    assert normalize("232 家\n门店") == "232家门店"


def _anchor(groups: list[list[str]]) -> Anchor:
    return {"case_id": "c", "kp_id": "m1", "source": "s", "groups": groups}


def test_check_kp_or_within_group_and_across_groups() -> None:
    texts = [normalize("营业收入 7,159,201,563.03 元"), normalize("同比 -11.02%")]

    hit = check_kp(_anchor([["不存在的形态", "7,159,201,563"], ["-11.02"]]), texts)
    assert hit["hit"] and hit["missed_groups"] == []  # 组内 OR，组间允许不同块

    miss = check_kp(_anchor([["7,159,201,563"], ["-99.99"]]), texts)
    assert not miss["hit"]
    assert miss["missed_groups"] == [["-99.99"]]


def test_check_presence_scopes_by_source() -> None:
    rows = [
        {"source_id": "s", "text": "长江国贸 21.00%"},
        {"source_id": "other", "text": "国有资产监督管理委员会"},
    ]
    result = check_presence([_anchor([["长江国贸"], ["国有资产监督管理委员会"]])], rows)[0]
    assert not result["hit"]  # 第二组只在别的 source，不算


def _cases(kp_ids: list[str]) -> list[dict]:
    kps = [{"kp_id": k, "text": "t", "kind": "must", "source": "s"} for k in kp_ids]
    kps.append({"kp_id": "t1", "text": "t", "kind": "trap"})
    return [{"case_id": "c", "company": "x", "keypoints": kps}]


def test_load_anchors_validates(tmp_path: Path) -> None:
    import json

    good = {"anchors": [{"case_id": "c", "kp_id": "m1", "source": "s", "groups": [["a"]]}]}
    p = tmp_path / "a.json"
    p.write_text(json.dumps(good), encoding="utf-8")
    assert load_anchors(p, _cases(["m1"]))[0]["kp_id"] == "m1"

    with pytest.raises(ValueError, match="缺锚点"):
        load_anchors(p, _cases(["m1", "m2"]))

    bad_source = {
        "anchors": [{"case_id": "c", "kp_id": "m1", "source": "WRONG", "groups": [["a"]]}]
    }
    p.write_text(json.dumps(bad_source), encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        load_anchors(p, _cases(["m1"]))

    orphan = {
        "anchors": [
            {"case_id": "c", "kp_id": "m1", "source": "s", "groups": [["a"]]},
            {"case_id": "c", "kp_id": "m9", "source": "s", "groups": [["a"]]},
        ]
    }
    p.write_text(json.dumps(orphan), encoding="utf-8")
    with pytest.raises(ValueError, match="不存在的要点"):
        load_anchors(p, _cases(["m1"]))

    empty_group = {
        "anchors": [{"case_id": "c", "kp_id": "m1", "source": "s", "groups": [["a"], []]}]
    }
    p.write_text(json.dumps(empty_group), encoding="utf-8")
    with pytest.raises(ValueError, match="锚点组为空"):
        load_anchors(p, _cases(["m1"]))


def test_repo_assets_are_consistent() -> None:
    """仓内真实金标与锚点互检：18 条 must 全覆盖、source 全一致。"""
    cases = load_cases(REPO / "eval" / "cases.jsonl")
    anchors = load_anchors(REPO / "eval" / "presence_anchors.json", cases)
    assert len(anchors) == 18
