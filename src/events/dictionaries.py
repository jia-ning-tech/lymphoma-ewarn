from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from ..config import get_cfg
from ..utils.log import get_logger, heartbeat


# --------------------------- Keyword banks --------------------------- #
# 这些关键词针对 MIMIC-IV ICU 的 d_items.label/abbreviation/category 进行匹配
VASOPRESSOR_KEYWORDS: Dict[str, List[str]] = {
    # key = 规范药名；值 = 关键词（不区分大小写）
    "norepinephrine": ["norepinephrine", "noradren", r"\bnorepi\b", r"\bnor-?epi\b"],
    "epinephrine": ["epinephrine", "adrenalin", r"\bepi\b", r"\bepine\b"],
    "vasopressin": ["vasopressin", r"\bvp\b", r"\bvaso\b"],
    "dopamine": ["dopamine", r"\bdopa\b"],
    "phenylephrine": ["phenylephrine", r"\bneo-?synephrine\b", r"\bneo\b", r"\bphenyl(e)?\b"],
}

IMV_KEYWORDS: Dict[str, List[str]] = {
    # 机械通气相关（模式/设备/通道）
    "vent_mode": [
        "vent", "ventilator", "mechanical ventilation", r"\bIMV\b", r"\bSIMV\b",
        "assist control", r"\bAC\b(?!\s*node)", "volume control", "pressure control",
        "pcv", "vcv", "aprv", "bilevel", r"\bpeep\b", r"\bfio?2\b",
        "endotracheal", r"\bett\b", "tracheostomy", "trach", "invasive ventilation",
    ],
}

CRRT_KEYWORDS: Dict[str, List[str]] = {
    # CRRT / 透析相关
    "crrt": [
        r"\bcrrt\b", r"\bcvvh(d|df)?\b", "continuous veno-venous", "prisma", "prismaflex",
        "dialysis", "hemofiltration", "ultrafiltration", r"\bUF\b", "dialysate",
        "replacement fluid", "effluent", "hemodiafiltration",
    ],
}


# --------------------------- Helper functions ------------------------ #

def _compile_keywords(words: Iterable[str]) -> List[re.Pattern]:
    return [re.compile(w, re.IGNORECASE) for w in words]


def _match_any(text: str, patterns: List[re.Pattern]) -> bool:
    if not isinstance(text, str):
        return False
    for p in patterns:
        if p.search(text):
            return True
    return False


def _find_itemids_by_keywords(d_items: pd.DataFrame, banks: Dict[str, List[str]]) -> Dict[str, Set[int]]:
    """
    在 d_items 的 label/abbreviation/category 中按关键词搜索，返回 {name: {itemids}}。
    """
    cols = [c for c in ["label", "abbreviation", "category"] if c in d_items.columns]
    out: Dict[str, Set[int]] = {k: set() for k in banks.keys()}
    if not cols:
        return out

    compiled = {k: _compile_keywords(v) for k, v in banks.items()}
    for _, row in d_items.iterrows():
        txt = " | ".join(str(row[c]) for c in cols if pd.notna(row[c]))
        for name, pats in compiled.items():
            if _match_any(txt, pats):
                try:
                    out[name].add(int(row["itemid"]))
                except Exception:
                    pass
    return out


def _merge_with_manual(auto_map: Dict[str, Set[int]], manual_map: Optional[Dict[str, Iterable[int]]]) -> Dict[str, List[int]]:
    """
    合并自动与手工（手工覆盖/添加）。返回排序后的 list。
    manual_map 示例：
      dictionaries.yaml:
        items:
          vasopressors:
            norepinephrine: [221906, 30047]
    """
    out: Dict[str, Set[int]] = {k: set(v) for k, v in auto_map.items()}
    if manual_map:
        for k, ids in manual_map.items():
            out.setdefault(k, set()).update(int(x) for x in ids)
    # 排序去重
    return {k: sorted({int(x) for x in v}) for k, v in out.items()}


# --------------------------- Public builders ------------------------- #

def build_event_dicts(save_json: bool = True) -> Dict[str, Dict[str, List[int]]]:
    """
    从 icu/d_items.csv 自动构建事件词典，并与 conf/dictionaries.yaml 合并。
    返回结构：
    {
      "vasopressors": {"norepinephrine": [...], "epinephrine": [...], ...},
      "imv": {"vent_mode": [...]},
      "crrt": {"crrt": [...]}
    }
    """
    cfg = get_cfg()
    lg = get_logger("events.dicts")

    d_items_path = os.path.join(cfg.paths.raw_icu, "d_items.csv")
    if not os.path.exists(d_items_path):
        raise FileNotFoundError(f"Missing ICU d_items.csv at: {d_items_path}")

    lg.info("Loading icu/d_items.csv ...")
    with heartbeat(lg, secs=int(cfg.logging.heartbeat_secs), note="load d_items"):
        d_items = pd.read_csv(
            d_items_path,
            sep=",", encoding="utf-8", low_memory=False,
            usecols=[c for c in ["itemid", "label", "abbreviation", "category", "linksto"] if os.path.exists(d_items_path)]
        )
    if "itemid" not in d_items.columns or "label" not in d_items.columns:
        raise RuntimeError("d_items.csv does not contain expected columns: itemid, label, abbreviation, category, linksto")

    # 自动发现
    lg.info("Auto-discovering ITEMIDs by keyword matching...")
    auto_vp = _find_itemids_by_keywords(d_items, VASOPRESSOR_KEYWORDS)
    auto_imv = _find_itemids_by_keywords(d_items, IMV_KEYWORDS)
    auto_crrt = _find_itemids_by_keywords(d_items, CRRT_KEYWORDS)

    # 手工补充（如果用户在 dictionaries.yaml 中提供）
    dicts = cfg.get("dictionaries_content", {}) or {}
    manual_items = dicts.get("items", {})

    vp_manual = (manual_items.get("vasopressors") or {}) if isinstance(manual_items.get("vasopressors"), dict) else {}
    imv_manual = (manual_items.get("imv") or {}) if isinstance(manual_items.get("imv"), dict) else {}
    crrt_manual = (manual_items.get("crrt") or {}) if isinstance(manual_items.get("crrt"), dict) else {}

    vp = _merge_with_manual(auto_vp, vp_manual)
    imv = _merge_with_manual(auto_imv, imv_manual)
    crrt = _merge_with_manual(auto_crrt, crrt_manual)

    result = {
        "vasopressors": vp,
        "imv": imv,
        "crrt": crrt,
    }

    if save_json:
        out_dir = os.path.join(cfg.paths.outputs, "artifacts")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "event_itemids.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        lg.info(f"Event dictionaries saved: {out_path}")

    # 日志摘要
    def _summ(k: str, m: Dict[str, List[int]]) -> str:
        parts = []
        for name, ids in m.items():
            parts.append(f"{name}={len(ids)}")
        return f"{k}: " + ", ".join(parts)

    lg.info(_summ("vasopressors", vp))
    lg.info(_summ("imv", imv))
    lg.info(_summ("crrt", crrt))

    # 打印每类前几个样例（方便人工校验）
    for cat, m in result.items():
        for name, ids in m.items():
            if ids:
                lg.info(f"sample {cat}.{name}: {ids[:10]}")

    return result


def save_event_dicts_json(path: Optional[str] = None) -> str:
    """
    shortcut：仅保存 JSON 到指定位置（默认为 outputs/artifacts/event_itemids.json）
    """
    cfg = get_cfg()
    res = build_event_dicts(save_json=False)
    out_dir = os.path.join(cfg.paths.outputs, "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = path or os.path.join(out_dir, "event_itemids.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return out_path


# ------------------------------- self-test --------------------------- #

if __name__ == "__main__":
    # 轻量自检：加载 d_items，构建并落盘 JSON，打印命中数量
    build_event_dicts(save_json=True)
