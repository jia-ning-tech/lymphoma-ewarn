#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a small Markdown snippet that lists chosen decision thresholds
from posthoc calibration JSONs, so README does not hardcode numbers.

Input JSONs (if exist):
  outputs/reports/posthoc_calibration_h{H}_{method}.json
Where method in {"isotonic","sigmoid"}, H in {24, 48} (extend as needed).

Output:
  outputs/docs/thresholds_for_readme.md  -- a Markdown table
"""

from __future__ import annotations
from pathlib import Path
import json

REPORTS = Path("outputs/reports")
OUT_DIR = Path("outputs/docs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MD = OUT_DIR / "thresholds_for_readme.md"

def pick_threshold(obj: dict) -> float | None:
    # your JSON had fields like chosen_threshold_val or stay_level.threshold
    if "chosen_threshold_val" in obj and obj["chosen_threshold_val"] is not None:
        return float(obj["chosen_threshold_val"])
    stay = obj.get("stay_level", {})
    thr = stay.get("threshold", None)
    return float(thr) if thr is not None else None

def load_one(h: int, method: str) -> dict:
    p = REPORTS / f"posthoc_calibration_h{h}_{method}.json"
    if not p.exists():
        return {"h": h, "method": method, "threshold": None}
    try:
        obj = json.loads(p.read_text())
        thr = pick_threshold(obj)
    except Exception:
        thr = None
    return {"h": h, "method": method, "threshold": thr}

def main():
    rows = []
    for h in (24, 48):
        for method in ("isotonic", "sigmoid"):
            rows.append(load_one(h, method))

    # build markdown
    md = []
    md.append("### Decision Thresholds (auto-generated)")
    md.append("")
    md.append("| Horizon | Method   | Chosen Threshold | Source JSON |")
    md.append("|:-------:|:--------:|:----------------:|:------------|")
    for r in rows:
        h = r["h"]
        m = r["method"]
        thr = r["threshold"]
        src = f"outputs/reports/posthoc_calibration_h{h}_{m}.json"
        if thr is None:
            md.append(f"| {h}h | {m} | *(missing)* | `{src}` |")
        else:
            md.append(f"| {h}h | {m} | **{thr:.4f}** | `{src}` |")
    md.append("")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(str(OUT_MD))

if __name__ == "__main__":
    main()
