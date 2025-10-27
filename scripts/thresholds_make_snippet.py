#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将各 horizon/split/method 的阈值 Markdown 片段合并为
  outputs/docs/thresholds_for_readme.md
供你现有的 scripts/inject_thresholds.py 使用（无需改动老脚本）。

用法：
  python scripts/thresholds_make_snippet.py --horizons 24,48 --split test --methods raw,isotonic,sigmoid
"""

from __future__ import annotations
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
DOCS = ROOT / "outputs" / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

def build_snippet(horizons: list[int], split: str, methods: list[str]) -> str:
    parts = []
    parts.append("## Threshold selection & confusion matrices")
    parts.append("")
    for h in horizons:
        parts.append(f"### Horizon h={h} ({split})")
        parts.append("")
        for m in methods:
            p = DOCS / f"threshold_h{h}_{split}_{m}.md"
            if p.exists():
                parts.append(p.read_text(encoding="utf-8").strip())
                parts.append("")
            else:
                parts.append(f"> _No report for h={h}, method={m} (file missing)_")
                parts.append("")
    return "\n".join(parts).rstrip() + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="24,48")
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--methods", default="raw,isotonic,sigmoid")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    methods  = [m.strip() for m in args.methods.split(",") if m.strip()]

    out_md = DOCS / "thresholds_for_readme.md"
    out_md.write_text(build_snippet(horizons, args.split, methods), encoding="utf-8")
    print(f"[snippet] wrote -> {out_md}")

if __name__ == "__main__":
    main()
