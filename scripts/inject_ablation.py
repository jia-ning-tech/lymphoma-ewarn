#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import re
from typing import List

EN_START = "<!-- ABLATION:START -->"
EN_END   = "<!-- ABLATION:END -->"
ZH_START = "<!-- ABLATION-ZH:START -->"
ZH_END   = "<!-- ABLATION-ZH:END -->"

def fmt_mean_std(m, s, nd=4):
    if pd.isna(m) or pd.isna(s):
        return ""
    return f"{m:.{nd}f} ± {s:.{nd}f}"

def _lc_map(df: pd.DataFrame) -> dict:
    """lower-case mapping of column -> original"""
    return {c.lower(): c for c in df.columns}

def _get_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """return the actual column name matched in df (case-insensitive)"""
    lc = _lc_map(df)
    for cand in candidates:
        k = cand.lower()
        if k in lc:
            return lc[k]
    return None

def _maybe_split_mean_std_col(df: pd.DataFrame, col_name: str, mean_target: str, std_target: str):
    """
    If df[col_name] looks like '0.7974 ± 0.0228' (or with spaces), split into two new float columns.
    """
    if col_name is None or col_name not in df.columns:
        return df
    patt = re.compile(r"^\s*([0-9.+-eE]+)\s*±\s*([0-9.+-eE]+)\s*$")
    means, stds = [], []
    dirty = False
    for val in df[col_name].astype(str).fillna(""):
        m = patt.match(val)
        if m:
            means.append(float(m.group(1)))
            stds.append(float(m.group(2)))
            dirty = True
        else:
            means.append(float("nan"))
            stds.append(float("nan"))
    if dirty:
        df = df.copy()
        df[mean_target] = means
        df[std_target]  = stds
    return df

def normalize_tidy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize various possible column namings into:
    ['setting','group','n_features','AUROC_mean','AUROC_std','AP_mean','AP_std']
    """
    df = df.copy()
    # 先尝试把“合并列”拆成 mean/std
    for merged, (m_name, s_name) in {
        "AUROC": ("AUROC_mean","AUROC_std"),
        "auroc": ("AUROC_mean","AUROC_std"),
        "AP":    ("AP_mean","AP_std"),
        "ap":    ("AP_mean","AP_std"),
    }.items():
        if merged in df.columns:
            df = _maybe_split_mean_std_col(df, merged, m_name, s_name)

    # 尝试匹配大小写不敏感列名
    col_setting   = _get_col(df, ["setting","mode","type"])
    col_group     = _get_col(df, ["group","feature_group","feat_group"])
    col_nfeat     = _get_col(df, ["n_features","nfeatures","#features","features"])
    col_au_mean   = _get_col(df, ["AUROC_mean","auroc_mean","mean_auroc","aurocmean"])
    col_au_std    = _get_col(df, ["AUROC_std","auroc_std","std_auroc","aurocstd"])
    col_ap_mean   = _get_col(df, ["AP_mean","ap_mean","mean_ap","apmean"])
    col_ap_std    = _get_col(df, ["AP_std","ap_std","std_ap","apstd"])

    # 兜底：如果还有缺失，尝试从合并列里再拆一次（防止大小写映射遗漏）
    if col_au_mean is None or col_au_std is None:
        au_col = _get_col(df, ["AUROC","auroc"])
        if au_col:
            df = _maybe_split_mean_std_col(df, au_col, "AUROC_mean","AUROC_std")
            col_au_mean = "AUROC_mean"
            col_au_std  = "AUROC_std"

    if col_ap_mean is None or col_ap_std is None:
        ap_col = _get_col(df, ["AP","ap"])
        if ap_col:
            df = _maybe_split_mean_std_col(df, ap_col, "AP_mean","AP_std")
            col_ap_mean = "AP_mean"
            col_ap_std  = "AP_std"

    missing = []
    if col_setting is None: missing.append("setting")
    if col_group is None:   missing.append("group")
    if col_nfeat is None:   missing.append("n_features")
    if col_au_mean is None: missing.append("AUROC_mean")
    if col_au_std is None:  missing.append("AUROC_std")
    if col_ap_mean is None: missing.append("AP_mean")
    if col_ap_std is None:  missing.append("AP_std")
    if missing:
        raise ValueError(f"tidy CSV missing columns (after normalization): {missing}")

    out = pd.DataFrame({
        "setting": df[col_setting],
        "group": df[col_group],
        "n_features": df[col_nfeat],
        "AUROC_mean": pd.to_numeric(df[col_au_mean], errors="coerce"),
        "AUROC_std":  pd.to_numeric(df[col_au_std],  errors="coerce"),
        "AP_mean":    pd.to_numeric(df[col_ap_mean], errors="coerce"),
        "AP_std":     pd.to_numeric(df[col_ap_std],  errors="coerce"),
    })
    # 类型规范
    out["n_features"] = pd.to_numeric(out["n_features"], errors="coerce").fillna(0).astype(int)
    return out

def build_tables_for_horizon(tidy_csv: Path, lang="en") -> str:
    df = pd.read_csv(tidy_csv)
    df = normalize_tidy(df)
    df["AUROC"] = df.apply(lambda r: fmt_mean_std(r["AUROC_mean"], r["AUROC_std"]), axis=1)
    df["AP"]    = df.apply(lambda r: fmt_mean_std(r["AP_mean"], r["AP_std"]), axis=1)
    df = df[["setting","group","n_features","AUROC","AP"]]

    if lang == "zh":
        header = "| 设置 | 分组 | 特征数 | AUROC(均值±std) | AP(均值±std) |"
    else:
        header = "| Setting | Group | #Features | AUROC(mean±std) | AP(mean±std) |"

    lines = [header, "|---|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['setting']} | {r['group']} | {int(r['n_features'])} | {r['AUROC']} | {r['AP']} |")
    return "\n".join(lines)

def build_section(h: int, reports_dir: Path, fig_dir: Path, lang="en") -> str:
    tidy = reports_dir / f"ablation_h{h}_tidy_for_readme.csv"
    bar  = fig_dir / f"ablation_h{h}_bar.png"
    forest = fig_dir / f"ablation_h{h}_forest.png"

    if lang == "zh":
        title = f"### 消融研究（h={h}）"
        fig_intro = "**图示**：左为柱状图（均值±标准差），右为森林图。"
    else:
        title = f"### Ablation (h={h})"
        fig_intro = "**Figures**: Left = bar (mean±std), Right = forest."

    parts = [title, ""]
    fig_lines = []
    if bar.exists():
        fig_lines.append(f"![ablation_h{h}_bar]({bar.as_posix()})")
    if forest.exists():
        fig_lines.append(f"![ablation_h{h}_forest]({forest.as_posix()})")
    if fig_lines:
        parts.append(fig_intro)
        parts.append("")
        parts.extend(fig_lines)
        parts.append("")

    if tidy.exists():
        parts.append(build_tables_for_horizon(tidy, lang=lang))
        parts.append("")
    else:
        parts.append("> (No tidy CSV found; run `make ablation-plots` first.)")
        parts.append("")
    return "\n".join(parts)

def assemble_full_md(horizons: List[int], reports_dir: Path, fig_dir: Path, lang="en") -> str:
    if lang == "zh":
        h2 = "## 消融实验"
        preface = "本节统一汇报 **keep-only** 与 **drop-one** 两类消融，所有点均为 5 折交叉验证的**均值±标准差**。"
    else:
        h2 = "## Ablation Studies"
        preface = "This section unifies **keep-only** and **drop-one** ablation; all points are 5-fold CV **mean ± std**."
    parts = [h2, "", preface, ""]
    for h in horizons:
        parts.append(build_section(h, reports_dir, fig_dir, lang=lang))
    return "\n".join(parts).rstrip() + "\n"

def inject(readme: Path, start_tag: str, end_tag: str, payload: str):
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    if start_tag in text and end_tag in text:
        before = text.split(start_tag)[0]
        after  = text.split(end_tag)[-1]
        new = f"{before}{start_tag}\n{payload}\n{end_tag}{after}"
    else:
        new_block = f"\n{start_tag}\n{payload}\n{end_tag}\n"
        new = text + ("\n" if text and not text.endswith("\n") else "") + new_block
    readme.write_text(new, encoding="utf-8")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="", help="comma list; if empty infer from tidy files")
    ap.add_argument("--reports_dir", default="outputs/reports")
    ap.add_argument("--fig_dir", default="outputs/figures")
    ap.add_argument("--out_dir", default="outputs/docs")
    ap.add_argument("--readme", default="", help="README.md to inject")
    ap.add_argument("--readme_zh", default="", help="README.zh-CN.md to inject")
    ap.add_argument("--emit_only", action="store_true", help="only emit md fragments; no injection")
    return ap.parse_args()

def infer_horizons(reports_dir: Path) -> list[int]:
    hs = []
    for p in reports_dir.glob("ablation_h*_tidy_for_readme.csv"):
        try:
            h = int(p.name.split("_")[1][1:])  # ablation_h{h}_...
            hs.append(h)
        except Exception:
            pass
    return sorted(set(hs))

def main():
    args = parse_args()
    reports_dir = Path(args.reports_dir).resolve()
    fig_dir = Path(args.fig_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.horizons:
        horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    else:
        horizons = infer_horizons(reports_dir)
    if not horizons:
        print("No horizons found. Run ablation plotting first.", file=sys.stderr)
        sys.exit(1)

    en_md = assemble_full_md(horizons, reports_dir, fig_dir, lang="en")
    zh_md = assemble_full_md(horizons, reports_dir, fig_dir, lang="zh")

    en_out = out_dir / "ablation_section.md"
    zh_out = out_dir / "ablation_section_zh.md"
    en_out.write_text(en_md, encoding="utf-8")
    zh_out.write_text(zh_md, encoding="utf-8")

    if not args.emit_only:
        if args.readme:
            inject(Path(args.readme), EN_START, EN_END, en_md)
        if args.readme_zh:
            inject(Path(args.readme_zh), ZH_START, ZH_END, zh_md)

if __name__ == "__main__":
    main()
