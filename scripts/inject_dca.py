from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPS = ROOT / "outputs" / "reports"
FIGS = ROOT / "outputs" / "figures"
DOCS = ROOT / "outputs" / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

EN_START = "<!-- DCA_START -->"
EN_END   = "<!-- DCA_END -->"
ZH_START = "<!-- DCA_ZH_START -->"
ZH_END   = "<!-- DCA_ZH_END -->"

def nearest_values(df: pd.DataFrame, targets: list[float]) -> pd.DataFrame:
    out = []
    thr_arr = df["threshold"].to_numpy()
    for t in targets:
        idx = int(np.argmin(np.abs(thr_arr - t)))
        row = df.iloc[idx]
        out.append(dict(
            threshold=float(row["threshold"]),
            nb_model=float(row["nb_model"]),
            nb_all=float(row["nb_all"]),
            nb_none=float(row["nb_none"]),
        ))
    return pd.DataFrame(out)

def build_one_block(h: int, split: str, lang: str = "en") -> str:
    """
    Assemble one section for a given (h, split).
    """
    variants = [
        ("raw",      REPS / f"dca_h{h}_{split}.csv",         FIGS / f"dca_h{h}_{split}.png"),
        ("isotonic", REPS / f"dca_h{h}_{split}_cal_isotonic.csv", FIGS / f"dca_h{h}_{split}_cal_isotonic.png"),
        ("sigmoid",  REPS / f"dca_h{h}_{split}_cal_sigmoid.csv",  FIGS / f"dca_h{h}_{split}_cal_sigmoid.png"),
    ]
    # 仅保留存在的变体
    exist = [(name, csv, png) for (name, csv, png) in variants if csv.exists() and png.exists()]
    if not exist:
        return ""

    # 读取 & 取常用阈值的 NB（若不在范围，自动取最近点）
    display_thr = [0.05, 0.10, 0.20]
    rows = []
    for name, csv, png in exist:
        df = pd.read_csv(csv)
        # 兼容列名大小写
        df.columns = [c.strip().lower() for c in df.columns]
        if "prevalence" in df.columns:
            prev = float(df["prevalence"].iloc[0])
        else:
            prev = np.nan
        sel = nearest_values(df, display_thr)
        for _, r in sel.iterrows():
            rows.append(dict(
                horizon=h, split=split, variant=name,
                threshold=r["threshold"],
                nb_model=r["nb_model"], nb_all=r["nb_all"], nb_none=r["nb_none"],
                prevalence=prev
            ))

    tidy = pd.DataFrame(rows)
    # 构造 MD
    if lang == "en":
        title = f"### Decision Curve Analysis — h={h}h, split={split}"
        desc  = f"- Thresholds shown at ~0.05 / 0.10 / 0.20 (nearest grid from CSV).\n- `net benefit` per-patient (or per 100 patients if you ran with `--per-100`)."
        tbl_hdr = "| Variant | Threshold | Net benefit (Model) | Treat-all | Treat-none |\n|---|---:|---:|---:|---:|\n"
    else:
        title = f"### 决策曲线分析（DCA）— 预测窗 {h} 小时，数据集：{split}"
        desc  = f"- 下表展示约 0.05 / 0.10 / 0.20 三个典型阈值（自动就近取 CSV 网格）。\n- `净获益` 为每例患者（若在绘图时用 `--per-100`，则为每百例患者）。"
        tbl_hdr = "| 变体 | 阈值 | 净获益（模型） | Treat-all | Treat-none |\n|---|---:|---:|---:|---:|\n"

    # 表格
    lines = [title, "", desc, ""]
    lines.append(tbl_hdr.strip())
    for name, csv, png in exist:
        sub = tidy[tidy["variant"] == name].sort_values("threshold")
        for _, r in sub.iterrows():
            lines.append(f"| {name} | {r['threshold']:.3f} | {r['nb_model']:.4f} | {r['nb_all']:.4f} | {r['nb_none']:.4f} |")
    lines.append("")

    # 图片
    if lang == "en":
        lines.append("**Curves**")
    else:
        lines.append("**曲线**")
    for name, _, png in exist:
        rel = png.as_posix()
        cap = f"h={h}, {split}, {name}"
        lines.append(f"![{cap}]({rel})")
    lines.append("")
    return "\n".join(lines)

def assemble(horizons: list[int], splits: list[str], lang: str="en") -> str:
    parts = []
    header = "## Decision Curve Analysis (DCA)" if lang=="en" else "## 决策曲线分析（DCA）"
    parts.append(header)
    parts.append("")
    for h in horizons:
        for s in splits:
            blk = build_one_block(h, s, lang=lang)
            if blk:
                parts.append(blk)
    parts.append("")
    return "\n".join(parts)

def inject(readme: Path, start_tag: str, end_tag: str, content: str):
    if not readme.exists():
        readme.write_text("", encoding="utf-8")
    text = readme.read_text(encoding="utf-8")
    if start_tag in text and end_tag in text:
        pre = text.split(start_tag)[0]
        post = text.split(end_tag)[-1]
        new_text = pre + start_tag + "\n" + content + "\n" + end_tag + post
    else:
        # 追加锚点
        add = f"\n\n{start_tag}\n{content}\n{end_tag}\n"
        new_text = text + add
    readme.write_text(new_text, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=str, default="24,48")
    ap.add_argument("--splits", type=str, default="val,test")
    ap.add_argument("--readme", type=str, default="README.md")
    ap.add_argument("--readme-zh", type=str, default="README.zh-CN.md")
    ap.add_argument("--no-inject", action="store_true", help="only write MD files, do not inject into README")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    en_md = assemble(horizons, splits, lang="en")
    zh_md = assemble(horizons, splits, lang="zh")

    en_path = DOCS / "dca_section.md"
    zh_path = DOCS / "dca_section_zh.md"
    en_path.write_text(en_md, encoding="utf-8")
    zh_path.write_text(zh_md, encoding="utf-8")
    print(f"[inject_dca] wrote: {en_path}")
    print(f"[inject_dca] wrote: {zh_path}")

    if not args.no_inject:
        inject(ROOT / args.readme, EN_START, EN_END, en_md)
        inject(ROOT / args.readme_zh, ZH_START, ZH_END, zh_md)
        print(f"[inject_dca] injected into {args.readme} and {args.readme_zh}")

if __name__ == "__main__":
    main()
