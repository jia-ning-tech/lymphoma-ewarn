from __future__ import annotations
from pathlib import Path
import argparse, json, pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
REPS = ROOT / "outputs" / "reports"
FIGS = ROOT / "outputs" / "figures"
DOCS = ROOT / "outputs" / "docs"

# --- robust import of rel_to_root ---
try:
    from scripts.util_paths import rel_to_root  # when run as a module: python -m scripts.inject_ablation
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/
    from util_paths import rel_to_root  # when run as a plain script: python scripts/inject_ablation.py

ABL_START = "<!-- ABLATION_START -->"
ABL_END   = "<!-- ABLATION_END -->"

def build_tables_for_horizon(tidy_csv: Path, lang="en") -> str:
    df = pd.read_csv(tidy_csv)
    needed = ["horizon","setting","group","n_features","metric","mean","std"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"{tidy_csv} missing column: {c}")

    h = int(df["horizon"].iloc[0])
    title_en = f"### Ablation (h={h}h)"
    title_zh = f"### 消融实验（窗口={h}小时）"
    title = title_en if lang=="en" else title_zh

    out = [title, ""]
    for met in ("AP","AUROC"):
        sub = df[df["metric"]==met].copy()
        sub["disp"] = sub.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
        sub = sub[["setting","group","n_features","disp"]].rename(columns={
            "setting":"Setting","group":"Group","n_features":"#Feat","disp":met
        })
        out.append(f"**{met}**")
        out.append("")
        out.append("| Setting | Group | #Feat | "+met+" |")
        out.append("|---|---:|---:|---:|")
        for _,r in sub.iterrows():
            out.append(f"| {r['Setting']} | {r['Group']} | {int(r['#Feat'])} | {r[met]} |")
        out.append("")
    return "\n".join(out)

def build_section(h: int, reports_dir: Path, fig_dir: Path, lang="en") -> str:
    tidy_csv = reports_dir / f"ablation_h{h}_tidy_for_readme.csv"
    tables = build_tables_for_horizon(tidy_csv, lang=lang)

    fig_bar = rel_to_root(fig_dir / f"ablation_h{h}_bar.png")
    fig_forest = rel_to_root(fig_dir / f"ablation_h{h}_forest.png")

    cap_bar_en = f"Ablation bar chart (h={h}h)"
    cap_bar_zh = f"消融柱状图（窗口={h}小时）"
    cap_for_en = f"Ablation forest plot (h={h}h)"
    cap_for_zh = f"消融森林图（窗口={h}小时）"

    cap_bar = cap_bar_en if lang=="en" else cap_bar_zh
    cap_for = cap_for_en if lang=="en" else cap_for_zh

    parts = [tables, ""]
    parts += [f"![{cap_bar}]({fig_bar})", ""]
    parts += [f"![{cap_for}]({fig_forest})", ""]
    return "\n".join(parts)

def assemble_full_md(horizons, reports_dir, fig_dir, lang="en") -> str:
    parts = []
    for h in horizons:
        parts.append(build_section(h, reports_dir, fig_dir, lang=lang))
    return "\n\n".join(parts)

def inject(readme_path: Path, block: str):
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    if ABL_START in text and ABL_END in text:
        pre = text.split(ABL_START)[0]
        post = text.split(ABL_END)[-1]
        new = pre + ABL_START + "\n\n" + block + "\n" + ABL_END + post
    else:
        new = text + ("\n\n" if not text.endswith("\n") else "\n") + ABL_START + "\n\n" + block + "\n" + ABL_END + "\n"
    readme_path.write_text(new, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="24,48")
    ap.add_argument("--reports-dir", default=str(REPS))
    ap.add_argument("--fig-dir", default=str(FIGS))
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--readme-zh", default="README.zh-CN.md")
    args = ap.parse_args()

    horizons = [int(x) for x in str(args.horizons).split(",") if x]

    reports_dir = Path(args.reports_dir)
    fig_dir = Path(args.fig_dir)
    DOCS.mkdir(parents=True, exist_ok=True)

    en_md = assemble_full_md(horizons, reports_dir, fig_dir, lang="en")
    zh_md = assemble_full_md(horizons, reports_dir, fig_dir, lang="zh")

    (DOCS / "ablation_section.md").write_text(en_md, encoding="utf-8")
    (DOCS / "ablation_section_zh.md").write_text(zh_md, encoding="utf-8")

    inject(ROOT / args.readme, en_md)
    inject(ROOT / args.readme_zh, zh_md)

    print("[inject_ablation] DONE (relative image paths, robust import).")

if __name__ == "__main__":
    main()
