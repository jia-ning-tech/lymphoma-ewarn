from __future__ import annotations
from pathlib import Path
import argparse, sys

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "outputs" / "figures"
DOCS = ROOT / "outputs" / "docs"

# --- robust import of rel_to_root ---
try:
    from scripts.util_paths import rel_to_root
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/
    from util_paths import rel_to_root

DCA_START = "<!-- DCA_START -->"
DCA_END   = "<!-- DCA_END -->"

def build_block(h: int, split: str, methods=("raw","isotonic","sigmoid"), lang="en") -> str:
    title_en = f"### Decision Curve Analysis (h={h}h, split={split})"
    title_zh = f"### 决策曲线（窗口={h}小时，数据集={split}）"
    title = title_en if lang=="en" else title_zh
    parts = [title, ""]
    for m in methods:
        if m == "raw":
            fname = f"dca_h{h}_{split}.png"
            cap_en = f"DCA (raw probability)"
            cap_zh = f"DCA（原始概率）"
        else:
            fname = f"dca_h{h}_{split}_cal_{m}.png"
            cap_en = f"DCA (calibrated: {m})"
            cap_zh = f"DCA（后处理校准：{m}）"
        p = rel_to_root(FIGS / fname)
        cap = cap_en if lang=="en" else cap_zh
        parts += [f"![{cap}]({p})", ""]
    return "\n".join(parts)

def inject(readme_path: Path, block: str):
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    if DCA_START in text and DCA_END in text:
        pre = text.split(DCA_START)[0]
        post = text.split(DCA_END)[-1]
        new = pre + DCA_START + "\n\n" + block + "\n" + DCA_END + post
    else:
        new = text + ("\n\n" if not text.endswith("\n") else "\n") + DCA_START + "\n\n" + block + "\n" + DCA_END + "\n"
    readme_path.write_text(new, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="24,48")
    ap.add_argument("--split", default="test")
    ap.add_argument("--methods", default="raw,isotonic,sigmoid")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--readme-zh", default="README.zh-CN.md")
    args = ap.parse_args()

    horizons = [int(x) for x in str(args.horizons).split(",") if x]
    methods = [m for m in str(args.methods).split(",") if m]

    en_parts, zh_parts = [], []
    for h in horizons:
        en_parts.append(build_block(h, args.split, methods=methods, lang="en"))
        zh_parts.append(build_block(h, args.split, methods=methods, lang="zh"))

    en_md = "\n\n".join(en_parts)
    zh_md = "\n\n".join(zh_parts)

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "dca_section.md").write_text(en_md, encoding="utf-8")
    (DOCS / "dca_section_zh.md").write_text(zh_md, encoding="utf-8")

    inject(ROOT / args.readme, en_md)
    inject(ROOT / args.readme_zh, zh_md)

    print("[inject_dca] DONE (relative image paths, robust import).")

if __name__ == "__main__":
    main()
