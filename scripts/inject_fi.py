#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG  = ROOT / "outputs" / "figures"
DOC  = ROOT / "outputs" / "docs"

READMES = [ROOT/"README.md", ROOT/"README.zh-CN.md"]
START = "<!-- FI_START -->"
END   = "<!-- FI_END -->"

def read_strict(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write_strict(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def inject_block(text: str, block: str) -> str:
    if START not in text or END not in text:
        raise RuntimeError("README 缺少 FI_START / FI_END 锚点")
    pre  = text.split(START, 1)[0]
    tail = text.split(END,   1)[1]
    return pre + START + "\n" + block.strip() + "\n" + END + tail

def build_section(lang: str) -> str:
    # 语言行文
    title = "## 特征重要性" if lang.startswith("zh") else "## Feature importance"
    note  = ("> 结果为置换重要度（多次重复取均值的 ΔAUC）。当前 SHAP 因模型封装未运行，后续会修复补充。"
             if lang.startswith("zh") else
             "> We report permutation importance (mean ΔAUC over repeats). SHAP is currently unavailable due to model wrapper; will be added later.")

    parts = [title, "", note, ""]

    for h in (24, 48):
        for split in ("val","test"):
            subt = f"### 预测提前窗 h={h} — {'验证集' if lang.startswith('zh') else 'validation'}" if split=="val" \
                   else f"### 预测提前窗 h={h} — {'测试集' if lang.startswith('zh') else 'test'}"
            fig_rel = f"outputs/figures/fi_forest_h{h}_{split}.png"
            table_md = DOC / f"fi_top_table_h{h}_{split}.md"
            parts.append(subt)
            parts.append("")
            parts.append(f"![{'特征重要性' if lang.startswith('zh') else 'Feature importance'} — h{h} {split}]({fig_rel})")
            parts.append("")
            # 直接把 table md 文件的内容嵌进来；它本身已保证无缩进、前后有空行
            if table_md.exists():
                parts.append(read_strict(table_md).strip())
            else:
                parts.append("*(表格缺失 / table missing)*")
            parts.append("")

    return "\n".join(parts).strip() + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-en", action="store_true")
    ap.add_argument("--no-zh", action="store_true")
    args = ap.parse_args()

    for rm in READMES:
        lang = "zh" if rm.name.endswith("zh-CN.md") else "en"
        if (lang=="zh" and args.no_zh) or (lang=="en" and args.no_en):
            continue
        text = read_strict(rm)
        block = build_section(lang)
        new = inject_block(text, block)
        write_strict(rm, new)
        print(f"[inject_fi] injected into {rm}")

if __name__ == "__main__":
    main()
