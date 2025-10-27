#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/"outputs/figures"
DOC = ROOT/"outputs/docs"

PAIRS = [
    (24, "val"),
    (24, "test"),
    (48, "val"),
    (48, "test"),
]

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""

def build_section_en() -> str:
    lines = []
    lines.append("## Feature Importance")
    lines.append("")
    lines.append("> Permutation importance (mean ΔAUC over repeats). SHAP is currently skipped due to model wrapper; we’ll add it in a later update.")
    lines.append("")
    for h, sp in PAIRS:
        fig_rel = f"outputs/figures/fi_forest_h{h}_{sp}.png"
        table_md = DOC/f"fi_top_table_h{h}_{sp}.md"
        lines.append(f"### h={h} — {sp}")
        lines.append("")
        lines.append(f"![Feature importance — h{h} {sp}]({fig_rel})")
        lines.append("")
        if table_md.exists():
            # 直接把表格 md 内联
            lines.append(read_text(table_md).strip())
            lines.append("")
        else:
            lines.append(f"*Table missing: {table_md}*")
            lines.append("")
    return "\n".join(lines).strip() + "\n"

def build_section_zh() -> str:
    lines = []
    lines.append("## 特征重要性")
    lines.append("")
    lines.append("> 结果为置换重要度（多次重复取均值的 ΔAUC）。当前 SHAP 因模型封装为字典未运行，后续会修复补充。")
    lines.append("")
    for h, sp in PAIRS:
        fig_rel = f"outputs/figures/fi_forest_h{h}_{sp}.png"
        table_md = DOC/f"fi_top_table_h{h}_{sp}.md"
        zh_split = "验证集" if sp=="val" else "测试集"
        lines.append(f"### 预测提前窗 h={h} — {zh_split}")
        lines.append("")
        lines.append(f"![特征重要性 — h{h} {sp}]({fig_rel})")
        lines.append("")
        if table_md.exists():
            lines.append(read_text(table_md).strip())
            lines.append("")
        else:
            lines.append(f"*缺少表格文件：{table_md}*")
            lines.append("")
    return "\n".join(lines).strip() + "\n"

def inject(readme: Path, start_tag: str, end_tag: str, payload: str):
    text = read_text(readme)
    if start_tag in text and end_tag in text:
        pre = text.split(start_tag)[0]
        post = text.split(end_tag)[1]
        new = pre + start_tag + "\n" + payload + "\n" + end_tag + post
    else:
        # 自动追加锚点块
        new = text.rstrip()+"\n\n"+start_tag+"\n"+payload+"\n"+end_tag+"\n"
    readme.write_text(new, encoding="utf-8")
    print(f"[inject_fi] injected into {readme}")

def main():
    en = build_section_en()
    zh = build_section_zh()

    # 写出独立片段，便于单独查看
    (DOC/"fi_section.md").write_text(en, encoding="utf-8")
    (DOC/"fi_section_zh.md").write_text(zh, encoding="utf-8")
    print("[inject_fi] wrote:", DOC/"fi_section.md", "and", DOC/"fi_section_zh.md")

    # 注入 README 与 README.zh-CN.md
    inject(ROOT/"README.md",
           "<!-- FI_START -->", "<!-- FI_END -->", en)
    zh_readme = ROOT/"README.zh-CN.md"
    if zh_readme.exists():
        inject(zh_readme,
               "<!-- FI_START -->", "<!-- FI_END -->", zh)
    else:
        print("[inject_fi] README.zh-CN.md not found, skipped.")

if __name__ == "__main__":
    sys.exit(main())
