# scripts/inject_fi.py  —— 只替换 inject() 与 _read_md()

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
README_EN = ROOT / "README.md"
README_ZH = ROOT / "README.zh-CN.md"
DOCS = ROOT / "outputs" / "docs"

START = "<!-- FI_START -->"
END   = "<!-- FI_END -->"

def _read_md(h:int, split:str) -> str:
    tab = DOCS / f"fi_top_table_h{h}_{split}.md"
    assert tab.exists(), f"missing {tab}"
    # 1) 逐行去掉前导空白，确保以 '|' 开头
    lines = [ln.lstrip() for ln in tab.read_text(encoding="utf-8").splitlines()]
    # 2) 前后各加一空行，避免与上文 blockquote/列表粘连
    body = "\n" + "\n".join(lines) + "\n"
    # 3) 返回“图 + 表”片段（相对路径）
    fig_rel = f"outputs/figures/fi_forest_h{h}_{split}.png"
    title = f"### 预测提前窗 h={h} — {'验证集' if split=='val' else '测试集'}"
    return (
        f"{title}\n\n"
        f"![特征重要性 — h{h} {split}]( {fig_rel} )\n"
        f"{body}"
    )

def inject(readme_path: Path) -> None:
    md_en = "## Feature importance\n\n> Results are permutation importance (ΔAUC, mean over repeats). SHAP will be added if available.\n\n" \
            + _read_md(24,"val") + "\n" + _read_md(24,"test") + "\n" + _read_md(48,"val") + "\n" + _read_md(48,"test") + "\n"
    md_zh = "## 特征重要性\n\n> 结果为置换重要度（ΔAUC，多次重复的均值）。若存在 SHAP，将一并展示。\n\n" \
            + _read_md(24,"val") + "\n" + _read_md(24,"test") + "\n" + _read_md(48,"val") + "\n" + _read_md(48,"test") + "\n"

    text = readme_path.read_text(encoding="utf-8")
    repl = md_zh if readme_path.name.endswith("README.zh-CN.md") else md_en
    patt = re.compile(re.escape(START)+r".*?"+re.escape(END), flags=re.S)
    new = patt.sub(START + "\n" + repl + END, text)
    readme_path.write_text(new, encoding="utf-8")
    print(f"[inject_fi] injected into {readme_path}")
