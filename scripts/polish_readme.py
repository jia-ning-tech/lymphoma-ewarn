from __future__ import annotations
import argparse, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README_EN = ROOT / "README.md"
README_ZH = ROOT / "README.zh-CN.md"

BADGES_START = "<!-- BADGES_START -->"
BADGES_END   = "<!-- BADGES_END -->"
TOC_START = "<!-- TOC_START -->"
TOC_END   = "<!-- TOC_END -->"

def build_badges(lang: str) -> str:
    repo = "jia-ning-tech/lymphoma-ewarn"
    if lang == "en":
        lang_switch = (
            f"[🇨🇳 中文版](README.zh-CN.md) ｜ "
            f"[⭐ Star](https://github.com/{repo}/stargazers) ｜ "
            f"[�� Issues](https://github.com/{repo}/issues)"
        )
        lines = [
            BADGES_START,
            "",
            f"{lang_switch}",
            "",
            # Shields
            f"![GitHub Repo stars](https://img.shields.io/github/stars/{repo}?style=flat)",
            f"![GitHub issues](https://img.shields.io/github/issues/{repo}?style=flat)",
            f"![License](https://img.shields.io/badge/License-MIT-green.svg)",
            # Zenodo/Release 占位（若未来有 DOI，可改为真实 DOI 徽章）
            f"![Release](https://img.shields.io/github/v/release/{repo}?display_name=tag&sort=semver)",
            "",
            BADGES_END,
        ]
    else:
        lang_switch = (
            f"[🇬🇧 English](README.md) ｜ "
            f"[⭐ 收藏](https://github.com/{repo}/stargazers) ｜ "
            f"[🐛 问题](https://github.com/{repo}/issues)"
        )
        lines = [
            BADGES_START,
            "",
            f"{lang_switch}",
            "",
            f"![GitHub Repo stars](https://img.shields.io/github/stars/{repo}?style=flat)",
            f"![GitHub issues](https://img.shields.io/github/issues/{repo}?style=flat)",
            f"![License](https://img.shields.io/badge/License-MIT-green.svg)",
            f"![Release](https://img.shields.io/github/v/release/{repo}?display_name=tag&sort=semver)",
            "",
            BADGES_END,
        ]
    return "\n".join(lines)

def slugify(title: str) -> str:
    # 与 GitHub 风格近似的锚点
    s = title.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s

def build_toc(md_text: str, lang: str) -> str:
    """
    解析 markdown 中的标题(#/##/###)，生成 TOC。
    """
    lines = md_text.splitlines()
    items = []
    for ln in lines:
        m = re.match(r"^(#{1,6})\s+(.+)$", ln.strip())
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        # 跳过文首 H1（通常是项目名），只收 H2/H3 常用层级
        if level == 1:
            continue
        anchor = slugify(title)
        indent = "  " * (level - 2)  # H2 ->0, H3->1, H4->2...
        items.append(f"{indent}- [{title}](#{anchor})")

    header = "## Table of Contents" if lang == "en" else "## 目录"
    out = [TOC_START, "", header, ""]
    if items:
        out.extend(items)
    else:
        out.append("- (empty)")
    out.append("")
    out.append(TOC_END)
    return "\n".join(out)

def inject_block(text: str, start_tag: str, end_tag: str, block: str, after_header: bool=False) -> str:
    if start_tag in text and end_tag in text:
        pre = text.split(start_tag)[0]
        post = text.split(end_tag)[-1]
        return pre + block + post
    # 不存在则追加
    if after_header:
        # 将 block 插到首个非空行后
        lines = text.splitlines()
        if not lines:
            return block + "\n" + text
        # 若第一行是 H1，则插到它后面一空行
        if re.match(r"^#\s+", lines[0]):
            new = [lines[0], "", block] + lines[1:]
        else:
            new = [block] + lines
        return "\n".join(new)
    else:
        return text + ("\n\n" if not text.endswith("\n") else "\n") + block + "\n"

def process(readme_path: Path, lang: str):
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    badges = build_badges(lang)
    # 徽章：加在最前面（H1 后）
    text = inject_block(text, BADGES_START, BADGES_END, badges, after_header=True)
    # TOC：加在徽章块之后
    toc = build_toc(text, lang)
    # 为了放在徽章后面：先把徽章块取出来的位置，再注入 toc
    if BADGES_START in text:
        pre = text.split(BADGES_START)[0]
        rest = text.split(BADGES_START)[1]
        # rest 内含 BADGES_END 及其后续文本
        # 先重组为：pre + badges + toc + 剩余
        badges_block = BADGES_START + rest.split(BADGES_END)[0] + BADGES_END
        remain = rest.split(BADGES_END)[1]
        combined = pre + badges_block + "\n\n" + toc + remain
        # 若已存在 TOC，则覆盖
        combined = inject_block(combined, TOC_START, TOC_END, toc, after_header=False)
        readme_path.write_text(combined, encoding="utf-8")
    else:
        # 兜底：直接注入
        text = inject_block(text, TOC_START, TOC_END, toc, after_header=False)
        readme_path.write_text(text, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--readme-zh", default="README.zh-CN.md")
    args = ap.parse_args()
    process(ROOT / args.readme, lang="en")
    process(ROOT / args.readme_zh, lang="zh")
    print("[polish_readme] badges + TOC injected into README & README.zh-CN")

if __name__ == "__main__":
    main()
