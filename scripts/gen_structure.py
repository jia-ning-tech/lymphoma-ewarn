#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

DESC = {
    "src": "Source code.",
    "src/cli": "Command-line tools for training, evaluation, plots, release.",
    "data_raw": "Original/raw data (not tracked).",
    "data_interim": "Intermediate/engineered feature tables for modeling.",
    "outputs": "Auto-generated artifacts.",
    "outputs/models": "Trained model bundles (.joblib).",
    "outputs/preds": "Inference & validation predictions (.parquet).",
    "outputs/reports": "Metrics/tables used in paper & README (CSV/JSON/Parquet).",
    "outputs/figures": "Figures (ROC/PR, calibration, SHAP, lead-time, etc.).",
    "outputs/release": "Model release package for monitoring.",
    "notebooks": "Exploratory notebooks (optional).",
    "scripts": "Utility scripts (optional).",
}

IGNORE_DIRS = {
    ".git", ".github", "__pycache__", ".ipynb_checkpoints",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "venv", ".venv"
}

def build_tree(start: Path, prefix: str = "") -> list[str]:
    lines = []
    try:
        entries = sorted(
            [p for p in start.iterdir() if p.name not in IGNORE_DIRS],
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except PermissionError:
        return lines

    for i, p in enumerate(entries):
        is_last = i == len(entries) - 1
        branch = "└── " if is_last else "├── "
        if p.is_dir():
            lines.append(f"{prefix}{branch}{p.name}/")
            ext_prefix = f"{prefix}    " if is_last else f"{prefix}│   "
            lines.extend(build_tree(p, ext_prefix))
        else:
            lines.append(f"{prefix}{branch}{p.name}")
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output markdown path, e.g., outputs/docs/project_structure.md")
    args = ap.parse_args()

    root = Path(".").resolve()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for k, v in sorted(DESC.items()):
        p = root / k
        exists = p.exists()
        n_files = sum(1 for _ in p.rglob("*") if _.is_file()) if exists else 0
        rows.append((k, v, "yes" if exists else "no", n_files))

    md = []
    md.append("## Project Structure")
    md.append("")
    md.append("### Key Directories (auto-generated)")
    md.append("")
    md.append("| Path | Description | Exists | #Files |")
    md.append("|---|---|:---:|---:|")
    for k, desc, exists, n in rows:
        md.append(f"| `{k}` | {desc} | {exists} | {n} |")
    md.append("")
    md.append("### Full Tree (auto-generated)")
    md.append("")
    md.append("```text")
    md.extend(build_tree(root))
    md.append("```")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(out_path)

if __name__ == "__main__":
    main()
