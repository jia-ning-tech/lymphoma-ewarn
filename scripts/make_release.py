#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_release.py
把当前结果打包到 outputs/release/<tag>/ 下，并在 README 中插入“可复现快照”段落（可选）。
仅用标准库，无第三方依赖。

用法示例：
  python scripts/make_release.py --tag v0.1-preprint --inject-readme
  python scripts/make_release.py --tag v0.1-preprint --dry-run
"""

from __future__ import annotations
import argparse, shutil, json, hashlib, os, datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs"
REL  = OUT / "release"

# 你可以在这里根据需要增删要快照的目录/文件模式
DEFAULT_GROUPS: Dict[str, List[str]] = {
    "reports": [
        "outputs/reports/*.json",
        "outputs/reports/*.csv",
        "outputs/reports/*.parquet",
    ],
    "figures": [
        "outputs/figures/*.png",
        "outputs/figures/*.pdf",
    ],
    "preds": [
        # 若不想包含预测明细，可注释掉
        "outputs/preds/preds_h*_val*.parquet",
        "outputs/preds/preds_h*_test*.parquet",
    ],
    "models": [
        "outputs/models/*",
    ],
    "docs": [
        "outputs/docs/*.md",
    ],
}

README_PATHS = [ROOT / "README.md", ROOT / "README.zh-CN.md"]
ANCHOR_START = "<!-- RELEASE_SNAPSHOT_START -->"
ANCHOR_END   = "<!-- RELEASE_SNAPSHOT_END -->"


@dataclass
class FileEntry:
    rel_path: str
    bytes: int
    md5: str

@dataclass
class Manifest:
    tag: str
    created_at: str
    groups: Dict[str, List[FileEntry]]

def _md5sum(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _glob_many(patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        for p in ROOT.glob(pat):
            if p.is_file():
                files.append(p)
    # 去重
    return sorted(set(files))

def copy_into_release(tag: str, groups=DEFAULT_GROUPS, dry: bool=False) -> Manifest:
    dst_root = REL / tag
    if not dry:
        dst_root.mkdir(parents=True, exist_ok=True)

    manifest_groups: Dict[str, List[FileEntry]] = {}
    for gname, patterns in groups.items():
        gdst = dst_root / gname
        if not dry:
            gdst.mkdir(parents=True, exist_ok=True)
        entries: List[FileEntry] = []
        files = _glob_many(patterns)
        for src in files:
            rel = src.relative_to(ROOT).as_posix()
            dst = gdst / src.name
            if not dry:
                shutil.copy2(src, dst)
                md5 = _md5sum(dst)
                size = dst.stat().st_size
            else:
                md5 = "DRYRUN"
                size = src.stat().st_size
            entries.append(FileEntry(rel_path=rel, bytes=size, md5=md5))
        manifest_groups[gname] = entries

    man = Manifest(
        tag=tag,
        created_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        groups=manifest_groups,
    )

    # 写 manifest.json
    if not dry:
        (dst_root / "manifest.json").write_text(
            json.dumps(asdict(man), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    return man

def _human_size(n: int) -> str:
    units = ["B","KB","MB","GB"]
    s = float(n); i=0
    while s>=1024 and i<len(units)-1:
        s/=1024; i+=1
    return f"{s:.2f} {units[i]}"

def build_readme_block(tag: str, man: Manifest) -> str:
    lines = []
    lines.append(ANCHOR_START)
    lines.append("")
    lines.append(f"## Reproducible snapshot — **{tag}**")
    lines.append("")
    rel_dir = f"outputs/release/{tag}"
    lines.append(f"Artifacts are saved under `{rel_dir}` (not committed to Git).")
    lines.append("")
    # 简表
    lines.append("| Group | #Files | Total Size | Example |")
    lines.append("|---:|---:|---:|---|")
    for g, entries in man.groups.items():
        n = len(entries)
        total = sum(e.bytes for e in entries)
        eg = entries[0].rel_path if n>0 else "-"
        lines.append(f"| `{g}` | {n} | {_human_size(total)} | `{eg}` |")
    lines.append("")
    lines.append(f"Generated at: `{man.created_at}`")
    lines.append("")
    lines.append("> Tip: pin this tag in your paper/README so others can reproduce figures & tables.")
    lines.append("")
    lines.append(ANCHOR_END)
    lines.append("")
    return "\n".join(lines)

def inject_readme(tag: str, man: Manifest, readmes=README_PATHS) -> None:
    block = build_readme_block(tag, man)
    for rp in readmes:
        if not rp.exists():
            continue
        old = rp.read_text(encoding="utf-8")
        if ANCHOR_START in old and ANCHOR_END in old:
            new = old.split(ANCHOR_START)[0] + block + old.split(ANCHOR_END)[1]
        else:
            # 若无锚点，默认在文件末尾追加
            new = old.rstrip() + "\n\n" + block
        rp.write_text(new, encoding="utf-8")
        print(f"[make_release] README injected -> {rp.relative_to(ROOT)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="snapshot tag, e.g. v0.1-preprint")
    ap.add_argument("--dry-run", action="store_true", help="do not copy, only print manifest")
    ap.add_argument("--inject-readme", action="store_true", help="inject snapshot block into README(.md / .zh-CN.md)")
    ap.add_argument("--no-preds", action="store_true", help="exclude outputs/preds/* from snapshot")
    ap.add_argument("--no-models", action="store_true", help="exclude outputs/models/* from snapshot")
    args = ap.parse_args()

    groups = dict(DEFAULT_GROUPS)
    if args.no_preds and "preds" in groups:
        groups.pop("preds")
    if args.no_models and "models" in groups:
        groups.pop("models")

    man = copy_into_release(args.tag, groups=groups, dry=args.dry_run)
    print(json.dumps(asdict(man), indent=2, ensure_ascii=False))

    if args.inject_readme:
        inject_readme(args.tag, man)

if __name__ == "__main__":
    main()
