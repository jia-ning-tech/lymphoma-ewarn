#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

def insert_block(readme: Path, start_tag: str, end_tag: str, block_text: str) -> str:
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
    else:
        text = ""

    if start_tag not in text or end_tag not in text:
        # append anchors at the end if not present
        parts = [text.rstrip(), "", start_tag, "", end_tag, ""]
        text = "\n".join([p for p in parts if p is not None])

    lines = text.splitlines()
    out_lines = []
    in_block = False
    for line in lines:
        if start_tag in line and not in_block:
            out_lines.append(line)
            out_lines.extend(block_text.splitlines())
            in_block = True
            continue
        if end_tag in line and in_block:
            out_lines.append(line)
            in_block = False
            continue
        if not in_block:
            out_lines.append(line)

    return "\n".join(out_lines) + ("\n" if (out_lines and not out_lines[-1].endswith("\n")) else "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", required=True, help="README file path.")
    ap.add_argument("--source", required=True, help="Markdown to inject.")
    ap.add_argument("--block-start", required=True, help="Start anchor string.")
    ap.add_argument("--block-end", required=True, help="End anchor string.")
    args = ap.parse_args()

    readme = Path(args.readme)
    src = Path(args.source)
    block = src.read_text(encoding="utf-8")

    updated = insert_block(readme, args.block_start, args.block_end, block)
    readme.write_text(updated, encoding="utf-8")
    print(f"[inject] injected {src} into {readme} between anchors.")

if __name__ == "__main__":
    main()
