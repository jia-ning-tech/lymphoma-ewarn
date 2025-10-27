#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inject thresholds snippet into README files between anchors:
  <!-- THRESHOLDS_START --> ... <!-- THRESHOLDS_END -->

Usage:
  python scripts/inject_thresholds.py
"""

from __future__ import annotations
from pathlib import Path
import re
import sys

SNIPPET = Path("outputs/docs/thresholds_for_readme.md")
TARGETS = [Path("README.md"), Path("README.zh-CN.md")]

START = "<!-- THRESHOLDS_START -->"
END   = "<!-- THRESHOLDS_END -->"

def inject_once(readme_path: Path, snippet_path: Path) -> bool:
    if not readme_path.exists():
        return False
    snippet = snippet_path.read_text(encoding="utf-8") if snippet_path.exists() else ""
    readme = readme_path.read_text(encoding="utf-8")

    block = f"{START}\n\n{snippet}\n{END}\n"

    if START in readme and END in readme:
        # replace existing block
        new_readme = re.sub(rf"{re.escape(START)}.*?{re.escape(END)}",
                            block, readme, flags=re.S)
    else:
        # append block to the end
        new_readme = readme.rstrip() + "\n\n" + block

    if new_readme != readme:
        readme_path.write_text(new_readme, encoding="utf-8")
        print(f"[inject] updated: {readme_path}")
        return True
    else:
        print(f"[inject] no change: {readme_path}")
        return False

def main():
    changed = False
    for rd in TARGETS:
        if rd.exists():
            changed |= inject_once(rd, SNIPPET)
    if not changed:
        print("[inject] nothing changed (check anchors or snippet exists).")

if __name__ == "__main__":
    sys.exit(main())
