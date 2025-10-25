from __future__ import annotations
import argparse
import logging
from pathlib import Path

from src.features.aggregate_windows import build_features

lg = logging.getLogger("cli.build_features")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def main():
    ap = argparse.ArgumentParser(description="Aggregate rolling-window features on labels grid.")
    ap.add_argument("--windows", nargs="+", type=int, default=[6, 24],
                    help="rolling windows in hours, e.g., --windows 6 24 48")
    ap.add_argument("--head", type=int, default=5, help="preview head rows after saving")
    ap.add_argument("--stats", action="store_true", help="print simple non-null stats")
    args = ap.parse_args()

    paths = build_features(windows=args.windows, head=args.head, stats=args.stats)
    for p in paths:
        print(p)

if __name__ == "__main__":
    main()
