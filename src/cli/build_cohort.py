from __future__ import annotations

import os
import sys
import argparse
import pandas as pd

from ..config import get_cfg
from ..utils.log import get_logger, heartbeat
from ..cohort.build_cohort import build_and_save_cohort


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m src.cli.build_cohort",
        description="Build and save the lymphoma ICU cohort parquet (adult, first ICU, LOS>=24h, lymphoma ICD).",
    )
    p.add_argument("--out", type=str, default=None,
                   help="Override output path (default: <cfg.paths.interim>/cohort.parquet)")
    p.add_argument("--if-exists", choices=["overwrite", "skip", "error"], default="overwrite",
                   help="Behavior if output file already exists (default: overwrite)")
    p.add_argument("--head", type=int, default=0,
                   help="After building (or skipping), show the first N rows (0 to skip).")
    p.add_argument("--stats", action="store_true",
                   help="After building (or skipping), print cohort basic stats (n, age/los summary).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = get_cfg()
    lg = get_logger("cli.build_cohort")

    # Decide output path
    out_default = os.path.join(cfg.paths.interim, "cohort.parquet")
    out_path = args.out or out_default

    # Existence policy
    do_build = True
    if os.path.exists(out_path):
        if args.if_exists == "skip":
            lg.info(f"Output already exists, skipping build: {out_path}")
            do_build = False
        elif args.if_exists == "error":
            lg.error(f"Output already exists: {out_path}")
            return 1
        else:
            lg.info(f"Output will be overwritten: {out_path}")
    else:
        # when user gave a custom output path that doesn't exist yet, we still build into default then copy.
        pass

    # Build cohort (or skip)
    if do_build:
        with heartbeat(lg, secs=int(cfg.logging.heartbeat_secs), note="build_cohort"):
            built_path = build_and_save_cohort()
            # 如用户指定 --out 且不同于默认路径，则额外复制一份
            if args.out and os.path.abspath(args.out) != os.path.abspath(built_path):
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                import shutil
                shutil.copy2(built_path, args.out)
                out_path = args.out
                lg.info(f"Copied cohort to: {out_path}")
            else:
                out_path = built_path

    # Optional preview
    if args.head and args.head > 0:
        lg.info(f"Showing first {args.head} rows from {out_path} ...")
        df = pd.read_parquet(out_path)
        print(df.head(args.head).to_string(index=False))

    # Optional basic stats
    if args.stats:
        df = pd.read_parquet(out_path, columns=["age", "los_hours", "gender", "first_careunit"])
        n = len(df)
        g = df["gender"].value_counts(dropna=False).to_dict()
        unit = df["first_careunit"].value_counts(dropna=False).head(8).to_dict()
        age_desc = df["age"].describe(percentiles=[0.1,0.25,0.5,0.75,0.9]).round(2).to_dict()
        los_desc = df["los_hours"].describe(percentiles=[0.1,0.25,0.5,0.75,0.9]).round(2).to_dict()
        lg.info(f"[stats] n={n}")
        lg.info(f"[stats] gender={g}")
        lg.info(f"[stats] first_careunit(top)={unit}")
        lg.info(f"[stats] age={age_desc}")
        lg.info(f"[stats] los_hours={los_desc}")

    lg.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
