from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(cmd: list[str]):
    print("+", " ".join(cmd), flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=str, default="24,48")
    ap.add_argument("--splits", type=str, default="val,test")
    ap.add_argument("--per-100", action="store_true")
    ap.add_argument("--n-thr", type=int, default=99)
    ap.add_argument("--thr-min", type=float, default=0.01)
    ap.add_argument("--thr-max", type=float, default=0.99)
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    cals = [None, "isotonic", "sigmoid"]

    base = [sys.executable, "-m", "src.cli.dca_plot",
            "--n-thr", str(args.n_thr),
            "--thr-min", str(args.thr_min),
            "--thr-max", str(args.thr_max)]
    if args.per_100:
        base.append("--per-100")

    for h in horizons:
        for split in splits:
            for cal in cals:
                cmd = base + ["--horizon", str(h), "--split", split]
                if cal is not None:
                    cmd += ["--calibrated", cal]
                run(cmd)

if __name__ == "__main__":
    main()
