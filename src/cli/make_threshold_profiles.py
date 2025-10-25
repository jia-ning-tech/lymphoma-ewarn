from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPS = ROOT / "outputs" / "reports"

def pick_thr_for_rate(df: pd.DataFrame, target: float) -> float:
    # df: columns = [threshold, alert_rate, precision, recall, f1, ...]
    # 选 alert_rate 最接近 target 的阈值；并在并列时选更高 precision 的
    df["gap"] = (df["alert_rate"] - target).abs()
    df = df.sort_values(["gap", "precision"], ascending=[True, False])
    return float(df.iloc[0]["threshold"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--release", required=True, help="outputs/release/h48_v0.x")
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--rates", default="0.05,0.10,0.15", help="comma rates e.g. 0.05,0.10,0.15")
    args = ap.parse_args()

    sweep_csv = REPS / f"thr_sweep_h{args.horizon}_{args.split}.csv"
    assert sweep_csv.exists(), f"not found: {sweep_csv}"
    df = pd.read_csv(sweep_csv)

    rate_list = [float(x) for x in args.rates.split(",")]
    profiles = {}
    for r in rate_list:
        thr = pick_thr_for_rate(df, r)
        row = df.loc[df["threshold"].round(10) == round(thr,10)].iloc[0].to_dict()
        profiles[f"{int(r*100)}p"] = {
            "threshold_window": thr,
            "alert_rate": float(row["alert_rate"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
        }

    rdir = Path(args.release)
    meta_p = rdir / "release.json"
    meta = json.loads(meta_p.read_text())
    meta["threshold_profiles"] = profiles
    meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(profiles, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
