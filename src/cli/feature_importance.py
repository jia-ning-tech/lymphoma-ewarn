from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "reports"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--topk", type=int, default=50)
    args=ap.parse_args()

    bundle=joblib.load(ROOT/"outputs"/"models"/f"baseline_h{args.horizon}.joblib")
    pipe=bundle["pipeline"]
    feats=bundle["features"]
    clf = pipe.named_steps["clf"]
    imp = getattr(clf, "feature_importances_", None)
    if imp is None:
        print("model has no feature_importances_")
        return
    df = pd.DataFrame({"feature":feats, "importance":imp})
    df = df.sort_values("importance", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"feature_importance_h{args.horizon}.csv"
    df.to_csv(p, index=False)
    print(str(p))
    print(df.head(args.topk).to_string(index=False))
if __name__=="__main__":
    main()
