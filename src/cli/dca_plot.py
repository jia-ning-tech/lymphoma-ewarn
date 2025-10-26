# src/cli/dca_plot.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.eval.dca import decision_curve

ROOT = Path(__file__).resolve().parents[2]
DATAI = ROOT / "data_interim"
PREDS = ROOT / "outputs" / "preds"
FIGS  = ROOT / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

def load_raw(h: int) -> pd.DataFrame:
    p = DATAI / f"trainset_h{h}.parquet"
    df = pd.read_parquet(p)[["stay_id","index_time","label"]]
    df["index_time"] = pd.to_datetime(df["index_time"])
    return df

def load_preds(h: int, split: str, calibrated: str|None) -> pd.DataFrame:
    if calibrated:
        p = PREDS / f"preds_h{h}_{split}_cal_{calibrated}.parquet"
        df = pd.read_parquet(p)
        if "prob" in df.columns:
            df = df.drop(columns=["prob"])
        df = df.rename(columns={"prob_cal":"prob"})
    else:
        p = PREDS / f"preds_h{h}_{split}.parquet"
        df = pd.read_parquet(p)
    keep = [c for c in ["stay_id","index_time","prob"] if c in df.columns]
    return df[keep]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, choices=[24,48], required=True)
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--calibrated", default=None, choices=[None,"isotonic","sigmoid"])
    ap.add_argument("--thr-min", type=float, default=1e-4)
    ap.add_argument("--thr-max", type=float, default=0.9999)
    ap.add_argument("--n-thr", type=int, default=200)
    args = ap.parse_args()

    raw = load_raw(args.horizon)
    preds = load_preds(args.horizon, args.split, args.calibrated)
    df = raw.merge(preds, on=["stay_id","index_time"], how="left").dropna(subset=["prob"])

    y = df["label"].astype(int).to_numpy()
    p = df["prob"].astype(float).to_numpy()

    thr = np.linspace(args.thr_min, args.thr_max, args.n_thr)
    dca = decision_curve(y, p, thresholds=thr, per=100)  # 只在这里 *100

    tag = f"h{args.horizon}_{args.split}"
    if args.calibrated:
        tag += f"_cal_{args.calibrated}"
    out = FIGS / f"dca_{tag}.png"

    plt.figure(figsize=(10,6))
    plt.plot(dca.thresholds, dca.nb_model, label="Model")
    plt.plot(dca.thresholds, dca.nb_all, "--", label="Treat-all")
    plt.plot(dca.thresholds, dca.nb_none, ":", label="Treat-none")
    plt.axhline(0, color="k", linewidth=1)
    plt.title(f"DCA (h={args.horizon}h, {args.split} {'| '+args.calibrated if args.calibrated else ''})  |  prevalence={dca.prevalence:.3f}")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit (per 100 patients)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"[cli.dca_plot] saved -> {out}")

if __name__ == "__main__":
    main()
