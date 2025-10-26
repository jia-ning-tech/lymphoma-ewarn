from __future__ import annotations
import argparse, logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.dca import decision_curve

ROOT = Path(__file__).resolve().parents[2]
DATAI = ROOT / "data_interim"
PREDS = ROOT / "outputs" / "preds"
REPS  = ROOT / "outputs" / "reports"
FIGS  = ROOT / "outputs" / "figures"
REPS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

lg = logging.getLogger("cli.dca")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def load_raw(h: int) -> pd.DataFrame:
    p = DATAI / f"trainset_h{h}.parquet"
    assert p.exists(), f"not found raw: {p}"
    df = pd.read_parquet(p)
    # 兼容各种命名，统一成 label
    for c in ["label","y","target","outcome","event","label_window","label_stay"]:
        if c in df.columns:
            if c != "label":
                df = df.rename(columns={c: "label"})
            break
    assert "label" in df.columns, "raw must contain label column"
    df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")
    df["label"] = df["label"].astype(int)
    return df[["stay_id","index_time","label"]]

def load_preds(h: int, split: str, calibrated: Optional[str]) -> pd.DataFrame:
    if calibrated:
        p = PREDS / f"preds_h{h}_{split}_cal_{calibrated}.parquet"
        assert p.exists(), f"not found calibrated preds: {p}"
        df = pd.read_parquet(p)
        if "prob" in df.columns:
            df = df.drop(columns=["prob"])
        assert "prob_cal" in df.columns, "calibrated file must contain prob_cal"
        df = df.rename(columns={"prob_cal":"prob"})
    else:
        p = PREDS / f"preds_h{h}_{split}.parquet"
        assert p.exists(), f"not found preds: {p}"
        df = pd.read_parquet(p)
        assert "prob" in df.columns, "preds must contain prob"
    df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")
    return df[["stay_id","index_time","prob"]]

def main():
    ap = argparse.ArgumentParser(description="Decision Curve Analysis (DCA) plotter")
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--split", type=str, default="test", choices=["val","test"])
    ap.add_argument("--calibrated", type=str, default=None, choices=[None,"isotonic","sigmoid"])
    ap.add_argument("--n-thr", type=int, default=99, help="number of threshold points")
    ap.add_argument("--thr-min", type=float, default=0.01)
    ap.add_argument("--thr-max", type=float, default=0.99)
    ap.add_argument("--per-100", action="store_true", help="net benefit per 100 patients")
    args = ap.parse_args()

    h = args.horizon
    split = args.split
    cal = args.calibrated

    raw = load_raw(h)
    preds = load_preds(h, split, cal)
    df = raw.merge(preds, on=["stay_id","index_time"], how="left")
    n_merged = len(df)
    n_nan = int(df["prob"].isna().sum())
    df = df.dropna(subset=["prob"])
    lg.info("merged rows=%d, dropped NaN prob=%d, final for DCA=%d", n_merged, n_nan, len(df))

    y = df["label"].astype(int).to_numpy()
    p = df["prob"].astype(float).to_numpy()
    thr = np.linspace(args.thr_min, args.thr_max, args.n_thr)
    dca = decision_curve(y, p, thresholds=thr, per_100=args.per_100)
    dca_df = dca.to_dataframe()

    tag = f"h{h}_{split}" + (f"_cal_{cal}" if cal else "")
    csv_out = REPS / f"dca_{tag}.csv"
    png_out = FIGS / f"dca_{tag}.png"
    dca_df.to_csv(csv_out, index=False)

    plt.figure(figsize=(7.5,5.5))
    plt.plot(dca_df["threshold"], dca_df["nb_model"], label="Model", linewidth=2)
    plt.plot(dca_df["threshold"], dca_df["nb_all"],   label="Treat-all", linestyle="--")
    plt.plot(dca_df["threshold"], dca_df["nb_none"],  label="Treat-none", linestyle=":")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit" + (" (per 100 patients)" if args.per_100 else ""))
    plt.title(f"DCA (h={h}h, {split}{' | '+cal if cal else ''})  |  prevalence={dca.prevalence:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_out, dpi=180)
    plt.close()

    print(csv_out)
    print(png_out)

if __name__ == "__main__":
    main()
