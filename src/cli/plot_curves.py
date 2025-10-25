from __future__ import annotations
import argparse, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PREDS = ROOT / "outputs" / "preds"
FIGS  = ROOT / "outputs" / "figures"
REPS  = ROOT / "outputs" / "reports"

lg = logging.getLogger("cli.plot_curves")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def load_preds(h:int, split:str)->pd.DataFrame:
    p = PREDS / f"preds_h{h}_{split}.parquet"
    assert p.exists(), f"not found preds: {p}"
    df = pd.read_parquet(p)
    assert {"label","prob"} <= set(df.columns), "pred file must have columns: label, prob"
    return df

def save_curve_csv(x, y, out_csv:Path, x_name:str, y_name:str):
    df = pd.DataFrame({x_name: x, y_name: y})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv

def plot_and_save(x, y, xlabel:str, ylabel:str, title:str, out_png:Path, auc_text:str|None=None):
    plt.figure(figsize=(6,5), dpi=150)
    plt.plot(x, y, lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title if auc_text is None else f"{title}\n{auc_text}")
    plt.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return out_png

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--split", choices=["train","val","test"], default="test")
    args = ap.parse_args()

    df = load_preds(args.horizon, args.split)
    y = df["label"].astype(int).values
    p = df["prob"].astype(float).values

    # ROC
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    roc_csv = REPS / f"roc_points_h{args.horizon}_{args.split}.csv"
    roc_png = FIGS / f"roc_h{args.horizon}_{args.split}.png"
    save_curve_csv(fpr, tpr, roc_csv, "fpr", "tpr")
    plot_and_save(fpr, tpr, "False Positive Rate", "True Positive Rate",
                  f"ROC (h={args.horizon}, {args.split})", roc_png, auc_text=f"AUROC={auc:.3f}")
    lg.info(f"ROC saved: csv={roc_csv} png={roc_png} | AUROC={auc:.4f}")

    # PR
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    pr_csv = REPS / f"pr_points_h{args.horizon}_{args.split}.csv"
    pr_png = FIGS / f"pr_h{args.horizon}_{args.split}.png"
    save_curve_csv(rec, prec, pr_csv, "recall", "precision")
    plot_and_save(rec, prec, "Recall", "Precision",
                  f"PR (h={args.horizon}, {args.split})", pr_png, auc_text=f"AP={ap:.3f}")
    lg.info(f"PR saved: csv={pr_csv} png={pr_png} | AP={ap:.4f}")

if __name__ == "__main__":
    main()
