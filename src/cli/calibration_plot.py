from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
PREDS = ROOT / "outputs" / "preds"
FIGS  = ROOT / "outputs" / "figures"
REPS  = ROOT / "outputs" / "reports"
FIGS.mkdir(parents=True, exist_ok=True)
REPS.mkdir(parents=True, exist_ok=True)

lg = logging.getLogger("cli.calibration_plot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def load_preds(h:int, split:str)->pd.DataFrame:
    p = PREDS / f"preds_h{h}_{split}.parquet"
    assert p.exists(), f"not found preds file: {p}\n请先运行 train_val_test 产出 preds_h{h}_{split}.parquet"
    df = pd.read_parquet(p)
    # 需要列：stay_id, index_time, label, prob
    need = {"stay_id","index_time","label","prob"}
    miss = need - set(df.columns)
    assert not miss, f"missing columns in preds: {miss}"
    df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")
    df["label"] = df["label"].astype(int)
    return df

def to_stay_level(df:pd.DataFrame)->pd.DataFrame:
    g = df.groupby("stay_id", as_index=False).agg(
        label=("label","max"),
        prob=("prob","max")
    )
    return g.rename(columns={"label":"stay_label","prob":"stay_prob"})

def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins:int=20, strategy:str="uniform")->float:
    # 计算 Expected Calibration Error
    if strategy == "quantile":
        # 分位数分箱
        qs = np.linspace(0,1,n_bins+1)
        bins = np.unique(np.quantile(y_prob, qs))
        # 若重复导致箱数减少，退回 uniform
        if len(bins) - 1 < n_bins:
            bins = np.linspace(0,1,n_bins+1)
    else:
        bins = np.linspace(0,1,n_bins+1)

    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, len(bins)-2)

    ece = 0.0
    N = len(y_true)
    for b in range(len(bins)-1):
        mask = (idx == b)
        if not np.any(mask): 
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

def plot_reliability(y_true:np.ndarray, y_prob:np.ndarray, h:int, split:str, tag:str, n_bins:int, strategy:str):
    # sklearn calibration curve points
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)

    # CSV 保存
    out_csv = REPS / f"calibration_points_h{h}_{split}_{tag}.csv"
    pd.DataFrame({"prob_pred":prob_pred, "prob_true":prob_true}).to_csv(out_csv, index=False)

    # 可靠性曲线
    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1], linestyle="--", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Reliability curve (h={h}, {split}, {tag})")
    plt.legend(loc="best")
    out_png = FIGS / f"calibration_h{h}_{split}_{tag}.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

    # 直方图（置信度分布）
    plt.figure(figsize=(6,4))
    plt.hist(y_prob, bins=20, edgecolor="k")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title(f"Confidence histogram (h={h}, {split}, {tag})")
    out_hist = FIGS / f"calibration_hist_h{h}_{split}_{tag}.png"
    plt.tight_layout(); plt.savefig(out_hist, dpi=200); plt.close()

    return str(out_csv), str(out_png), str(out_hist)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--split", choices=["val","test"], default="test")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--strategy", choices=["uniform","quantile"], default="uniform")
    ap.add_argument("--stay_level", action="store_true", help="若指定，则以住院级（max prob, max label）做校准")
    args = ap.parse_args()

    df = load_preds(args.horizon, args.split)

    if args.stay_level:
        tag = "stay"
        sdf = to_stay_level(df)
        y = sdf["stay_label"].values.astype(int)
        p = sdf["stay_prob"].values.astype(float)
    else:
        tag = "window"
        y = df["label"].values.astype(int)
        p = df["prob"].values.astype(float)

    brier = brier_score_loss(y, p)
    ece  = ece_score(y, p, n_bins=args.bins, strategy=args.strategy)

    out_csv, out_png, out_hist = plot_reliability(y, p, args.horizon, args.split, tag, args.bins, args.strategy)

    summary = {
        "horizon": args.horizon,
        "split": args.split,
        "level": tag,
        "n": int(len(y)),
        "pos_rate": float(np.mean(y)),
        "brier": float(brier),
        "ece": float(ece),
        "bins": int(args.bins),
        "strategy": args.strategy,
        "curve_csv": out_csv,
        "curve_png": out_png,
        "hist_png": out_hist
    }
    out_json = REPS / f"calibration_summary_h{args.horizon}_{args.split}_{tag}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    lg.info(json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
