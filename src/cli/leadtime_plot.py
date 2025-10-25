from __future__ import annotations
import argparse, logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REPS = ROOT / "outputs" / "reports"
FIGS = ROOT / "outputs" / "figures"

lg = logging.getLogger("cli.leadtime_plot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def load_details(h:int, split:str, thr:float)->tuple[pd.DataFrame, str]:
    tag = f"h{h}_{split}_thr{thr:.4f}"
    p = REPS / f"leadtime_details_{tag}.parquet"
    assert p.exists(), f"not found: {p}"
    df = pd.read_parquet(p)

    # ---- 兼容不同明细结构，确保有 hours 列 ----
    if "hours" in df.columns:
        pass
    elif "lead_hours" in df.columns:
        df = df.rename(columns={"lead_hours": "hours"})
    else:
        cols = {c.lower(): c for c in df.columns}
        fat = cols.get("first_alert_time")
        fet = cols.get("first_event_time")
        if fat is None or fet is None:
            raise AssertionError("leadtime details must have column 'hours' or both 'first_alert_time' and 'first_event_time'")
        df[fat] = pd.to_datetime(df[fat], errors="coerce")
        df[fet] = pd.to_datetime(df[fet], errors="coerce")
        delta = (df[fet] - df[fat]).dt.total_seconds() / 3600.0
        df["hours"] = np.clip(delta, a_min=0, a_max=None)

    df = df.loc[df["hours"].notna() & (df["hours"] > 0)].copy()
    return df, tag

def plot_hist(df:pd.DataFrame, title:str, out_png:Path):
    plt.figure(figsize=(6,5), dpi=150)
    plt.hist(df["hours"].values, bins=24)
    plt.xlabel("Lead time (hours)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return out_png

def plot_box(df:pd.DataFrame, title:str, out_png:Path):
    plt.figure(figsize=(4,5), dpi=150)
    plt.boxplot(df["hours"].values, vert=True, showfliers=True)
    plt.ylabel("Lead time (hours)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return out_png

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--split", choices=["val","test"], default="test")
    ap.add_argument("--threshold", type=float, required=True)
    args = ap.parse_args()

    df, tag = load_details(args.horizon, args.split, args.threshold)
    title = f"Lead time distribution (h={args.horizon}, {args.split}, thr={args.threshold:.4f})"
    out_hist = FIGS / f"leadtime_hist_{tag}.png"
    out_box  = FIGS / f"leadtime_box_{tag}.png"
    plot_hist(df, title, out_hist)
    plot_box(df, title, out_box)

    desc = df["hours"].describe(percentiles=[.1,.25,.5,.75,.9]).to_dict()
    stats = {k: (float(v) if isinstance(v, (int,float,np.floating)) else v) for k, v in desc.items()}
    lg.info(f"lead-time plots saved: {out_hist}, {out_box} | n={len(df)} | stats={stats}")

if __name__ == "__main__":
    main()
