# src/cli/dca_plot.py
from __future__ import annotations
import argparse, logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.eval.dca import decision_curve, plot_dca

lg = logging.getLogger("cli.dca")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

ROOT   = Path(__file__).resolve().parents[2]
DATAI  = ROOT / "data_interim"
PREDS  = ROOT / "outputs" / "preds"
FIGDIR = ROOT / "outputs" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

REQ_PRED_COLS = {"stay_id","index_time","prob"}
CAND_LABELS   = ["label","y","target","outcome","event","label_window","label_stay"]

def _coerce_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 df 中第一个命中的候选结局列统一重命名为 'label'，类型转 int。
    若找不到，抛出包含可用列名的报错。
    """
    for c in CAND_LABELS:
        if c in df.columns:
            if c != "label":
                df = df.rename(columns={c: "label"})
            df["label"] = df["label"].astype(int)
            return df
    raise KeyError(f"could not find outcome column among {CAND_LABELS}; got columns={df.columns.tolist()}")

def load_raw(h: int) -> pd.DataFrame:
    p = DATAI / f"trainset_h{h}.parquet"
    assert p.exists(), f"not found raw: {p}"
    df = pd.read_parquet(p)
    need = [c for c in ["stay_id","index_time"] if c in df.columns]
    if len(need) < 2:
        raise KeyError(f"{p} missing stay_id or index_time; got {df.columns.tolist()}")
    df = df[need + [c for c in CAND_LABELS if c in df.columns]]
    df = df.loc[:, ~df.columns.duplicated()]
    df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")
    df = _coerce_label(df)
    return df

def load_preds(h: int, split: str, calibrated: str|None) -> pd.DataFrame:
    if calibrated:
        p = PREDS / f"preds_h{h}_{split}_cal_{calibrated}.parquet"
        assert p.exists(), f"not found calibrated preds: {p}"
        df = pd.read_parquet(p)
        if "prob" in df.columns:
            df = df.drop(columns=["prob"])
        assert "prob_cal" in df.columns, "calibrated preds must contain prob_cal"
        df = df.rename(columns={"prob_cal":"prob"})
    else:
        p = PREDS / f"preds_h{h}_{split}.parquet"
        assert p.exists(), f"not found preds: {p}"
        df = pd.read_parquet(p)
        assert "prob" in df.columns, f"'prob' not found in {p}; got {df.columns.tolist()}"
    keep = [c for c in ["stay_id","index_time","prob"] if c in df.columns]
    df = df[keep]
    df = df.loc[:, ~df.columns.duplicated()]
    df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")
    return df

def main():
    ap = argparse.ArgumentParser(description="Decision Curve Analysis (DCA) plot")
    ap.add_argument("--horizon", required=True, type=int, choices=[24,48])
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--calibrated", default=None, choices=[None,"isotonic","sigmoid"],
                    help="use calibrated probs if provided")
    ap.add_argument("--n-thr", type=int, default=99, help="number of threshold points in (0,1)")
    ap.add_argument("--thr-min", type=float, default=0.01, dest="thr_min")
    ap.add_argument("--thr-max", type=float, default=0.99, dest="thr_max")
    ap.add_argument("--per-100", action="store_true", default=True, help="net benefit per 100 (default True)")
    ap.add_argument("--no-per-100", dest="per_100", action="store_false")
    args = ap.parse_args()

    # 载入并合并
    raw   = load_raw(args.horizon)          # 含 'label'
    preds = load_preds(args.horizon, args.split, args.calibrated)  # 含 'prob'
    df = raw.merge(preds, on=["stay_id","index_time"], how="left")
    n_all = len(df)
    n_nan = int(df["prob"].isna().sum())
    df = df.dropna(subset=["prob"])
    lg.info("merged rows=%d, dropped NaN prob=%d, final for DCA=%d", n_all, n_nan, len(df))

    # 取结局与概率
    if "label" not in df.columns:
        df = _coerce_label(df)
    y = df["label"].astype(int).to_numpy()
    p = df["prob"].astype(float).to_numpy()

    # 阈值序列
    thr = np.linspace(max(1e-6, args.thr_min), min(1-1e-6, args.thr_max), args.n_thr)
    dca = decision_curve(y, p, thresholds=thr, per_100=args.per_100)

    # 标题 & 输出
    tag = f"h{args.horizon}_{args.split}"
    if args.calibrated:
        tag += f"_cal_{args.calibrated}"
    out_png = FIGDIR / f"dca_{tag}.png"

    # 合理的 y 轴范围；如更希望自适应，可去掉 ylim
    plot_dca(dca, title=tag, outfile=out_png, ylim=(-5, 10))
    lg.info("Saved DCA figure -> %s", out_png)

if __name__ == "__main__":
    main()
