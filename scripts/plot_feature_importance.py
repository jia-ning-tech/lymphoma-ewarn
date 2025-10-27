#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
REP  = ROOT / "outputs" / "reports"
FIG  = ROOT / "outputs" / "figures"
DOC  = ROOT / "outputs" / "docs"
FIG.mkdir(parents=True, exist_ok=True)
DOC.mkdir(parents=True, exist_ok=True)



def ensure_shap_csv(h:int, split:str) -> Path|None:
    shap_csv = REP / f"shap_values_top_h{h}_{split}.csv"  # 与 src/cli/shap_explain.py 的输出保持一致
    if shap_csv.exists():
        return shap_csv
    # 自动调用你现有的 CLI 生成
    try:
        cmd = [sys.executable, "-m", "src.cli.shap_explain", "--horizon", str(h), "--split", split, "--top_n", "200", "--top_k_individual", "5"]
        print("[plot_fi] SHAP csv not found, running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return shap_csv if shap_csv.exists() else None
    except Exception as e:
        print("[plot_fi] fail to auto-generate SHAP:", e)
        return None

def plot_forest(df, h:int, split:str, topn:int=20, outfile:Path=Path("out.png")):
    top = df.head(topn).copy()
    # 只画置换重要度（errorbar），SHAP 列存在就叠一条细柱子；否则不画 SHAP
    fig, ax = plt.subplots(figsize=(8, 8))  # 更窄的横轴
    y = range(len(top))[::-1]
    ax.errorbar(top["perm_mean"].values[:topn], y,
                xerr=top["perm_std"].values[:topn],
                fmt="o", capsize=3, label="Permutation (mean±std)")

    if "shap_mean_abs" in top.columns and top["shap_mean_abs"].max() > 0:
        ax.barh(y, top["shap_mean_abs"].values[:topn], alpha=0.4, label="SHAP mean(|value|)")

    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"].values[:topn])
    ax.invert_yaxis()
    xmax = float(top["perm_mean"].max() * 1.1) if top["perm_mean"].max() > 0 else 0.01
    ax.set_xlim(0, xmax)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_xlabel("importance")
    ax.set_title(f"Feature importance (h={h}, {split})")
    ax.legend(loc="lower right")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close()

def _pretty_name(s: str) -> str:
    # 轻度清理: 连续双下划线 -> 单下划线；把一些窗口后缀更可读
    s = s.replace("__", "_")
    s = s.replace("_24h_", " 24h ")
    s = s.replace("_6h_",  " 6h ")
    s = s.replace("_mean", " mean")
    s = s.replace("_min",  " min")
    s = s.replace("_max",  " max")
    s = s.replace("_std",  " std")
    s = s.replace("_last", " last")
    s = s.replace("_count"," count")
    return s

def load_imp(h: int, split: str) -> pd.DataFrame:
    perm_csv = REP / f"fi_perm_h{h}_{split}.csv"
    shap_csv = REP / f"fi_shap_h{h}_{split}.csv"  # 可能不存在

    if not perm_csv.exists():
        raise FileNotFoundError(f"missing {perm_csv}")

    dfp = pd.read_csv(perm_csv)

    # --- 兼容各种置换重要度列名 ---
    # 允许的候选：("mean","std"), ("perm_mean","perm_std"),
    # ("delta_auc_mean","delta_auc_std"), ("importance_mean","importance_std")
    cand_mean = [c for c in ["perm_mean","mean","delta_auc_mean","importance_mean"] if c in dfp.columns]
    cand_std  = [c for c in ["perm_std","std","delta_auc_std","importance_std"]    if c in dfp.columns]
    feat_col  = "feature" if "feature" in dfp.columns else None
    if feat_col is None:
        # 兜底：有时列名可能是 'feat' 或 'name'
        for c in ["feat","name","variable"]:
            if c in dfp.columns:
                feat_col = c
                break
    if feat_col is None or not cand_mean or not cand_std:
        raise KeyError(
            f"Unrecognized columns in {perm_csv}.\n"
            f"Got: {dfp.columns.tolist()}\n"
            f"Expect a feature column and one mean/std pair."
        )

    dfp = dfp.rename(columns={
        feat_col: "feature",
        cand_mean[0]: "perm_mean",
        cand_std[0]:  "perm_std",
    })[["feature","perm_mean","perm_std"]]

     # --- SHAP 兼容（可无，若无则自动生成） ---
    shap_csv = ensure_shap_csv(h, split)
    dfs = None
    if shap_csv and shap_csv.exists():
        dfs_raw = pd.read_csv(shap_csv)
        shap_cands = [c for c in ["mean_abs_shap","shap_mean_abs","shap_mean","mean_abs"] if c in dfs_raw.columns]
        feat2 = "feature" if "feature" in dfs_raw.columns else None
        if feat2 is None:
            for c in ["feat","name","variable"]:
                if c in dfs_raw.columns:
                    feat2 = c; break
        if feat2 and shap_cands:
            dfs = dfs_raw.rename(columns={feat2:"feature", shap_cands[0]:"shap_mean_abs"})[["feature","shap_mean_abs"]]

    if dfs is not None:
        df = dfp.merge(dfs, on="feature", how="left")
    else:
        df = dfp.copy()
        df["shap_mean_abs"] = 0.0

    # 数值化 + 排序
    df["perm_mean"] = pd.to_numeric(df["perm_mean"], errors="coerce").fillna(0.0)
    df["perm_std"]  = pd.to_numeric(df["perm_std"],  errors="coerce").fillna(0.0)
    df["shap_mean_abs"] = pd.to_numeric(df["shap_mean_abs"], errors="coerce").fillna(0.0)
    df = df.sort_values("perm_mean", ascending=False).reset_index(drop=True)
    return df


def save_table_md(df: pd.DataFrame, h: int, split: str, topn: int = 20) -> Path:
    out = DOC / f"fi_top_table_h{h}_{split}.md"
    sub = df.head(topn).copy()
    sub["Feature"] = sub["feature"]
    sub["SHAP mean(|v|)"] = sub["shap_mean_abs"].fillna(0).map(lambda x: f"{x:.5f}")
    sub["Perm mean"]      = sub["perm_mean"].map(lambda x: f"{x:.5f}")
    sub["Perm std"]       = sub["perm_std"].map(lambda x: f"{x:.5f}")

    # 注意：Markdown 表格前后**留空行**，且每行**不带任何前导空格**
    lines = []
    lines.append("")  # 前空行
    lines.append("| Feature | SHAP mean(|v|) | Perm mean | Perm std |")
    lines.append("|---|---:|---:|---:|")
    for _, r in sub.iterrows():
        lines.append(f"| `{r['Feature']}` | {r['SHAP mean(|v|)']} | {r['Perm mean']} | {r['Perm std']} |")
    lines.append("")  # 后空行
    out.write_text("\n".join(lines), encoding="utf-8")
    return out

def plot_one(df: pd.DataFrame, h: int, split: str, topn: int = 20) -> Path:
    sub = df.head(topn).copy()
    # 美化显示名，仅用于图（表格仍保留原始特征名以便复现）
    sub["pretty"] = sub["feature"].map(_pretty_name)

    y_labels = sub["pretty"].iloc[::-1]  # 从下到上显示 TopN
    x_perm   = sub["perm_mean"].iloc[::-1].to_numpy()
    x_std    = sub["perm_std"].iloc[::-1].to_numpy()
    x_shap   = sub["shap_mean_abs"].iloc[::-1].to_numpy()

    xmax = float(np.nanmax([x_perm.max() if len(x_perm) else 0.0,
                            x_shap.max() if len(x_shap) else 0.0]))
    if xmax <= 0:
        xmax = 1e-3
    xmax *= 1.25  # 自适应上限

    plt.figure(figsize=(10, max(6, 0.45*len(sub)+2)))

    # 画置换重要度（点+误差棒）
    y = np.arange(len(y_labels))
    plt.errorbar(x_perm, y, xerr=x_std, fmt="o", capsize=3, label="Permutation (mean±std)")
    # 画 SHAP（若为 0 基本看不见，但不影响）
    plt.scatter(x_shap, y, marker="s", s=28, label="SHAP mean(|value|)")

    # 在点右侧标注数值（保留三位小数）
    for xi, yi in zip(x_perm, y):
        plt.text(xi + xmax*0.01, yi, f"{xi:.3f}", va="center", fontsize=9)

    plt.yticks(y, y_labels)
    plt.xlabel("importance")
    plt.xlim(0, xmax)
    plt.title(f"Feature importance (h={h}, {split})")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(axis="x", alpha=0.15)
    plt.tight_layout()

    out_png = FIG / f"fi_forest_h{h}_{split}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", "-H", type=int, required=True, choices=[24,48])
    ap.add_argument("--split", "-S", type=str, required=True, choices=["val","test"])
    ap.add_argument("--topn", type=int, default=20)
    args = ap.parse_args()

    df = load_imp(args.horizon, args.split)
    fig = plot_one(df, args.horizon, args.split, topn=args.topn)
    md  = save_table_md(df, args.horizon, args.split, topn=args.topn)
    print(f"[plot_fi] saved figure -> {fig}")
    print(f"[plot_fi] saved table  -> {md}")

if __name__ == "__main__":
    main()
