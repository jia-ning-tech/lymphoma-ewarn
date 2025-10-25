import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_ablation(h: int, reports_dir: Path) -> pd.DataFrame:
    p = reports_dir / f"ablation_h{h}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    df = pd.read_csv(p)
    # 兼容不同导出字段名
    rename_map = {
        'AUROC_mean':'auroc_mean', 'AUROC_std':'auroc_std',
        'AP_mean':'ap_mean', 'AP_std':'ap_std',
        'n_features':'n_features', 'setting':'setting', 'group':'group'
    }
    for k, v in list(rename_map.items()):
        if k in df.columns:
            df = df.rename(columns={k: v})
    df.columns = [c.strip() for c in df.columns]

    need = {'setting','group','n_features','auroc_mean','auroc_std','ap_mean','ap_std'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Columns={df.columns.tolist()}")

    # 缺失 std 视为 0，防止 errorbar 报错
    for c in ['auroc_std','ap_std']:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # 统一字符串类型，避免 group 为 NaN
    df['setting'] = df['setting'].astype(str)
    df['group'] = df['group'].fillna('-').astype(str)
    df['horizon'] = h
    return df

def order_rows(df: pd.DataFrame) -> pd.DataFrame:
    # 展示顺序：baseline_all -> drop-one(按 group 名字排序) -> keep-only(按 group 名字排序)
    order_map = {'baseline_all': 0, 'drop-one': 1, 'keep-only': 2}
    df = df.copy()
    df['__sort0'] = df['setting'].map(order_map).fillna(9).astype(int)
    df['__sort1'] = np.where(df['setting'].eq('baseline_all'), 0, 1)  # 仅用于把 baseline 固定在前
    df['__sort2'] = df['group'].astype(str)
    df = df.sort_values(['__sort0', '__sort2', '__sort1']).reset_index(drop=True)
    return df.drop(columns=['__sort0','__sort1','__sort2'])

def bar_with_error(df: pd.DataFrame, metric: str, out_png: Path, title: str):
    mean_col = f"{metric}_mean"
    std_col  = f"{metric}_std"

    labels, colors, means, errs = [], [], [], []
    for _, r in df.iterrows():
        lab = f"{r['setting']}\n[{r['group']}]" if r['group'] != '-' else r['setting']
        labels.append(lab)
        means.append(float(r[mean_col]))
        errs.append(float(r[std_col]))
        if r['setting'] == 'baseline_all':
            colors.append('#1f77b4')  # 蓝
        elif r['setting'] == 'drop-one':
            colors.append('#d62728')  # 红
        else:
            colors.append('#2ca02c')  # 绿

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*0.6), 5))
    ax.bar(x, means, yerr=errs, capsize=4, alpha=0.9, edgecolor='black', linewidth=0.6, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def forest_plot(df: pd.DataFrame, metric: str, out_png: Path, title: str):
    mean_col = f"{metric}_mean"
    std_col  = f"{metric}_std"

    labels, means, errs, colors = [], [], [], []
    for _, r in df.iterrows():
        lab = f"{r['setting']} [{r['group']}]" if r['group'] != '-' else r['setting']
        labels.append(lab)
        means.append(float(r[mean_col]))
        errs.append(float(r[std_col]))
        if r['setting'] == 'baseline_all':
            colors.append('#1f77b4')
        elif r['setting'] == 'drop-one':
            colors.append('#d62728')
        else:
            colors.append('#2ca02c')

    y = np.arange(len(labels))[::-1]  # 反向让基线在最上
    means = np.array(means)
    errs  = np.array(errs)

    fig, ax = plt.subplots(figsize=(8, max(5, len(labels)*0.5)))
    # 误差棒点
    ax.errorbar(means, y, xerr=errs, fmt='o', ecolor='gray', capsize=4, color='black')
    # 背景色条表示不同设置
    for i, (yy, c) in enumerate(zip(y, colors)):
        ax.plot([means[i]-errs[i], means[i]+errs[i]], [yy, yy], '-', color=c, lw=6, alpha=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(metric.upper())
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=str, default="24,48", help="comma-separated, e.g., 24,48")
    ap.add_argument("--reports_dir", type=str, default="outputs/reports")
    ap.add_argument("--fig_dir", type=str, default="outputs/figures")
    ap.add_argument("--metrics", type=str, default="auroc,ap", help="choose from {auroc,ap}")
    args = ap.parse_args()

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    reports_dir = Path(args.reports_dir)
    fig_dir = Path(args.fig_dir)

    for h in horizons:
        try:
            df = load_ablation(h, reports_dir)
        except FileNotFoundError:
            # 没有该 horizon 的文件就跳过
            continue

        # 只保留 baseline_all、drop-one、keep-only 三类
        df = df[df['setting'].isin(['baseline_all','drop-one','keep-only'])].copy()
        df = order_rows(df)

        # 保存整洁表，便于 README 使用
        df['label_for_table'] = df.apply(
            lambda r: f"{r['setting']} [{r['group']}] (n_feat={int(r['n_features'])})" if r['group']!='-' else f"{r['setting']} (n_feat={int(r['n_features'])})",
            axis=1
        )
        tidy_csv = reports_dir / f"ablation_h{h}_tidy_for_readme.csv"
        df[['horizon','setting','group','n_features','auroc_mean','auroc_std','ap_mean','ap_std','label_for_table']].to_csv(tidy_csv, index=False)

        for metric in metrics:
            bar_png = fig_dir / f"ablation_h{h}_{metric}_bar.png"
            bar_with_error(df, metric, bar_png, title=f"Ablation ({metric.upper()}) — horizon={h}h")

            forest_png = fig_dir / f"ablation_h{h}_{metric}_forest.png"
            forest_plot(df, metric, forest_png, title=f"Ablation ({metric.upper()}) Forest — horizon={h}h")

            print(str(bar_png))
            print(str(forest_png))

if __name__ == "__main__":
    main()
