from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import joblib
import pandas as pd
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
MODELS = OUT / "models"
REPORTS = OUT / "reports"
RELEASE = OUT / "release"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=48, choices=[24,48])
    ap.add_argument("--threshold", type=float, required=True)  # 例如 0.24142504229084366
    ap.add_argument("--name", default="v0.1")
    ap.add_argument("--refractory_hours", type=int, default=24, help="同一住院重复报警抑制窗口")
    ap.add_argument("--daily_alert_cap", type=int, default=999999, help="每日最大报警条数（按需要限制）")
    args = ap.parse_args()

    # 读取模型
    model_p = MODELS / f"grouped_tvt_h{args.horizon}.joblib"
    if not model_p.exists():
        # 回退到 baseline 模型也行
        model_p = MODELS / f"baseline_h{args.horizon}.joblib"
    assert model_p.exists(), f"model not found: {model_p}"

    bundle = joblib.load(model_p)
    feats = bundle["features"]

    # 读取评估与提前量（如果存在就采集）
    rep_test = REPORTS / f"report_h{args.horizon}_test.json"
    test_metrics = None
    if rep_test.exists():
        test_metrics = json.loads(rep_test.read_text())

    # 提前量统计（用你刚刚算好的那个阈值文件，可能不存在则跳过）
    lead_csv = list(REPORTS.glob(f"leadtime_h{args.horizon}_test_thr*.csv"))
    lead_stats = None
    if lead_csv:
        # 取最新
        lead_stats = pd.read_csv(sorted(lead_csv)[-1], index_col=0).to_dict()["hours"]

    # 写入 release 目录
    rdir = RELEASE / f"h{args.horizon}_{args.name}"
    rdir.mkdir(parents=True, exist_ok=True)
    # 复制模型
    model_out = rdir / "model.joblib"
    shutil.copy2(model_p, model_out)

    # 写 config.yaml（最小可用：供推理/服务读取）
    cfg = {
        "model_path": str(model_out),
        "horizon_hours": args.horizon,
        "threshold_window": float(args.threshold),
        "refractory_hours": args.refractory_hours,
        "daily_alert_cap": args.daily_alert_cap,
        "feature_count": len(feats),
    }
    (rdir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    # 写 release.json：记录元数据与度量
    meta = {
        "release_name": args.name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_model": str(model_p),
        "features": feats,
        "test_metrics": test_metrics,
        "leadtime_stats_hours": lead_stats,
    }
    (rdir / "release.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(str(rdir))

if __name__ == "__main__":
    main()
