from __future__ import annotations
import argparse
import logging
from src.models.baseline import train

lg = logging.getLogger("cli.train_baseline")

def main():
    ap = argparse.ArgumentParser(description="CLI: train baseline logistic regression.")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-features", type=int, default=None)
    args = ap.parse_args()

    model_path, report_path = train(horizon=args.horizon, seed=args.seed, max_features=args.max_features)
    print(model_path)
    print(report_path)

if __name__ == "__main__":
    main()
