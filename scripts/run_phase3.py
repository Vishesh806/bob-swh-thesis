"""Phase 3: persistence, climatology, SARIMAX and Direct Multi-Step LightGBM / CatBoost.

    python scripts/run_phase3.py --buoy BD08
    python scripts/run_phase3.py --buoy BD08 --trials 60 --backends lightgbm catboost xgboost
    python scripts/run_phase3.py --buoy BD08 --target era5_swh --trials 0      # ERA5-only, untuned
    python scripts/run_phase3.py --buoy BD08 --sarimax-select                  # choose (p,d,q) by AIC

Run scripts/build_features.py first.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from bobswh.phase3 import Phase3Runner  # noqa: E402
from bobswh.trees import BACKENDS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buoy", required=True)
    ap.add_argument("--target", default="buoy_hs")
    ap.add_argument("--backends", nargs="+", choices=BACKENDS, default=["lightgbm", "catboost"])
    ap.add_argument("--trials", type=int, default=40, help="Optuna trials per backend (0 = default parameters)")
    ap.add_argument("--timeout-min", type=float, default=None, help="time budget per study, minutes")
    ap.add_argument("--per-horizon", action="store_true", help="tune each horizon separately")
    ap.add_argument("--tune-stride", type=int, default=3)
    ap.add_argument("--sarimax-select", action="store_true")
    ap.add_argument("--no-sarimax", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("optuna").setLevel(logging.WARNING)

    runner = Phase3Runner(
        buoy=args.buoy,
        target_col=args.target,
        backends=tuple(args.backends),
        n_trials=args.trials,
        timeout_s=None if args.timeout_min is None else args.timeout_min * 60,
        per_horizon=args.per_horizon,
        tune_stride=args.tune_stride,
        sarimax_orders=((1, 1, 0), (1, 1, 1), (2, 1, 1), (2, 1, 2), (3, 1, 1), (2, 0, 1)) if args.sarimax_select else None,
        run_sarimax=not args.no_sarimax,
    )
    metrics = runner.run()
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print("\nTest RMSE (m):")
        print(metrics.xs("test", level="split")["rmse"].unstack("horizon_h"))


if __name__ == "__main__":
    main()
