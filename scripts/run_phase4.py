"""Phase 4: Seq2Seq (Bi)LSTM with attention, peak-weighted loss vs MSE ablation.

    python scripts/run_phase4.py --buoy BD08
    python scripts/run_phase4.py --buoy BD08 --inputs extended --units 128
    python scripts/run_phase4.py --buoy BD08 --target era5_swh --variants peak     # ERA5-only

Run scripts/build_features.py first. A GPU (e.g. Colab) cuts training time several-fold.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from bobswh.phase4 import INPUT_SETS, Phase4Runner  # noqa: E402
from bobswh.seq2seq import Seq2SeqConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buoy", required=True)
    ap.add_argument("--target", default="buoy_hs")
    ap.add_argument("--inputs", choices=list(INPUT_SETS), default="core")
    ap.add_argument("--variants", nargs="+", choices=["peak", "mse"], default=["peak", "mse"])
    ap.add_argument("--units", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1, help="encoder LSTM layers")
    ap.add_argument("--no-attention", action="store_true")
    ap.add_argument("--unidirectional", action="store_true")
    ap.add_argument("--stride", type=int, default=1, help="encoder step spacing in hours")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--patience", type=int, default=10, help="early-stopping patience in epochs")
    ap.add_argument("--gamma", type=float, default=4.0, help="extra under-prediction weight at peaks")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = Seq2SeqConfig(units=args.units, encoder_layers=args.layers, attention=not args.no_attention,
                        bidirectional=not args.unidirectional, epochs=args.epochs, batch_size=args.batch,
                        gamma=args.gamma, patience=args.patience)
    runner = Phase4Runner(buoy=args.buoy, target_col=args.target, input_set=args.inputs,
                          variants=tuple(args.variants), sequence_stride_h=args.stride, model_config=cfg)
    metrics = runner.run()
    cols = ["rmse", "bias", "rmse_peak", "bias_peak", "under_rate_peak"]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print("\nTest scores:")
        print(metrics.xs("test", level="split")[cols])


if __name__ == "__main__":
    main()
