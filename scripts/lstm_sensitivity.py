"""LSTM sensitivity check at one buoy: learning rate, dropout and input set (MSE loss).

    python scripts/lstm_sensitivity.py --buoy BD08

Writes results/phase4_sensitivity/{buoy}/{config}/ and prints test RMSE per configuration.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from bobswh import config  # noqa: E402
from bobswh.phase4 import Phase4Runner  # noqa: E402
from bobswh.seq2seq import Seq2SeqConfig  # noqa: E402

CONFIGS = {
    "core_lr3e-4_do0.2": ("core", dict(learning_rate=3e-4, dropout=0.2)),
    "extended_lr3e-4_do0.2": ("extended", dict(learning_rate=3e-4, dropout=0.2)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buoy", default="BD08")
    ap.add_argument("--variants", nargs="+", default=["mse"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base = Seq2SeqConfig(epochs=40, patience=6)
    rows = []
    for name, (inputs, kw) in CONFIGS.items():
        out = config.RESULTS_DIR / "phase4_sensitivity" / args.buoy / name
        runner = Phase4Runner(buoy=args.buoy, target_col="era5_swh", out_dir=out.parent, model_dir=out / "models",
                              input_set=inputs, variants=tuple(args.variants), sequence_stride_h=3,
                              model_config=replace(base, **kw), verbose=0)
        runner.out_dir = out.parent / name
        m = runner.run()
        for v in args.variants:
            h = pd.read_csv(out / args.buoy / f"history_{v}.csv", index_col=0)
            r = m.xs((f"lstm_{v}", "test"), level=("model", "split"))["rmse"]
            rows.append({"config": name, "variant": v, "epochs": len(h), "best_epoch": int(h["val_loss"].idxmin()) + 1,
                         **{f"rmse_{k}h": round(float(x), 4) for k, x in r.items()}})
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "phase4_sensitivity" / f"{args.buoy}_summary.csv", index=False)


if __name__ == "__main__":
    main()
