"""Run Phase 1 + 2: raw ERA5 NetCDF and INCOIS buoy files -> feature tables.

    python scripts/build_features.py                       # all buoys in config.BUOYS
    python scripts/build_features.py --buoys BD08 BD11
    python scripts/build_features.py --era5-only           # ERA5 swh as target, no buoy files needed
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bobswh import config  # noqa: E402
from bobswh.pipeline import Phase12Pipeline  # noqa: E402
from bobswh.preprocess import FILL_METHODS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buoys", nargs="+", default=list(config.BUOYS))
    ap.add_argument("--era5-only", action="store_true")
    ap.add_argument("--freq", default=config.BASE_FREQ)
    ap.add_argument("--max-gap-h", type=float, default=config.MAX_GAP_H)
    ap.add_argument("--fill-method", choices=FILL_METHODS, default=config.FILL_METHOD)
    ap.add_argument("--k-nodes", type=int, default=1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pipe = Phase12Pipeline(freq=args.freq, max_gap_h=args.max_gap_h, fill_method=args.fill_method, k_nodes=args.k_nodes)
    for name, path in pipe.run(args.buoys, era5_only=args.era5_only).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
