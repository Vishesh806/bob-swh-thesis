"""Download hourly ERA5 single-level fields for the Bay of Bengal, one file per month.

    python scripts/download_era5.py --years 2015 2024
    python scripts/download_era5.py --years 2020 2020 --months 5     # Amphan only

Needs a CDS account and ~/.cdsapirc. Re-running skips months already on disk.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bobswh import config  # noqa: E402
from bobswh.era5 import ERA5Downloader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    default=(config.ERA5_YEARS[0], config.ERA5_YEARS[-1]))
    ap.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    ap.add_argument("--out", type=Path, default=config.RAW_ERA5_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ERA5Downloader(out_dir=args.out).download(range(args.years[0], args.years[1] + 1), args.months, args.overwrite)


if __name__ == "__main__":
    main()
