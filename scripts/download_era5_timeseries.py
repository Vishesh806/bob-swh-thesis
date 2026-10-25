"""Download 2015-2024 ERA5 for each buoy from the CDS time-series dataset (one request per buoy).

    python scripts/download_era5_timeseries.py                     # all buoys in config.BUOYS
    python scripts/download_era5_timeseries.py --buoys BD08
    python scripts/download_era5_timeseries.py --start 2016-01-01 --end 2025-12-31

Each result is unpacked into data/raw/era5/ts_{buoy}/. Re-running skips buoys already downloaded.
Needs a CDS account, ~/.cdsapirc and the CC-BY licence of the dataset accepted on the website.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bobswh import config  # noqa: E402
from bobswh.era5 import ERA5TimeseriesDownloader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buoys", nargs="+", default=list(config.BUOYS))
    ap.add_argument("--start", default=config.ERA5_TS_PERIOD[0])
    ap.add_argument("--end", default=config.ERA5_TS_PERIOD[1])
    ap.add_argument("--out", type=Path, default=config.RAW_ERA5_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sites = {b: (config.BUOYS[b].lat, config.BUOYS[b].lon) for b in args.buoys}
    dl = ERA5TimeseriesDownloader(out_dir=args.out, period=(args.start, args.end))
    for site, paths in dl.download(sites, overwrite=args.overwrite).items():
        print(site, [p.name for p in paths])


if __name__ == "__main__":
    main()
