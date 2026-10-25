"""Phase 1 + 2 orchestration: raw ERA5 and buoy files -> one feature table per buoy.

Outputs in ``out_dir``:

* ``{buoy}_features.parquet``   hourly, merged ERA5 + buoy, all engineered features
* ``{buoy}_imputed.parquet``    boolean mask of imputed values (same shape as the raw columns)
* ``era5_node_alignment.csv``   matched ERA5 nodes and distances, per site and grid
* ``qc_report.csv``             per-column QC and gap-filling counts

Windowing and splitting happen at model time (Phase 3/4), from these tables.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from . import config
from .buoy import BuoyReader
from .era5 import ERA5PointExtractor
from .features import FeatureEngineer
from .preprocess import Preprocessor, drop_sparse_columns

log = logging.getLogger(__name__)

BUOY_SUFFIXES = {".csv", ".txt", ".dat"}


class Phase12Pipeline:
    """Ingest, QC, align, merge and engineer features for a set of buoys.

    Parameters
    ----------
    era5_dir:
        Folder searched recursively for ``*.nc``.
    buoy_dir:
        Folder with one sub-folder per buoy name holding its CSV/TXT files.
    sites:
        Buoy metadata; the nominal position is used only when the files carry
        no coordinates (or in ERA5-only mode).
    k_nodes:
        ERA5 nodes per site (1 = nearest valid node).
    min_era5_coverage:
        ERA5 variables present in fewer than this share of hours are dropped
        (with a warning) instead of shrinking every model's sample set.
    """

    def __init__(
        self,
        era5_dir: Path = config.RAW_ERA5_DIR,
        buoy_dir: Path = config.RAW_BUOY_DIR,
        out_dir: Path = config.PROC_DIR,
        sites: Mapping[str, config.BuoySite] = config.BUOYS,
        reader: BuoyReader | None = None,
        freq: str = config.BASE_FREQ,
        max_gap_h: float = config.MAX_GAP_H,
        fill_method: str = config.FILL_METHOD,
        k_nodes: int = 1,
        features: FeatureEngineer | None = None,
        min_era5_coverage: float = 0.9,
    ) -> None:
        self.era5_dir = Path(era5_dir)
        self.buoy_dir = Path(buoy_dir)
        self.out_dir = Path(out_dir)
        self.sites = dict(sites)
        self.reader = reader or BuoyReader()
        self.freq = freq
        self.max_gap_h = max_gap_h
        self.fill_method = fill_method
        self.k_nodes = k_nodes
        self.features = features or FeatureEngineer()
        self.min_era5_coverage = min_era5_coverage

    def _buoy_files(self, name: str) -> list[Path]:
        folder = self.buoy_dir / name
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in BUOY_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"no buoy files for {name} in {folder}")
        return files

    def _era5_files(self) -> list[Path]:
        files = sorted(self.era5_dir.rglob("*.nc"))
        if not files:
            raise FileNotFoundError(f"no ERA5 NetCDF files in {self.era5_dir}")
        return files

    def run(self, names: Iterable[str] | None = None, era5_only: bool = False) -> dict[str, Path]:
        """Process the given buoys (default: all in ``sites``). Returns {buoy: features path}.

        ``era5_only`` builds the tables from ERA5 alone at the nominal buoy
        positions, with ``era5_swh`` as the target; useful while the INCOIS
        data request is pending.
        """
        names = list(names or self.sites)
        positions: dict[str, tuple[float, float]] = {}
        buoy_raw: dict[str, pd.DataFrame] = {}
        for name in names:
            site = self.sites[name]
            positions[name] = (site.lat, site.lon)
            if era5_only:
                continue
            raw = self.reader.read_many(self._buoy_files(name))
            pos = BuoyReader.position(raw)
            if pos is not None:
                positions[name] = pos
            buoy_raw[name] = raw.drop(columns=["lat", "lon"], errors="ignore")
            log.info("%s: position %.3f N %.3f E", name, *positions[name])

        extractor = ERA5PointExtractor(positions, k=self.k_nodes)
        era5 = extractor.extract(self._era5_files())
        self.out_dir.mkdir(parents=True, exist_ok=True)
        extractor.report().to_csv(self.out_dir / "era5_node_alignment.csv", index=False)

        era5_prep = Preprocessor(self.freq, self.max_gap_h, self.fill_method, instantaneous=True, despike=False)
        buoy_prep = Preprocessor(self.freq, self.max_gap_h, self.fill_method, instantaneous=False, despike=True)
        outputs: dict[str, Path] = {}
        qc_tables = []
        for name in names:
            series, dropped = drop_sparse_columns(era5[name], self.min_era5_coverage)
            if dropped:
                log.warning("%s: dropped ERA5 variables with < %.0f%% coverage: %s",
                            name, 100 * self.min_era5_coverage, ", ".join(dropped))
            e = era5_prep.run(series)
            frames, masks = [e.data], [e.imputed]
            qc_tables.append(e.report.assign(site=name, source="era5"))
            if not era5_only:
                b = buoy_prep.run(buoy_raw[name])
                frames.append(b.data.add_prefix("buoy_"))
                masks.append(b.imputed.add_prefix("buoy_"))
                qc_tables.append(b.report.rename(index=lambda c: f"buoy_{c}").assign(site=name, source="buoy"))
            index = e.data.index
            merged = pd.concat([f.reindex(index) for f in frames], axis=1)
            imputed = pd.concat([m.reindex(index, fill_value=False) for m in masks], axis=1)
            table = self.features.transform(merged)
            path = self.out_dir / f"{name}_features.parquet"
            table.to_parquet(path)
            imputed.to_parquet(self.out_dir / f"{name}_imputed.parquet")
            outputs[name] = path
            log.info("%s: %d rows x %d columns -> %s", name, *table.shape, path.name)
        pd.concat(qc_tables).to_csv(self.out_dir / "qc_report.csv")
        return outputs
