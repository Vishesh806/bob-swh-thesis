"""Readers for INCOIS OMNI moored-buoy files (CSV / TXT).

INCOIS exports differ between data requests, portals and years in their column
names, units in headers, date layout and missing-value codes. The reader
therefore maps raw headers to canonical names through an alias table, which a
``BuoyFileSpec`` can override. Check the first rows of your own files and
put any header the alias table does not know into ``BuoyFileSpec.column_map``.

Canonical columns: hs, tp, tm, mwd, wspd, wdir, mslp, sst, airt, lat, lon.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from . import config
from .features import wind_to_10m

log = logging.getLogger(__name__)

CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "hs": ("hs", "swh", "hm0", "hsig", "wvht", "sig_wave_height", "significant_wave_height", "wave_height"),
    "tp": ("tp", "dpd", "peak_period", "peak_wave_period"),
    "tm": ("tm", "tz", "tavg", "mwp", "apd", "mean_period", "mean_wave_period", "avg_wave_period"),
    "mwd": ("mwd", "mean_wave_direction", "wave_direction", "wave_dir", "mwdir"),
    "wspd": ("wspd", "ws", "wind_speed", "windspeed"),
    "wdir": ("wdir", "wd", "wind_direction", "winddir"),
    "mslp": ("mslp", "slp", "pres", "pressure", "air_pressure", "bp", "atm_pressure"),
    "sst": ("sst", "wtmp", "water_temp", "water_temperature", "sea_surface_temperature"),
    "airt": ("airt", "at", "atmp", "air_temp", "air_temperature"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "long", "longitude"),
}
DATETIME_ALIASES = ("datetime", "date_time", "timestamp", "time_utc", "obs_time", "date_time_utc")
DATE_ALIASES = ("date", "obs_date")
TIME_ALIASES = ("time", "hh_mm", "hhmm")


def normalise_header(name: object) -> str:
    """Lower-case, drop bracketed units, collapse non-alphanumerics: 'SWH (m)' -> 'swh'."""
    s = re.sub(r"[\(\[].*?[\)\]]", "", str(name)).strip().lower()
    return re.sub(r"[^0-9a-z]+", "_", s).strip("_")


@dataclass
class BuoyFileSpec:
    """How to read one family of buoy files.

    Attributes
    ----------
    column_map:
        Explicit ``{raw header: canonical name}``; raw headers are matched
        after ``normalise_header``. Takes precedence over the alias table.
    datetime_cols:
        Raw columns that together form the timestamp (joined with spaces).
        None = auto-detect.
    datetime_format, dayfirst:
        Passed to ``pandas.to_datetime``.
    utc_offset_hours:
        Offset of the file's clock from UTC; 5.5 for IST timestamps.
    missing_values:
        Sentinel codes replaced by NaN. Positive codes such as 999.0 are left
        out on purpose because 999 hPa is a valid pressure in a cyclone; the
        physical range check removes the remaining sentinels.
    anemometer_height_m:
        If set, buoy wind speed is reduced to 10 m with a neutral log profile.
    """

    column_map: Mapping[str, str] = field(default_factory=dict)
    datetime_cols: tuple[str, ...] | None = None
    datetime_format: str | None = None
    dayfirst: bool = False
    utc_offset_hours: float = 0.0
    delimiter: str | None = None
    skiprows: int = 0
    comment: str | None = None
    missing_values: tuple[float, ...] = (-999.0, -999.9, -9999.0, -99.0, -99.9, 9999.0, 99999.0)
    anemometer_height_m: float | None = config.ANEMOMETER_HEIGHT_M


class BuoyReader:
    """Read INCOIS OMNI buoy files into a UTC-indexed DataFrame with canonical columns."""

    def __init__(self, spec: BuoyFileSpec | None = None) -> None:
        self.spec = spec or BuoyFileSpec()

    # -- public --------------------------------------------------------
    def read(self, path: Path | str) -> pd.DataFrame:
        path = Path(path)
        raw = self._read_table(path)
        raw.columns = [normalise_header(c) for c in raw.columns]
        index = self._parse_time(raw, path)
        mapping = self._map_columns(raw.columns)
        if "hs" not in mapping.values():
            raise ValueError(f"{path.name}: no significant-wave-height column found in {list(raw.columns)}")
        out = pd.DataFrame(index=index)
        for src, canon in mapping.items():
            x = pd.to_numeric(raw[src], errors="coerce").to_numpy(float)
            x[np.isin(np.round(x, 3), self.spec.missing_values)] = np.nan
            out[canon] = x
        out = out[out.index.notna()]
        if "wspd" in out and self.spec.anemometer_height_m:
            out["wspd"] = wind_to_10m(out["wspd"], self.spec.anemometer_height_m)
        log.info("%s: %d records, columns %s", path.name, len(out), list(out.columns))
        return out

    def read_many(self, paths: Iterable[Path | str]) -> pd.DataFrame:
        """Concatenate files, sort by time and keep the first record of duplicate timestamps."""
        frames = [self.read(p) for p in paths]
        if not frames:
            raise FileNotFoundError("no buoy files given")
        df = pd.concat(frames).sort_index()
        n_dup = int(df.index.duplicated().sum())
        if n_dup:
            log.info("dropped %d duplicate timestamps", n_dup)
        return df[~df.index.duplicated(keep="first")]

    @staticmethod
    def position(df: pd.DataFrame) -> tuple[float, float] | None:
        """Median (lat, lon) reported in the file, or None if absent."""
        if {"lat", "lon"} <= set(df.columns) and df["lat"].notna().any() and df["lon"].notna().any():
            lon = ((float(df["lon"].median()) + 180.0) % 360.0) - 180.0
            return float(df["lat"].median()), lon
        return None

    # -- internals -----------------------------------------------------
    def _read_table(self, path: Path) -> pd.DataFrame:
        kw = dict(skiprows=self.spec.skiprows, comment=self.spec.comment)
        if self.spec.delimiter is not None:
            return pd.read_csv(path, sep=self.spec.delimiter, **kw)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, **kw)
        try:
            df = pd.read_csv(path, sep=None, engine="python", **kw)
        except Exception:
            df = pd.DataFrame()
        if df.shape[1] <= 1:
            df = pd.read_csv(path, sep=r"\s+", **kw)
        return df

    def _map_columns(self, columns: Iterable[str]) -> dict[str, str]:
        columns = list(columns)
        mapping: dict[str, str] = {}
        for raw, canon in self.spec.column_map.items():
            raw = normalise_header(raw)
            if raw in columns:
                mapping[raw] = canon
        taken = set(mapping.values())
        for canon, aliases in CANONICAL_ALIASES.items():
            if canon in taken:
                continue
            src = next((a for a in aliases if a in columns and a not in mapping), None)
            if src is not None:
                mapping[src] = canon
        return mapping

    def _parse_time(self, df: pd.DataFrame, path: Path) -> pd.DatetimeIndex:
        cols = set(df.columns)
        if self.spec.datetime_cols:
            parts = [normalise_header(c) for c in self.spec.datetime_cols]
        elif (dt := next((c for c in DATETIME_ALIASES if c in cols), None)) is not None:
            parts = [dt]
        elif {"year", "month", "day", "hour"} <= cols:
            fields = ["year", "month", "day", "hour"] + (["minute"] if "minute" in cols else [])
            ts = pd.to_datetime(df[fields].apply(pd.to_numeric, errors="coerce"), errors="coerce")
            return self._to_utc(pd.DatetimeIndex(ts), path)
        else:
            d = next((c for c in DATE_ALIASES if c in cols), None)
            t = next((c for c in TIME_ALIASES if c in cols), None)
            parts = [c for c in (d, t) if c is not None]
            if not parts:
                raise ValueError(f"{path.name}: cannot find a timestamp column in {sorted(cols)}")
        text = df[parts].astype(str).agg(" ".join, axis=1)
        ts = pd.to_datetime(text, format=self.spec.datetime_format, dayfirst=self.spec.dayfirst, errors="coerce")
        return self._to_utc(pd.DatetimeIndex(ts), path)

    def _to_utc(self, ts: pd.DatetimeIndex, path: Path) -> pd.DatetimeIndex:
        n_bad = int(ts.isna().sum())
        if n_bad:
            log.warning("%s: %d unparseable timestamps dropped", path.name, n_bad)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        elif self.spec.utc_offset_hours:
            ts = ts - pd.Timedelta(hours=self.spec.utc_offset_hours)
        return ts.rename("time")
