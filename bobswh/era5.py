"""ERA5 single-level download (Copernicus CDS) and point extraction at buoy sites.

ERA5 atmospheric fields are delivered on a 0.25 deg grid, wave fields on a
0.5 deg grid (WAM). The current CDS returns a request that mixes both streams
as a zip holding one NetCDF per stream. Each file is therefore handled on its
own grid: nodes are matched per grid, point series are extracted per file, and
the streams are merged on time afterwards. Extracting points file by file also
means the multi-year, basin-wide archive never has to be held in memory.
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .alignment import NearestNodeAligner, NodeMatch, grid_key

log = logging.getLogger(__name__)

_SPATIAL_DIMS = ("time", "latitude", "longitude")


# ---------------------------------------------------------------- download
class ERA5Downloader:
    """Download hourly ERA5 single-level fields from the CDS, one request per month.

    Needs ``pip install "cdsapi>=0.7"`` and a personal access token in
    ``~/.cdsapirc`` (see https://cds.climate.copernicus.eu/how-to-api).
    """

    def __init__(
        self,
        out_dir: Path = config.RAW_ERA5_DIR,
        area: Sequence[float] = config.ERA5_AREA,
        variables: Sequence[str] = tuple(config.ERA5_VARIABLES),
        dataset: str = "reanalysis-era5-single-levels",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.area = list(area)
        self.variables = list(variables)
        self.dataset = dataset

    def request(self, year: int, month: int) -> dict:
        """CDS request body for one month (invalid days such as 31 Feb are ignored by the CDS)."""
        return {
            "product_type": ["reanalysis"],
            "variable": self.variables,
            "year": [f"{year}"],
            "month": [f"{month:02d}"],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": self.area,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

    def target(self, year: int, month: int) -> Path:
        return self.out_dir / f"era5_bob_{year}{month:02d}.nc"

    def download(self, years: Iterable[int], months: Iterable[int] = range(1, 13), overwrite: bool = False,
                 retries: int = 5, wait_s: float = 120.0) -> list[Path]:
        """Fetch every (year, month) sequentially; skip months already on disk.

        The CDS limits the number of queued requests per user, so requests are
        sent one at a time. A rejected or failed month is retried with linear
        back-off (wait_s, 2 wait_s, ...); after ``retries`` failures it is
        skipped and listed at the end. Re-running fetches only what is missing.
        Returns the NetCDF paths written.
        """
        import time

        import cdsapi  # imported here so the rest of the package works without it

        client = cdsapi.Client()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        months = list(months)
        written: list[Path] = []
        failed: list[str] = []
        for year in years:
            for month in months:
                target = self.target(year, month)
                unpacked = target.with_suffix("")
                if not overwrite and (target.exists() or unpacked.is_dir()):
                    log.info("skip %s (already downloaded)", target.name)
                    continue
                part = target.with_suffix(".part")
                for attempt in range(1, retries + 1):
                    try:
                        log.info("requesting %s (attempt %d)", target.name, attempt)
                        client.retrieve(self.dataset, self.request(year, month), str(part))
                        written.extend(self._finalise(part, target, unpacked))
                        log.info("done %s", target.name)
                        break
                    except Exception as exc:          # CDS rejections, timeouts, dropped connections
                        part.unlink(missing_ok=True)
                        log.warning("%s failed: %s", target.name, str(exc).splitlines()[-1] if str(exc) else exc)
                        if attempt == retries:
                            failed.append(target.name)
                        else:
                            time.sleep(wait_s * attempt)
        if failed:
            log.error("%d months failed, re-run to retry: %s", len(failed), " ".join(failed))
        return written

    @staticmethod
    def _finalise(part: Path, target: Path, unpacked: Path) -> list[Path]:
        """Move a finished download into place, unzipping multi-stream responses."""
        if zipfile.is_zipfile(part):
            unpacked.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(part) as zf:
                names = [n for n in zf.namelist() if n.endswith(".nc")]
                for name in names:
                    dest = unpacked / Path(name).name
                    dest.write_bytes(zf.read(name))
            part.unlink()
            return sorted(unpacked.glob("*.nc"))
        part.rename(target)
        return [target]


def datastores_client():
    """``ecmwf.datastores.Client`` built from the credentials in ``~/.cdsapirc``."""
    from ecmwf.datastores import Client

    rc = Path("~/.cdsapirc").expanduser().read_text().splitlines()
    cfg = dict(line.split(":", 1) for line in rc if ":" in line)
    return Client(url=cfg["url"].strip(), key=cfg["key"].strip())


def site_box(lat: float, lon: float, half_width: float = config.ERA5_TS_HALF_WIDTH_DEG) -> list[float]:
    """[N, W, S, E] box around a site, snapped to the 0.25 deg grid."""
    snap = lambda x: round(x * 4) / 4
    return [snap(lat + half_width), snap(lon - half_width), snap(lat - half_width), snap(lon + half_width)]


class ERA5TimeseriesDownloader:
    """Download the whole study period for each buoy from the ERA5 time-series dataset.

    One request per site: a box of ``2 * half_width`` deg around the buoy over
    ``period``. Requests are submitted together. A request refused because
    the per-user queue is full is resubmitted later, and a failed request is
    retried up to ``retries`` times. Each result is unpacked into
    ``out_dir/ts_{site}/``; sites whose folder already exists are skipped.
    """

    def __init__(
        self,
        out_dir: Path = config.RAW_ERA5_DIR,
        variables: Sequence[str] = config.ERA5_TS_VARIABLES,
        period: tuple[str, str] = config.ERA5_TS_PERIOD,
        half_width: float = config.ERA5_TS_HALF_WIDTH_DEG,
        dataset: str = config.ERA5_TS_DATASET,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.variables = list(variables)
        self.period = period
        self.half_width = half_width
        self.dataset = dataset

    def request(self, lat: float, lon: float) -> dict:
        return {
            "variable": self.variables,
            "area": site_box(lat, lon, self.half_width),
            "date": f"{self.period[0]}/{self.period[1]}",
            "data_format": "netcdf",
        }

    def folder(self, site: str) -> Path:
        return self.out_dir / f"ts_{site}"

    def download(self, sites: Mapping[str, tuple[float, float]], overwrite: bool = False,
                 retries: int = 3, poll_s: float = 60.0) -> dict[str, list[Path]]:
        """Submit, wait for and unpack one request per site. Returns {site: NetCDF paths}."""
        import time

        client = datastores_client()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        pending = {s: ll for s, ll in sites.items() if overwrite or not self.folder(s).is_dir()}
        for s in set(sites) - set(pending):
            log.info("skip %s (already downloaded)", s)
        remotes: dict[str, object] = {}
        attempts = {s: 0 for s in pending}
        written: dict[str, list[Path]] = {}
        while pending or remotes:
            for site in list(pending):
                try:
                    remotes[site] = client.submit(self.dataset, self.request(*pending[site]))
                    attempts[site] += 1
                    log.info("%s submitted (request %s)", site, remotes[site].request_id)
                    del pending[site]
                except Exception as exc:
                    if "queued requests" in str(exc):
                        log.info("%s waits: per-user queue is full", site)
                        break
                    raise
            for site, remote in list(remotes.items()):
                status = remote.status
                if status == "successful":
                    part = self.out_dir / f"ts_{site}.part"
                    remote.download(str(part))
                    folder = self.folder(site)
                    folder.mkdir(parents=True, exist_ok=True)
                    written[site] = ERA5Downloader._finalise(part, folder / f"ts_{site}.nc", folder)
                    log.info("done %s: %s", site, ", ".join(p.name for p in written[site]))
                    del remotes[site]
                elif status in ("failed", "rejected", "dismissed", "deleted"):
                    del remotes[site]
                    if attempts[site] < retries:
                        log.warning("%s request %s, resubmitting", site, status)
                        pending[site] = sites[site]
                    else:
                        log.error("%s request %s after %d attempts", site, status, attempts[site])
            if pending or remotes:
                time.sleep(poll_s)
        return written


# ---------------------------------------------------------------- loading
def standardise(ds: xr.Dataset) -> xr.Dataset:
    """Bring old-CDS and new-CDS ERA5 files to one layout.

    * ``valid_time`` (new CDS) is renamed to ``time``.
    * An ``expver`` dimension (old CDS: final ERA5 = 1 stacked with
      near-real-time ERA5T = 5) is collapsed, preferring the first version.
    * Scalar bookkeeping coordinates (``number``, ``expver``, ...) are dropped.
    * Longitude is wrapped to [-180, 180) and both axes sorted ascending
      (ERA5 stores latitude north to south).
    """
    if "valid_time" in ds.variables and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "expver" in ds.dims:
        parts = [ds.sel(expver=v, drop=True) for v in ds["expver"].values]
        merged = parts[0]
        for part in parts[1:]:
            merged = merged.combine_first(part)
        ds = merged
    drop = [c for c in ("number", "expver", "step", "surface") if c in ds.coords and c not in ds.dims]
    ds = ds.drop_vars(drop)
    lon = ((ds["longitude"] + 180.0) % 360.0) - 180.0
    return ds.assign_coords(longitude=lon).sortby(["latitude", "longitude"])


def gridded_variables(ds: xr.Dataset) -> list[str]:
    """Data variables defined on (time, latitude, longitude)."""
    return [v for v in ds.data_vars if set(_SPATIAL_DIMS) <= set(ds[v].dims)]


def to_working_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ERA5 SI units to the working units: msl Pa -> hPa, temperatures K -> degC."""
    out = df.copy()
    if "msl" in out and out["msl"].median() > 2000.0:
        out["msl"] = out["msl"] / 100.0
    for col in ("sst", "t2m", "d2m"):
        if col in out and out[col].median() > 200.0:
            out[col] = out[col] - 273.15
    return out


class ERA5PointExtractor:
    """Extract ERA5 time series at a set of sites from many NetCDF files.

    Parameters
    ----------
    sites:
        ``{name: (lat, lon)}``.
    k:
        Number of grid nodes per site. 1 = nearest valid node (default, keeps
        the reanalysis value unsmoothed); k > 1 = inverse-distance weighting.
        Directional variables are averaged as unit vectors, never as raw
        degrees.
    max_distance_km:
        Fail loudly if a site's nearest valid node is farther than this.
    """

    def __init__(
        self,
        sites: Mapping[str, tuple[float, float]],
        k: int = 1,
        power: float = 2.0,
        max_distance_km: float = 50.0,
    ) -> None:
        self.sites = dict(sites)
        self.k = k
        self.power = power
        self.max_distance_km = max_distance_km
        self._aligners: dict[str, NearestNodeAligner] = {}
        self._matches: dict[tuple[str, str], tuple[NodeMatch, list[str], float]] = {}

    # -- alignment -----------------------------------------------------
    def _aligner_for(self, ds: xr.Dataset, names: list[str]) -> tuple[str, NearestNodeAligner]:
        first = np.stack([ds[v].isel(time=0).values for v in names])
        valid = np.isfinite(first).all(axis=0)
        key = grid_key(ds["latitude"].values, ds["longitude"].values, valid)
        if key not in self._aligners:
            self._aligners[key] = NearestNodeAligner(ds["latitude"].values, ds["longitude"].values, valid)
        return key, self._aligners[key]

    def report(self) -> pd.DataFrame:
        """One row per (site, grid, node): matched node position, distance and weight."""
        rows = []
        for (site, _), (m, names, res) in self._matches.items():
            for j in range(len(m.weights)):
                rows.append({
                    "site": site,
                    "site_lat": self.sites[site][0],
                    "site_lon": self.sites[site][1],
                    "grid_resolution_deg": res,
                    "variables": " ".join(names),
                    "node_lat": float(m.lat[j]),
                    "node_lon": float(m.lon[j]),
                    "distance_km": float(m.distance_km[j]),
                    "weight": float(m.weights[j]),
                })
        return pd.DataFrame(rows)

    # -- extraction ----------------------------------------------------
    def extract_file(self, path: Path) -> pd.DataFrame:
        """Long table (rows: time x site, columns: variables) for one NetCDF file."""
        with xr.open_dataset(path) as raw:
            ds = standardise(raw)
            names = gridded_variables(ds)
            if not names:
                log.warning("%s holds no gridded variables, skipped", path)
                return pd.DataFrame()
            key, aligner = self._aligner_for(ds, names)
            resolution = float(np.abs(np.diff(ds["latitude"].values)).min()) if ds.sizes["latitude"] > 1 else np.nan
            frames = []
            for site, (lat, lon) in self.sites.items():
                try:
                    m = aligner.query(site, lat, lon, k=self.k, power=self.power, max_distance_km=self.max_distance_km)
                except ValueError:
                    # the file does not cover this site (e.g. a time-series box around another buoy)
                    log.debug("%s: %s has no node within %s km, skipped", site, path.name, self.max_distance_km)
                    continue
                self._matches.setdefault((site, key), (m, names, resolution))
                sub = ds[names].isel(
                    latitude=xr.DataArray(m.iy, dims="node"),
                    longitude=xr.DataArray(m.ix, dims="node"),
                ).load()
                w = xr.DataArray(m.weights, dims="node")
                values = {}
                for v in names:
                    da = sub[v]
                    if config.is_circular(v):
                        rad = np.deg2rad(da)
                        s = (np.sin(rad) * w).sum("node", skipna=False)
                        c = (np.cos(rad) * w).sum("node", skipna=False)
                        values[v] = np.rad2deg(np.arctan2(s.values, c.values)) % 360.0
                    else:
                        values[v] = (da * w).sum("node", skipna=False).values
                frame = pd.DataFrame(values, index=pd.DatetimeIndex(sub["time"].values, name="time"))
                frame["site"] = site
                frames.append(frame)
        return pd.concat(frames) if frames else pd.DataFrame()

    def extract(self, paths: Iterable[Path]) -> dict[str, pd.DataFrame]:
        """Point series per site: hourly, working units, columns prefixed ``era5_``.

        Files from different streams (atmosphere / waves), different products
        (gridded monthly files, per-buoy time-series boxes) and overlapping
        downloads are merged by taking, for each (time, site), the first
        non-missing value of every variable. A file that does not cover a site
        is skipped for that site; a site covered by no file at all is an error.
        """
        paths = sorted(Path(p) for p in paths)
        if not paths:
            raise FileNotFoundError("no ERA5 NetCDF files given")
        parts = [p for p in (self.extract_file(p) for p in paths) if not p.empty]
        covered = set().union(*(set(p["site"]) for p in parts)) if parts else set()
        missing = sorted(set(self.sites) - covered)
        if missing:
            raise ValueError(f"no ERA5 file has a valid node within {self.max_distance_km} km of {missing}")
        long = pd.concat(parts)
        long = long.reset_index().groupby(["site", "time"]).first()
        out: dict[str, pd.DataFrame] = {}
        for site, group in long.groupby(level="site"):
            df = group.droplevel("site").sort_index()
            out[site] = to_working_units(df).add_prefix("era5_")
        return out
