"""Shared settings for the Bay of Bengal significant-wave-height (SWH) pipeline.

Every module reads its defaults from here, so one edit propagates through
ingestion, preprocessing, feature engineering and windowing. Set the
environment variable ``BOBSWH_DIR`` to relocate the project (for example to
``/content/drive/MyDrive/bob_swh_thesis`` on Google Colab).

Conventions used throughout the package
---------------------------------------
* Timestamps are timezone-naive UTC (the ERA5 convention).
* Directions are in degrees clockwise from true north, "coming from"
  (meteorological / oceanographic convention used by ERA5 ``mwd`` and buoys).
* Working units: m, s, m s-1, hPa, degC, N m-2, kg m-3.
* Column prefixes: ``era5_`` for reanalysis point series, ``buoy_`` for
  INCOIS OMNI observations.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_DIR = Path(os.environ.get("BOBSWH_DIR", Path(__file__).resolve().parents[1]))
RAW_ERA5_DIR = PROJECT_DIR / "data" / "raw" / "era5"
RAW_BUOY_DIR = PROJECT_DIR / "data" / "raw" / "buoy"   # one sub-folder per buoy, e.g. buoy/BD08/*.csv
PROC_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
MODEL_DIR = PROJECT_DIR / "models"

# ---------------------------------------------------------------- ERA5 request
# CDS "area" order is [North, West, South, East]. The box covers the whole bay
# with a southern margin so that swell arriving from the southern bay and the
# equatorial Indian Ocean is inside the extraction domain.
ERA5_AREA: tuple[float, float, float, float] = (23.0, 79.0, 4.0, 100.0)
ERA5_YEARS: tuple[int, ...] = tuple(range(2015, 2025))   # 10 years, 120 months

# CDS request name -> short name inside the NetCDF file.
ERA5_VARIABLES: dict[str, str] = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "sea_surface_temperature": "sst",
    "2m_temperature": "t2m",
    "significant_height_of_combined_wind_waves_and_swell": "swh",
    "mean_wave_direction": "mwd",
    "mean_wave_period": "mwp",
    "peak_wave_period": "pp1d",
    "significant_height_of_wind_waves": "shww",
    "significant_height_of_total_swell": "shts",
}

# ERA5 hourly time-series dataset: the same reanalysis values, served for a
# small box (at most 2 x 2 deg) over any date range, so one request covers the
# whole study period at one buoy. It has 8 of the variables above: everything
# the features and models use, but not pp1d, shww or shts.
ERA5_TS_DATASET = "reanalysis-era5-single-levels-timeseries"
ERA5_TS_VARIABLES: tuple[str, ...] = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_direction",
    "mean_wave_period",
)
ERA5_TS_PERIOD: tuple[str, str] = ("2015-01-01", "2024-12-31")
# Half-width of the box around each buoy. 0.75 deg (~80 km) keeps the right
# 0.25 and 0.5 deg nodes inside the box even if a mooring was redeployed some
# distance from its nominal position.
ERA5_TS_HALF_WIDTH_DEG = 0.75


# ---------------------------------------------------------------- buoys
@dataclass(frozen=True)
class BuoySite:
    """Nominal mooring position of an INCOIS OMNI buoy (decimal degrees, WGS84)."""

    name: str
    lat: float
    lon: float


# APPROXIMATE nominal positions: verify them against the metadata that INCOIS
# supplies with your files. Moorings are redeployed during servicing cruises,
# so whenever a file carries its own latitude/longitude columns the pipeline
# uses the median of those instead of the values below.
BUOYS: dict[str, BuoySite] = {
    "BD08": BuoySite("BD08", 18.15, 89.67),
    "BD10": BuoySite("BD10", 16.32, 87.98),
    "BD11": BuoySite("BD11", 13.58, 84.11),
}

# Height of the buoy anemometer above sea level (m). None means "leave the
# buoy wind as reported". Set it from the INCOIS sensor metadata so that buoy
# winds are reduced to the 10 m reference height used by ERA5.
ANEMOMETER_HEIGHT_M: float | None = None

# ---------------------------------------------------------------- time base
BASE_FREQ = "1h"                       # use "3h" if the buoy wave record is 3-hourly
HORIZONS_H: tuple[int, ...] = (6, 12, 24, 48, 72)
LOOKBACK_H = 72
MAX_GAP_H = 3                          # longest gap (hours) that may be imputed
FILL_METHOD = "ffill"                  # causal. "linear" / "pchip" / "cubic" read past the gap

# Chronological splits (inclusive bounds) over 2015-2024. The training period
# (2015-2020) contains Fani (2019) and Amphan (2020), validation (2021-2022)
# Yaas and Sitrang, and test (2023-2024) Mocha, Michaung and Remal, so
# extreme-event skill is tuned and measured on storms the models never saw.
SPLITS: dict[str, tuple[str, str]] = {
    "train": ("2015-01-01 00:00", "2020-12-31 23:00"),
    "val": ("2021-01-01 00:00", "2022-12-31 23:00"),
    "test": ("2023-01-01 00:00", "2024-12-31 23:00"),
}
# The last part of the training period is held out as the early-stopping block
# (models are fitted on 2015-2019 and early-stopped on 2020), so that
# validation is used only to choose hyperparameters (see tuning.py).
# 2020 is a leap year: 366 days is exactly the calendar year.
EARLY_STOPPING_DAYS = 366

# ---------------------------------------------------------------- physics
GRAVITY = 9.81             # m s-2
R_DRY_AIR = 287.05         # J kg-1 K-1
RHO_AIR_FALLBACK = 1.17    # kg m-3, tropical marine air at ~28 degC and ~1008 hPa
SEA_ROUGHNESS_M = 2.0e-4   # open-ocean aerodynamic roughness length z0 (m)

# ---------------------------------------------------------------- variable metadata
# Keys are base names, i.e. column names without the era5_/buoy_ prefix.
CIRCULAR_BASE = frozenset({"mwd", "wdir", "wdir10", "mdww", "mdts", "dwi"})
NON_NEGATIVE_BASE = frozenset({"hs", "swh", "shww", "shts", "wspd", "wspd10", "tp", "tm", "mwp", "pp1d"})

# Physically admissible ranges for the Bay of Bengal (values outside -> NaN).
QC_BOUNDS: dict[str, tuple[float, float]] = {
    "hs": (0.0, 20.0), "swh": (0.0, 20.0), "shww": (0.0, 20.0), "shts": (0.0, 20.0),
    "tp": (1.0, 30.0), "tm": (1.0, 30.0), "mwp": (1.0, 30.0), "pp1d": (1.0, 30.0),
    "mwd": (0.0, 360.0), "wdir": (0.0, 360.0),
    "wspd": (0.0, 75.0), "u10": (-75.0, 75.0), "v10": (-75.0, 75.0),
    "mslp": (870.0, 1060.0), "msl": (870.0, 1060.0),
    "sst": (10.0, 36.0), "airt": (5.0, 45.0), "t2m": (5.0, 45.0),
}

# Isolated-spike rule per sampling step: (jump, neighbour tolerance) in the
# variable's units. A sample is a spike when it departs from BOTH neighbours
# by more than `jump` in the same sense while the two neighbours agree within
# `neighbour_tol`. Sustained cyclone peaks therefore survive. Tuned for an
# hourly step; scale the jump up if BASE_FREQ is coarser.
SPIKE_RULES: dict[str, tuple[float, float]] = {
    "hs": (1.5, 0.5),
}


def base_name(col: str) -> str:
    """Strip the ``era5_`` / ``buoy_`` source prefix from a column name."""
    for prefix in ("era5_", "buoy_"):
        if col.startswith(prefix):
            return col[len(prefix):]
    return col


def is_circular(col: str) -> bool:
    """True if the column holds a direction in degrees."""
    return base_name(col) in CIRCULAR_BASE


def is_non_negative(col: str) -> bool:
    """True if the column is physically bounded below by zero."""
    return base_name(col) in NON_NEGATIVE_BASE


def qc_bounds(col: str) -> tuple[float, float] | None:
    """Admissible (min, max) range for a column, or None if unchecked."""
    return QC_BOUNDS.get(base_name(col))


def spike_rule(col: str) -> tuple[float, float] | None:
    """(jump, neighbour_tol) of the isolated-spike test, or None."""
    return SPIKE_RULES.get(base_name(col))
