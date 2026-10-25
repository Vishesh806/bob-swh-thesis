"""Synthetic ERA5 + buoy feature frames shared by the Phase 3 and Phase 4 tests.

Wind follows a slow AR(1) process around a monsoon-like seasonal cycle. Hs
responds to the wind averaged over the previous 6-24 h, plus a swell
component, so Hs at t + h depends on forcing the forecaster has partly seen.
With ``storms=True`` a few cyclone-like wind bursts (a Gaussian pulse of up to
25 m/s lasting ~2 days) produce isolated extreme-Hs events, which is what the
peak-weighted loss is for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bobswh.features import FeatureEngineer


def synthetic_frame(seed: int = 0, storms: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2016-01-01", "2018-12-31 23:00", freq="1h")
    n = len(idx)
    rng = np.random.default_rng(seed)
    season = 4.0 + 3.0 * np.sin(2 * np.pi * (idx.dayofyear.to_numpy() - 150) / 365.25)
    a = np.zeros((n, 2))
    for t in range(1, n):
        a[t] = 0.995 * a[t - 1] + rng.normal(0, 0.35, 2)
    u = season + a[:, 0]
    v = 0.5 * a[:, 1]
    burst = np.zeros(n)
    if storms:
        hours = np.arange(n)
        for centre in rng.choice(np.arange(500, n - 500), size=12, replace=False):
            burst += rng.uniform(15, 25) * np.exp(-0.5 * ((hours - centre) / 18.0) ** 2)
        u = u + burst
    speed = pd.Series(np.hypot(u, v), index=idx)
    forcing = speed.shift(6).rolling(18, min_periods=1).mean().bfill()
    swell = 0.3 + 0.2 * np.sin(2 * np.pi * np.arange(n) / (24 * 9))
    hs = 0.12 * forcing.to_numpy() + swell + rng.normal(0, 0.03, n)
    df = pd.DataFrame({
        "era5_u10": u, "era5_v10": v, "era5_msl": 1008 - 0.3 * a[:, 0] - 0.8 * burst + rng.normal(0, 0.2, n),
        "era5_t2m": 28.0, "era5_sst": 29.0,
        "era5_swh": 0.95 * hs + rng.normal(0, 0.05, n), "era5_mwp": 6.5 + 0.1 * burst, "era5_mwd": 200.0 + 10 * a[:, 1],
        "buoy_hs": np.maximum(hs, 0.05),
    }, index=idx)
    df.iloc[5000:5100, df.columns.get_loc("buoy_hs")] = np.nan        # a long buoy outage
    imputed = pd.DataFrame(False, index=idx, columns=df.columns)
    imputed.iloc[9000:9002, imputed.columns.get_loc("buoy_hs")] = True
    return FeatureEngineer().transform(df), imputed
