"""Live forecasts: recent hourly data from Open-Meteo -> the same features -> the trained tree models.

ERA5 is published about five days after real time, so it cannot drive a forecast of the coming days. The free
Open-Meteo APIs serve the same variables, hourly, up to the present: wind, pressure and air temperature from ECMWF's
operational forecast system (IFS), and wave height, period, direction and SST from its operational wave model (WAM)
and ocean analyses. They are close to, but not the same as, ERA5, so a live forecast is a demonstration of the
trained models on near-real-time inputs, not a validated result. Open-Meteo also returns its own numerical wave
forecast for the next days, which the dashboard shows for comparison.

Only data at or before the forecast origin enter the model inputs, exactly as in training.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .features import FeatureEngineer

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_VARS = {"wave_height": "era5_swh", "wave_period": "era5_mwp", "wave_direction": "era5_mwd",
               "sea_surface_temperature": "era5_sst"}
WEATHER_VARS = {"wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_direction",
                "pressure_msl": "era5_msl", "temperature_2m": "era5_t2m"}
# 30-day pressure anomaly + 72 h lags + margin
PAST_DAYS = 40
FORECAST_DAYS = 4


def _get_json(url: str, params: Mapping[str, object], timeout: float) -> dict:
    query = urllib.parse.urlencode({k: (",".join(v) if isinstance(v, (list, tuple)) else v) for k, v in params.items()})
    req = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "bobswh-dashboard"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wind_components(speed, direction_from_deg) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) from speed and meteorological direction (the direction the wind blows FROM)."""
    rad = np.deg2rad(np.asarray(direction_from_deg, float))
    s = np.asarray(speed, float)
    return -s * np.sin(rad), -s * np.cos(rad)


def frame_from_json(marine: dict, weather: dict) -> pd.DataFrame:
    """Hourly UTC frame with the ERA5 column names used in training (era5_u10, era5_v10, era5_msl, ...)."""
    m = pd.DataFrame(marine["hourly"]).rename(columns=MARINE_VARS)
    w = pd.DataFrame(weather["hourly"]).rename(columns=WEATHER_VARS)
    df = m.merge(w, on="time", how="outer")
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index().astype(float)
    df["era5_u10"], df["era5_v10"] = wind_components(df.pop("wind_speed"), df.pop("wind_direction"))
    full = pd.date_range(df.index[0], df.index[-1], freq="1h", name="time")
    return df.reindex(full)[["era5_u10", "era5_v10", "era5_msl", "era5_t2m", "era5_sst", "era5_swh", "era5_mwd", "era5_mwp"]]


def fetch_open_meteo(lat: float, lon: float, past_days: int = PAST_DAYS, forecast_days: int = FORECAST_DAYS,
                     timeout: float = 30) -> pd.DataFrame:
    """Recent and forecast hourly data at one point (UTC), in the training column names."""
    common = {"latitude": lat, "longitude": lon, "past_days": past_days, "forecast_days": forecast_days, "timezone": "GMT"}
    marine = _get_json(MARINE_URL, {**common, "hourly": list(MARINE_VARS)}, timeout)
    weather = _get_json(WEATHER_URL, {**common, "hourly": list(WEATHER_VARS), "wind_speed_unit": "ms"}, timeout)
    return frame_from_json(marine, weather)


def inputs_at(features: pd.DataFrame, feature_names: Sequence[str], origin: pd.Timestamp) -> pd.DataFrame:
    """One row of model inputs at ``origin``: column '{col}_lag{k}h' = features[col] at origin - k hours.

    This is the same layout as ``SlidingWindowGenerator.tabular`` (checked in the tests).
    """
    origin = pd.Timestamp(origin)
    row = {}
    for name in feature_names:
        m = re.fullmatch(r"(.+)_lag(\d+)h", name)
        if m is None:
            raise ValueError(f"unexpected feature name {name!r}")
        col, lag = m.group(1), int(m.group(2))
        row[name] = float(features.at[origin - pd.Timedelta(hours=lag), col])
    return pd.DataFrame([row], index=pd.DatetimeIndex([origin], name="origin"))


def latest_origin(raw: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Most recent full hour at or before ``now`` (UTC) at which every input variable is available."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    past = raw[raw.index <= now.floor("1h")].dropna()
    if past.empty:
        raise ValueError("no complete hour of live data at or before now")
    return past.index[-1]


def live_forecast(models: Mapping[str, object], raw: pd.DataFrame, origin: pd.Timestamp,
                  max_fill_h: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecasts from ``origin`` by each model (DirectMultiStepRegressor), using only data up to ``origin``.

    Returns (forecasts: rows = models, columns = lead times in hours; the model input row of the first model).
    Gaps of up to ``max_fill_h`` hours are forward-filled (causal), as in training.
    """
    hist = raw[raw.index <= origin].ffill(limit=max_fill_h)
    feats = FeatureEngineer().transform(hist)
    rows, first_x = {}, None
    for name, model in models.items():
        x = inputs_at(feats, model.feature_names_, origin)
        if x.isna().any(axis=None):
            missing = list(x.columns[x.isna().iloc[0]])[:5]
            raise ValueError(f"missing inputs at {origin}: {missing}")
        rows[name] = model.predict(x)[0]
        first_x = x if first_x is None else first_x
    horizons = next(iter(models.values())).horizons_h
    return pd.DataFrame(rows, index=list(horizons)).T, first_x
