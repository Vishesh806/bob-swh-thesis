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
from dataclasses import dataclass
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


# ---------------------------------------------------------------- retraining on live-source data
HIST_WEATHER_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def fetch_open_meteo_history(lat: float, lon: float, start: str, end: str, timeout: float = 180) -> pd.DataFrame:
    """Hourly history (UTC) from Open-Meteo's archives of the same operational sources, in the training column names.

    The archives also hold forecast values for the rest of the end date, so callers must cut the frame at the
    current hour before using it for training (``retrain_lightgbm`` takes ``until`` for this).
    """
    common = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end, "timezone": "GMT"}
    marine = _get_json(MARINE_URL, {**common, "hourly": list(MARINE_VARS)}, timeout)
    weather = _get_json(HIST_WEATHER_URL, {**common, "hourly": list(WEATHER_VARS), "wind_speed_unit": "ms"}, timeout)
    return frame_from_json(marine, weather)


def design_matrix(features: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Model inputs for every hour at once: column '{col}_lag{k}h' = features[col] shifted by k hours.

    Row t equals ``inputs_at(features, feature_names, t)`` (checked in the tests).
    """
    cols = {}
    for name in feature_names:
        m = re.fullmatch(r"(.+)_lag(\d+)h", name)
        if m is None:
            raise ValueError(f"unexpected feature name {name!r}")
        cols[name] = features[m.group(1)].shift(int(m.group(2)))
    return pd.DataFrame(cols, index=features.index)


def target_matrix(features: pd.DataFrame, horizons: Sequence[int], col: str = "era5_swh") -> pd.DataFrame:
    """Targets for every origin: Hs at t + h for each lead time h (hourly index)."""
    return pd.DataFrame({h: features[col].shift(-h) for h in horizons}, index=features.index)


@dataclass
class RetrainResult:
    model: object                    # DirectMultiStepRegressor refitted on all data up to ``until``
    scores: pd.DataFrame             # RMSE by lead time on the evaluation window: retrained / original / persistence
    n_fit: int
    n_eval: int
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    best_rounds: list[int]


def retrain_lightgbm(raw: pd.DataFrame, feature_names: Sequence[str], params: Mapping, original=None,
                     until: pd.Timestamp | None = None, eval_days: int = 30, es_days: int = 60,
                     horizons: Sequence[int] = (6, 12, 24, 48, 72), seed: int = 42, n_jobs: int = -1) -> RetrainResult:
    """Retrain the direct multi-step LightGBM on live-source history, with the same leakage rules as the thesis.

    Blocks (by forecast origin, each containing all targets of its samples):
        fit  : origins with t + 72 h before the early-stopping block
        es   : the ``es_days`` before the evaluation block (chooses the number of boosting rounds per lead time)
        eval : the last ``eval_days`` whose targets are already observed (scores retrained vs original vs persistence)
    The returned model is then refitted on fit + es + eval with the rounds found by early stopping, i.e. on all
    data up to ``until``. Hyperparameters are the ones tuned in the thesis (``params``), not re-tuned.
    """
    from .trees import DirectMultiStepRegressor, fit_estimator, make_estimator

    until = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("1h") if until is None else pd.Timestamp(until)
    hist = raw[raw.index <= until].ffill(limit=3)          # never train on values after `until` (they are forecasts)
    feats = FeatureEngineer().transform(hist)
    X, Y = design_matrix(feats, feature_names), target_matrix(feats, horizons)
    ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
    X, Y = X[ok], Y[ok]
    hmax = pd.Timedelta(hours=max(horizons))
    last_origin = X.index[-1]                             # its 72 h target is the last observed hour
    eval_start = last_origin - pd.Timedelta(days=eval_days)
    es_start = eval_start - hmax - pd.Timedelta(days=es_days)
    is_eval = X.index > eval_start
    is_es = (X.index > es_start) & (X.index + hmax <= eval_start)
    is_fit = X.index + hmax <= es_start
    if is_fit.sum() < 1000:
        raise ValueError(f"too little history to retrain ({int(is_fit.sum())} training samples); choose a longer period")

    model = DirectMultiStepRegressor("lightgbm", horizons, params, max_rounds=3000, early_stopping_rounds=50,
                                     seed=seed, n_jobs=n_jobs)
    model.fit(X[is_fit], Y[is_fit], X[is_es], Y[is_es])
    Xe, Ye = X[is_eval], Y[is_eval].to_numpy()
    rmse = lambda p: np.sqrt(np.mean((np.asarray(p) - Ye) ** 2, axis=0))
    scores = {"Retrained on live data": rmse(model.predict(Xe)),
              "Persistence": rmse(np.repeat(Xe["era5_swh_lag0h"].to_numpy()[:, None], len(horizons), axis=1))}
    if original is not None:
        scores["Original (trained on ERA5)"] = rmse(original.predict(Xe))
    scores = pd.DataFrame(scores, index=list(horizons)).rename_axis("horizon_h")

    # refit on everything up to `until`, keeping each lead time's number of rounds
    final = DirectMultiStepRegressor("lightgbm", horizons, params, seed=seed, n_jobs=n_jobs)
    final.feature_names_ = list(X.columns)
    for j, rounds in enumerate(model.best_iterations_):
        est = make_estimator("lightgbm", params, max(int(rounds), 1), None, seed + j, n_jobs)
        est, _ = fit_estimator(est, "lightgbm", X, Y.iloc[:, j].to_numpy())
        final.models_.append(est)
        final.best_iterations_.append(int(rounds))
    return RetrainResult(final, scores, int(is_fit.sum()), int(is_eval.sum()), eval_start, last_origin + pd.Timedelta(0),
                         list(model.best_iterations_))
