"""Live-forecast tests: Open-Meteo parsing, wind components, and input rows identical to the training layout."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bobswh.live import (design_matrix, frame_from_json, inputs_at, latest_origin, retrain_lightgbm,
                         target_matrix, wind_components)
from bobswh.windowing import SlidingWindowGenerator, WindowSpec


def test_wind_components_follow_the_coming_from_convention():
    u, v = wind_components([10, 10, 10, 10], [0, 90, 180, 270])        # from N, E, S, W
    np.testing.assert_allclose(u, [0, -10, 0, 10], atol=1e-9)
    np.testing.assert_allclose(v, [-10, 0, 10, 0], atol=1e-9)


def test_frame_from_json_maps_names_and_fills_the_hourly_grid():
    times = ["2026-11-01T00:00", "2026-11-01T01:00", "2026-11-01T03:00"]
    marine = {"hourly": {"time": times, "wave_height": [1.0, 1.1, None], "wave_period": [7.0, 7.1, 7.2],
                         "wave_direction": [200, 205, 210], "sea_surface_temperature": [28.5, 28.5, 28.6]}}
    weather = {"hourly": {"time": times, "wind_speed_10m": [5.0, 6.0, 7.0], "wind_direction_10m": [180, 180, 180],
                          "pressure_msl": [1008.0, 1007.5, 1007.0], "temperature_2m": [27.0, 27.1, 27.2]}}
    df = frame_from_json(marine, weather)
    assert list(df.columns) == ["era5_u10", "era5_v10", "era5_msl", "era5_t2m", "era5_sst", "era5_swh", "era5_mwd", "era5_mwp"]
    assert len(df) == 4 and df.index[2] == pd.Timestamp("2026-11-01T02:00") and df.iloc[2].isna().all()
    assert df["era5_v10"].iloc[0] == pytest.approx(5.0)                    # wind from the south blows north
    assert latest_origin(df, pd.Timestamp("2026-11-01T05:00")) == pd.Timestamp("2026-11-01T01:00")   # 03:00 lacks Hs


def test_inputs_at_matches_the_training_layout():
    idx = pd.date_range("2024-01-01", periods=400, freq="1h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"era5_swh": rng.uniform(0.5, 3, 400), "era5_tau": rng.uniform(0, 1, 400),
                       "era5_msl": rng.normal(1008, 2, 400)}, index=idx)
    spec = WindowSpec(target_col="era5_swh", input_cols=("era5_tau", "era5_msl"), tabular_lag_cols=("era5_swh", "era5_tau"))
    X, _ = SlidingWindowGenerator(spec).tabular(df)
    for origin in X.index[[0, 57, -1]]:
        row = inputs_at(df, list(X.columns), origin)
        np.testing.assert_allclose(row.to_numpy(float)[0], X.loc[origin].to_numpy(float), rtol=1e-6)


def test_design_matrix_rows_equal_inputs_at_and_targets_look_ahead():
    idx = pd.date_range("2024-01-01", periods=200, freq="1h")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"era5_swh": rng.uniform(0.5, 3, 200), "era5_msl": rng.normal(1008, 2, 200)}, index=idx)
    names = ["era5_swh_lag0h", "era5_swh_lag6h", "era5_msl_lag24h"]
    X, Y = design_matrix(df, names), target_matrix(df, (6, 72))
    for origin in idx[[30, 99, 127]]:
        np.testing.assert_allclose(X.loc[origin].to_numpy(float), inputs_at(df, names, origin).to_numpy(float)[0])
        assert Y.at[origin, 72] == df.at[origin + pd.Timedelta(hours=72), "era5_swh"]
    assert X.iloc[:24]["era5_msl_lag24h"].isna().all() and Y.iloc[-72:][72].isna().all()


def _synthetic_raw(days: int) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=days * 24, freq="1h")
    rng = np.random.default_rng(2)
    t = np.arange(len(idx))
    wind = 6 + 3 * np.sin(2 * np.pi * t / (24 * 9)) + rng.normal(0, 0.5, len(idx))
    swh = 0.5 + 0.12 * pd.Series(wind).rolling(12, min_periods=1).mean().to_numpy() + rng.normal(0, 0.05, len(idx))
    return pd.DataFrame({"era5_u10": wind, "era5_v10": 0.3 * wind, "era5_msl": 1008 + rng.normal(0, 1, len(idx)),
                         "era5_t2m": 300.0, "era5_sst": 302.0, "era5_swh": swh, "era5_mwd": 200.0, "era5_mwp": 7.0},
                        index=idx)


def test_retrain_keeps_blocks_apart_and_ignores_data_after_until(monkeypatch):
    from bobswh import trees
    seen = {}
    orig_fit = trees.DirectMultiStepRegressor.fit

    def spy(self, X, Y, X_es=None, Y_es=None, *a, **k):
        if X_es is not None:
            seen["fit"], seen["es"] = X.index, X_es.index
        return orig_fit(self, X, Y, X_es, Y_es, *a, **k)

    monkeypatch.setattr(trees.DirectMultiStepRegressor, "fit", spy)
    raw = _synthetic_raw(240)
    until = raw.index[-1] - pd.Timedelta(days=5)
    raw.loc[raw.index > until, "era5_swh"] = 99.0          # "forecast" values after `until` must never be used
    names = ["era5_swh_lag0h", "era5_swh_lag6h", "era5_u10_lag0h", "era5_u10_lag12h", "era5_msl_lag0h"]
    params = {"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 20}
    res = retrain_lightgbm(raw, names, params, until=until, eval_days=30, es_days=60, n_jobs=1)

    hmax = pd.Timedelta(hours=72)
    assert res.eval_end == until - hmax                    # last origin whose 72 h target is observed
    assert res.n_eval == 30 * 24
    assert (seen["fit"] + hmax).max() <= seen["es"].min()  # fit targets end before early stopping begins
    assert (seen["es"] + hmax).max() <= res.eval_start     # early-stopping targets end before the scored block
    sc = res.scores
    assert list(sc.index) == [6, 12, 24, 48, 72] and list(sc.columns) == ["Retrained on live data", "Persistence"]
    assert (sc["Retrained on live data"] < 1).all()        # would be huge if the 99 m values had leaked in
    assert (sc["Retrained on live data"].loc[[24, 48, 72]] < sc["Persistence"].loc[[24, 48, 72]]).all()
    assert len(res.model.models_) == 5 and res.model.feature_names_ == names
