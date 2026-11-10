"""Live-forecast tests: Open-Meteo parsing, wind components, and input rows identical to the training layout."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bobswh.live import frame_from_json, inputs_at, latest_origin, wind_components
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
