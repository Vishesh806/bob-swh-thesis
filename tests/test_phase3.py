"""Phase 3 tests: metrics, baselines, SARIMAX fast path, DMS trees, Optuna tuning, runner.

Synthetic physics: wind follows a slow AR(1) process, and Hs responds to the
wind averaged over the previous 6-24 h plus a swell component. Hs at t + h
therefore depends on wind the forecaster has partly seen, so a model that uses
the forcing should beat persistence at the longer horizons.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from bobswh.baselines import ClimatologyForecaster, PersistenceForecaster
from bobswh.metrics import horizon_metrics, multi_horizon_rmse
from bobswh.phase3 import Phase3Runner, select_inputs
from bobswh.sarimax import SARIMAXForecaster, select_order
from bobswh.trees import DirectMultiStepRegressor
from bobswh.tuning import MultiHorizonTuner
from bobswh.windowing import ChronologicalSplitter, carve_early_stopping_block
from tests.synthetic import synthetic_frame

pytest.importorskip("lightgbm")
pytest.importorskip("catboost")
pytest.importorskip("optuna")

H = (6, 12, 24, 48, 72)
SPLITS = {"train": ("2016-01-01 00:00", "2017-06-30 23:00"),
          "val": ("2017-07-01 00:00", "2017-12-31 23:00"),
          "test": ("2018-01-01 00:00", "2018-12-31 23:00")}


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    folder = tmp_path_factory.mktemp("proc")
    df, imputed = synthetic_frame()
    df.to_parquet(folder / "SYN_features.parquet")
    imputed.to_parquet(folder / "SYN_imputed.parquet")
    return folder, df, imputed


def small_runner(folder, tmp, **kw) -> Phase3Runner:
    base = dict(buoy="SYN", features_dir=folder, out_dir=tmp / "results", model_dir=tmp / "models",
                splits=SPLITS, es_days=60, n_trials=0, max_rounds=300, early_stopping_rounds=30,
                sarimax_fit_days=60, n_jobs=2)
    base.update(kw)
    return Phase3Runner(**base)


# ---------------------------------------------------------------- metrics and baselines
def test_metrics_known_values():
    obs = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]]).repeat(1, axis=0)
    obs = np.column_stack([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    pred = obs + np.array([[0.0, 1.0]])
    m = horizon_metrics(obs, pred, (6, 12))
    assert m.loc[6, "rmse"] == 0 and m.loc[6, "r2"] == 1
    assert m.loc[12, "rmse"] == pytest.approx(1.0) and m.loc[12, "bias"] == pytest.approx(1.0)
    assert m.loc[12, "si"] == pytest.approx(0.5) and m.loc[12, "si_ub"] == pytest.approx(0.0)
    assert multi_horizon_rmse(obs, pred) == pytest.approx(np.sqrt(0.5))


def test_persistence_and_climatology():
    p = PersistenceForecaster(H).predict(np.array([1.0, 2.0]))
    assert p.shape == (2, 5) and (p[1] == 2.0).all()
    idx = pd.date_range("2015-01-01", "2016-12-31 23:00", freq="1h")
    s = pd.Series(np.where(idx.month == 7, 3.0, 1.0), index=idx)
    clim = ClimatologyForecaster(H, smooth_days=0).fit(s)
    out = clim.predict(pd.DatetimeIndex(["2020-07-10", "2020-01-10"]))
    assert out[0, 0] == pytest.approx(3.0) and out[1, -1] == pytest.approx(1.0)


# ---------------------------------------------------------------- SARIMAX
def test_sarimax_fast_path_matches_statsmodels():
    rng = np.random.default_rng(1)
    n = 3000
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = 1.6 * x[t - 1] - 0.65 * x[t - 2] + rng.normal(0, 0.05)
    s = pd.Series(1.5 + x, index=pd.date_range("2020-01-01", periods=n, freq="1h"))
    s.iloc[2500:2510] = np.nan
    f = SARIMAXForecaster(order=(2, 0, 1), trend="c", fit_window_days=None).fit(s.iloc[:2000])
    origins = [2100, 2520, 2900]
    fast = f.forecast_origins(s, s.index[origins], (6, 12, 24))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, a in enumerate(origins):
            ref = f.result_.apply(s.to_numpy()[: a + 1], refit=False).forecast(24)[[5, 11, 23]]
            np.testing.assert_allclose(fast[i], np.maximum(ref, 0), atol=1e-9)
    # a forecast from origin t must not change when data after t change
    s2 = s.copy()
    s2.iloc[2101:] += 5.0
    np.testing.assert_allclose(f.forecast_origins(s2, s.index[[2100]], (6, 12, 24)), fast[[0]], atol=1e-9)


def test_sarimax_order_selection():
    rng = np.random.default_rng(2)
    s = pd.Series(np.cumsum(rng.normal(0, 0.05, 1500)) + 2, index=pd.date_range("2020-01-01", periods=1500, freq="1h"))
    best, table = select_order(s, [(1, 1, 0), (0, 1, 1)], fit_window_days=None)
    assert len(table) == 2 and best.aic == pytest.approx(table["aic"].min())


# ---------------------------------------------------------------- trees and tuning
def _blocks(features):
    _, df, imputed = features
    runner = small_runner(features[0], features[0])
    X, Y, idx, _ = runner.build_samples(df, imputed)
    return {k: (X.iloc[v], Y.iloc[v].to_numpy()) for k, v in idx.items()}


@pytest.mark.parametrize("backend", ["lightgbm", "catboost", "xgboost"])
def test_dms_backends(features, backend, tmp_path):
    pytest.importorskip(backend)
    b = _blocks(features)
    m = DirectMultiStepRegressor(backend, H, max_rounds=200, early_stopping_rounds=20, n_jobs=2)
    m.fit(*b["fit"], *b["es"])
    pred = m.predict(b["val"][0])
    assert pred.shape == (len(b["val"][0]), 5) and (pred >= 0).all()
    assert len(m.models_) == 5 and all(1 <= k <= 200 for k in m.best_iterations_)
    imp = m.feature_importance()
    np.testing.assert_allclose(imp.sum().to_numpy(), 1.0)
    m.save(tmp_path / "m.joblib")
    np.testing.assert_allclose(DirectMultiStepRegressor.load(tmp_path / "m.joblib").predict(b["val"][0]), pred)


def test_tuner_shared_and_per_horizon(features):
    b = _blocks(features)
    args = (*b["fit"], *b["es"], *b["val"])
    shared = MultiHorizonTuner("lightgbm", H, n_trials=3, max_rounds=100, early_stopping_rounds=10, n_jobs=2).tune(*args)
    assert isinstance(shared.params, dict) and shared.params["subsample_freq"] == 1
    assert np.isfinite(shared.best_value)
    per = MultiHorizonTuner("lightgbm", (6, 24), n_trials=2, per_horizon=True, max_rounds=50,
                            early_stopping_rounds=10, n_jobs=2).tune(b["fit"][0], b["fit"][1][:, [0, 2]],
                                                                     b["es"][0], b["es"][1][:, [0, 2]],
                                                                     b["val"][0], b["val"][1][:, [0, 2]])
    assert isinstance(per.params, list) and len(per.params) == 2


def test_early_stopping_block_is_disjoint():
    s = carve_early_stopping_block(SPLITS, 60)
    assert set(s) == {"fit", "es", "val", "test"}
    assert pd.Timestamp(s["fit"][1]) < pd.Timestamp(s["es"][0])
    assert pd.Timestamp(s["es"][1]) == pd.Timestamp(SPLITS["train"][1])
    idx = pd.date_range("2016-01-01", "2018-12-31 23:00", freq="1h")
    pos = ChronologicalSplitter(s, 72, 72).split(idx)
    assert idx[pos["fit"]].max() + pd.Timedelta(hours=72) < idx[pos["es"]].min() - pd.Timedelta(hours=72)


def test_select_inputs_excludes_buoy_exog_and_raw_directions(features):
    df = features[1].assign(buoy_wspd=5.0)
    cols = select_inputs(df, "buoy_hs")
    assert "buoy_wspd" not in cols and "buoy_hs" not in cols and "era5_mwd" not in cols
    assert "buoy_hs_tend3h" in cols and "era5_tau" in cols
    assert "buoy_wspd" in select_inputs(df, "buoy_hs", include_buoy_exog=True)


# ---------------------------------------------------------------- end to end
def test_runner_end_to_end(features, tmp_path):
    folder, df, _ = features
    runner = small_runner(folder, tmp_path, backends=("lightgbm", "catboost"), n_trials=2)
    metrics = runner.run()
    models = set(metrics.index.get_level_values("model"))
    assert models == {"persistence", "climatology", "sarimax", "lightgbm", "catboost"}
    n = metrics["n"].groupby(["split", "horizon_h"]).nunique()
    assert (n == 1).all()                                             # every model scored on the same origins
    out = tmp_path / "results" / "SYN"
    preds = pd.read_parquet(out / "predictions.parquet")
    test_origins = preds.index[preds["split"] == "test"]
    assert test_origins.min() - pd.Timedelta(hours=72) >= pd.Timestamp(SPLITS["test"][0])
    assert json.loads((out / "lightgbm_params.json").read_text())["params"]
    rmse = metrics.xs("test", level="split")["rmse"].unstack("horizon_h")
    assert rmse.loc["lightgbm", 24] < rmse.loc["persistence", 24]      # the forcing carries real skill
    assert rmse.loc["lightgbm", 72] < rmse.loc["climatology", 72]
    assert (tmp_path / "models" / "SYN_catboost.joblib").exists()
