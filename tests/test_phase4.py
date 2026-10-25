"""Phase 4 tests: peak-weighted loss, scaling, Seq2Seq architecture, training, save/load, runner."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

import keras  # noqa: E402

from bobswh.metrics import horizon_metrics  # noqa: E402
from bobswh.phase4 import Phase4Runner  # noqa: E402
from bobswh.seq2seq import (PeakWeightedLoss, SequenceScaler, Seq2SeqConfig, Seq2SeqForecaster,  # noqa: E402
                            build_seq2seq, decoder_inputs, peak_weighted_loss_numpy)
from tests.synthetic import synthetic_frame  # noqa: E402

H = (6, 12, 24, 48, 72)
SPLITS = {"train": ("2016-01-01 00:00", "2017-06-30 23:00"),
          "val": ("2017-07-01 00:00", "2017-12-31 23:00"),
          "test": ("2018-01-01 00:00", "2018-12-31 23:00")}
TINY = Seq2SeqConfig(units=8, epochs=3, batch_size=512, patience=2)


# ---------------------------------------------------------------- loss
def test_peak_loss_matches_reference_and_is_asymmetric_at_peaks():
    rng = np.random.default_rng(0)
    y = rng.uniform(0.5, 8.0, (64, 5)).astype("float32")
    p = (y + rng.normal(0, 0.5, y.shape)).astype("float32")
    loss = PeakWeightedLoss(threshold=4.0)
    assert float(loss(y, p)) == pytest.approx(peak_weighted_loss_numpy(y, p, 4.0), rel=1e-5)
    calm, peak = np.array([[1.0]]), np.array([[8.0]])
    # far below the threshold: plain squared error, symmetric
    assert float(loss(calm, calm - 0.5)) == pytest.approx(0.25, rel=1e-3)
    assert float(loss(calm, calm + 0.5)) == pytest.approx(0.25, rel=1e-3)
    # far above it: over-prediction x(1 + beta) = 2, under-prediction x(1 + beta + gamma) = 6
    assert float(loss(peak, peak + 0.5)) == pytest.approx(0.25 * 2, rel=1e-3)
    assert float(loss(peak, peak - 0.5)) == pytest.approx(0.25 * 6, rel=1e-3)


def test_peak_loss_serialises():
    loss = PeakWeightedLoss(threshold=3.2, gamma=2.0)
    clone = PeakWeightedLoss.from_config(loss.get_config())
    assert (clone.threshold, clone.gamma, clone.beta) == (3.2, 2.0, 1.0)


# ---------------------------------------------------------------- inputs
def test_scaler_and_decoder_inputs():
    X = np.random.default_rng(1).normal(5, 2, (100, 10, 3))
    X[:, :, 2] = 7.0                                            # constant feature
    s = SequenceScaler().fit(X)
    Z = s.transform(X)
    np.testing.assert_allclose(Z[..., :2].reshape(-1, 2).mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(Z[..., 2], 0)
    np.testing.assert_allclose(SequenceScaler.from_dict(s.to_dict()).transform(X), Z)
    d = decoder_inputs(pd.DatetimeIndex(["2020-01-01 00:00", "2020-06-30 12:00"]), H)
    assert d.shape == (2, 5, 3)
    np.testing.assert_allclose(d[0, :, 0], np.array(H) / 72)
    np.testing.assert_allclose(np.hypot(d[..., 1], d[..., 2]), 1, atol=1e-6)


@pytest.mark.parametrize("bidirectional,attention,layers", [(True, True, 1), (False, False, 2), (True, False, 2)])
def test_architecture_shapes(bidirectional, attention, layers):
    m = build_seq2seq(25, 5, 5, units=8, bidirectional=bidirectional, attention=attention, encoder_layers=layers)
    out = m.predict([np.zeros((4, 25, 5)), np.zeros((4, 5, 3)), np.ones((4, 1))], verbose=0)
    assert out.shape == (4, 5)


def test_residual_connection_starts_from_persistence():
    m = build_seq2seq(13, 3, 5, units=8)
    head = m.get_layer("head_out")
    head.set_weights([np.zeros_like(w) for w in head.get_weights()])
    hs0 = np.array([[1.3], [4.2]], "float32")
    out = m.predict([np.random.rand(2, 13, 3), np.random.rand(2, 5, 3), hs0], verbose=0)
    np.testing.assert_allclose(out, np.repeat(hs0, 5, axis=1), atol=1e-6)


# ---------------------------------------------------------------- training
@pytest.fixture(scope="module")
def features(tmp_path_factory):
    folder = tmp_path_factory.mktemp("proc4")
    df, imputed = synthetic_frame(storms=True)
    df.to_parquet(folder / "SYN_features.parquet")
    imputed.to_parquet(folder / "SYN_imputed.parquet")
    return folder, df, imputed


def test_forecaster_fit_predict_save_load(features, tmp_path):
    folder = features[0]
    runner = Phase4Runner("SYN", features_dir=folder, splits=SPLITS, es_days=60, sequence_stride_h=6)
    data, idx, _ = runner.build_samples(features[1], features[2])
    fit, es, val = (data.subset(idx[k]) for k in ("fit", "es", "val"))
    assert data.feature_names[0] == "buoy_hs" and data.X.shape[1] == 13      # 72 h / 6 h + 1 steps
    fc = Seq2SeqForecaster(H, data.feature_names, 0, TINY)
    hist = fc.fit(fit.X, fit.Y, fit.anchors, es.X, es.Y, es.anchors, verbose=0)
    assert {"loss", "val_loss", "rmse"} <= set(hist.columns) and 1 <= len(hist) <= 3
    assert fc.threshold_ == pytest.approx(np.quantile(fit.Y, 0.95))          # threshold from the fit block only
    p = fc.predict(val.X, val.anchors)
    assert p.shape == val.Y.shape and (p >= 0).all() and np.isfinite(p).all()
    fc.save(tmp_path / "m")
    np.testing.assert_allclose(Seq2SeqForecaster.load(tmp_path / "m").predict(val.X, val.anchors), p, atol=1e-5)


def test_peak_metrics_columns():
    obs = np.array([[1.0], [5.0], [6.0]])
    m = horizon_metrics(obs, obs - np.array([[0.0], [1.0], [-1.0]]), (6,), peak_threshold=5.0)
    assert m.loc[6, "n_peak"] == 2 and m.loc[6, "under_rate_peak"] == pytest.approx(0.5)
    assert m.loc[6, "rmse_peak"] == pytest.approx(1.0) and m.loc[6, "bias_peak"] == pytest.approx(0.0)


def test_runner_end_to_end(features, tmp_path):
    folder = features[0]
    runner = Phase4Runner("SYN", features_dir=folder, out_dir=tmp_path / "res", model_dir=tmp_path / "models",
                          splits=SPLITS, es_days=60, sequence_stride_h=6, model_config=TINY, verbose=0)
    metrics = runner.run()
    assert set(metrics.index.get_level_values("model")) == {"lstm_peak", "lstm_mse"}
    assert {"rmse_peak", "bias_peak", "under_rate_peak"} <= set(metrics.columns)
    out = tmp_path / "res" / "SYN"
    for f in ("metrics.csv", "predictions.parquet", "history_peak.csv", "history_mse.csv", "split_summary.csv"):
        assert (out / f).exists(), f
    preds = pd.read_parquet(out / "predictions.parquet")
    assert preds.index[preds["split"] == "test"].min() - pd.Timedelta(hours=72) >= pd.Timestamp(SPLITS["test"][0])
    assert keras.models.load_model(tmp_path / "models" / "SYN_lstm_peak" / "model.keras") is not None
