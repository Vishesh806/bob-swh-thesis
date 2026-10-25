"""Encoder-decoder (Seq2Seq) LSTM for direct multi-horizon Hs forecasting.

Architecture
------------
Encoder  : (Bi)LSTM over the multivariate history [x_{t-L}, ..., x_t]. Its final
           hidden and cell states summarise the recent sea state and forcing.
Decoder  : LSTM initialised with the encoder states, run over one step per
           forecast horizon. Its inputs are only quantities known at forecast
           time: the normalised lead time h / h_max and the calendar phase
           (sin, cos of day of year) at t + h. No ERA5 value after t is used.
Attention: additive (Bahdanau) attention of each decoder step over all
           encoder steps, so each horizon can draw on the part of the history
           most relevant to it (e.g. the recent wind for 6 h, the swell
           history for 72 h).
Output   : a dense head per decoder step. With ``residual=True`` the network
           predicts the change from the value at the origin,
           Hs(t + h) = Hs(t) + delta_h, which makes persistence the starting
           point the network improves on.

The decoder is non-autoregressive: all horizons are produced in one pass and
no prediction is fed back as an input, so there is no error accumulation and
no train/inference mismatch from teacher forcing. The horizon steps are
irregular (6, 12, 24, 48, 72 h), which is why the lead time is an explicit
decoder input.

Peak-weighted loss
------------------
With e = y - y_hat (e > 0 is an under-prediction) and a soft extreme-sea-state
indicator s(y) = sigmoid((y - q) / tau), q a high quantile of training Hs:

    loss = mean[ (1 + s(y) * (beta + gamma * 1{e > 0})) * e^2 ]

In ordinary conditions (s ~ 0) this is the MSE. In extreme conditions
(s ~ 1) every error weighs (1 + beta) times as much and an under-prediction
(1 + beta + gamma) times as much (6x with the defaults beta = 1, gamma = 4).
Under-forecasting a cyclone peak is the costly error for coastal safety, so
the loss deliberately accepts some positive bias at the peaks in return for
fewer missed extremes. The sigmoid keeps the weight smooth in y; the loss
stays continuous in e because the weight only switches where e^2 = 0.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import keras
from keras import layers, ops


# ---------------------------------------------------------------- scaling
@dataclass
class SequenceScaler:
    """Per-feature standardisation of (n, steps, features) arrays, fitted on training data only."""

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SequenceScaler":
        flat = np.asarray(X, float).reshape(-1, X.shape[-1])
        self.mean_ = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        self.std_ = np.where(std > 1e-8, std, 1.0)        # constant features pass through centred
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("call fit first")
        return ((np.asarray(X, float) - self.mean_) / self.std_).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "SequenceScaler":
        return cls(np.asarray(d["mean"]), np.asarray(d["std"]))


# ---------------------------------------------------------------- loss
@keras.saving.register_keras_serializable(package="bobswh")
class PeakWeightedLoss(keras.losses.Loss):
    """Squared error with extra weight on extreme sea states and on under-prediction there.

    Parameters
    ----------
    threshold:
        q, the Hs (m) above which a sea state counts as extreme, e.g. the 95th
        percentile of training targets.
    tau:
        Width (m) of the sigmoid transition around q.
    beta:
        Extra weight on every error in extreme conditions.
    gamma:
        Further extra weight on under-predictions in extreme conditions.
    """

    def __init__(self, threshold: float, tau: float = 0.25, beta: float = 1.0, gamma: float = 4.0,
                 name: str = "peak_weighted_mse", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tau = float(tau)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, y_pred.dtype)
        e = y_true - y_pred
        s = ops.sigmoid((y_true - self.threshold) / self.tau)
        under = ops.cast(ops.greater(e, 0.0), e.dtype)
        weight = 1.0 + s * (self.beta + self.gamma * under)
        return ops.mean(weight * ops.square(e), axis=-1)

    def get_config(self) -> dict:
        return {**super().get_config(), "threshold": self.threshold, "tau": self.tau,
                "beta": self.beta, "gamma": self.gamma}


def peak_weighted_loss_numpy(y_true, y_pred, threshold, tau=0.25, beta=1.0, gamma=4.0) -> float:
    """Reference NumPy implementation of ``PeakWeightedLoss`` (used in the tests)."""
    e = np.asarray(y_true, float) - np.asarray(y_pred, float)
    s = 1.0 / (1.0 + np.exp(-(np.asarray(y_true, float) - threshold) / tau))
    return float(np.mean((1.0 + s * (beta + gamma * (e > 0))) * e ** 2))


# ---------------------------------------------------------------- model
def decoder_inputs(anchors: pd.DatetimeIndex, horizons_h: Sequence[int]) -> np.ndarray:
    """Known-future decoder covariates, shape (n, H, 3): h / h_max and sin, cos of day of year at t + h."""
    anchors = pd.DatetimeIndex(anchors)
    h_max = float(max(horizons_h))
    out = np.empty((len(anchors), len(horizons_h), 3), np.float32)
    for j, h in enumerate(horizons_h):
        valid = anchors + pd.Timedelta(hours=h)
        doy = valid.dayofyear.to_numpy() - 1 + valid.hour.to_numpy() / 24.0
        ang = 2.0 * np.pi * doy / 365.25
        out[:, j, 0] = h / h_max
        out[:, j, 1] = np.sin(ang)
        out[:, j, 2] = np.cos(ang)
    return out


def build_seq2seq(
    n_steps: int,
    n_features: int,
    n_horizons: int,
    n_decoder_features: int = 3,
    units: int = 64,
    encoder_layers: int = 1,
    bidirectional: bool = True,
    attention: bool = True,
    dropout: float = 0.1,
    residual: bool = True,
) -> keras.Model:
    """Functional Keras model: inputs [encoder (n, L, F), decoder (n, H, D), hs_origin (n, 1)] -> (n, H) in metres."""
    enc_in = keras.Input((n_steps, n_features), name="encoder_inputs")
    dec_in = keras.Input((n_horizons, n_decoder_features), name="decoder_inputs")
    hs0 = keras.Input((1,), name="hs_origin")

    x = enc_in
    for i in range(encoder_layers - 1):
        lstm = layers.LSTM(units, return_sequences=True, dropout=dropout, name=f"encoder_lstm_{i}")
        x = layers.Bidirectional(lstm, name=f"encoder_bilstm_{i}")(x) if bidirectional else lstm(x)
    last = layers.LSTM(units, return_sequences=True, return_state=True, dropout=dropout, name="encoder_lstm_last")
    if bidirectional:
        enc_seq, h_f, c_f, h_b, c_b = layers.Bidirectional(last, name="encoder_bilstm_last")(x)
        h = layers.Concatenate(name="encoder_h")([h_f, h_b])
        c = layers.Concatenate(name="encoder_c")([c_f, c_b])
        dec_units = 2 * units
    else:
        enc_seq, h, c = last(x)
        dec_units = units

    dec_seq = layers.LSTM(dec_units, return_sequences=True, dropout=dropout, name="decoder_lstm")(
        dec_in, initial_state=[h, c])
    if attention:
        context = layers.AdditiveAttention(name="attention")([dec_seq, enc_seq])
        dec_seq = layers.Concatenate(name="decoder_context")([dec_seq, context])
    z = layers.Dense(dec_units, activation="relu", name="head_hidden")(dec_seq)
    z = layers.Dropout(dropout, name="head_dropout")(z)
    out = layers.Reshape((n_horizons,), name="head_reshape")(layers.Dense(1, name="head_out")(z))
    if residual:
        out = layers.Add(name="hs_residual")([out, layers.Concatenate()([hs0] * n_horizons)])
    return keras.Model([enc_in, dec_in, hs0], out, name="seq2seq_lstm")


# ---------------------------------------------------------------- forecaster
@dataclass
class Seq2SeqConfig:
    units: int = 64
    encoder_layers: int = 1
    bidirectional: bool = True
    attention: bool = True
    dropout: float = 0.1
    residual: bool = True
    loss: str = "peak"                 # "peak" or "mse"
    peak_quantile: float = 0.95
    tau: float = 0.25
    beta: float = 1.0
    gamma: float = 4.0
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    patience: int = 10
    seed: int = 42


@dataclass
class Seq2SeqForecaster:
    """Train / predict / save wrapper around ``build_seq2seq``.

    Attributes
    ----------
    horizons_h:
        Forecast lead times (decoder steps).
    feature_names:
        Encoder input columns, in the order of the last axis of X.
    target_index:
        Position of the target's own history in ``feature_names`` (0 with
        ``WindowSpec(include_target_history=True)``); its unscaled value at
        the origin feeds the residual connection.
    """

    horizons_h: tuple[int, ...]
    feature_names: list[str]
    target_index: int = 0
    config: Seq2SeqConfig = field(default_factory=Seq2SeqConfig)
    scaler_: SequenceScaler = field(default_factory=SequenceScaler, init=False)
    threshold_: float | None = field(default=None, init=False)
    model_: keras.Model | None = field(default=None, init=False, repr=False)

    def _inputs(self, X: np.ndarray, anchors: pd.DatetimeIndex) -> list[np.ndarray]:
        hs0 = np.asarray(X[:, -1, self.target_index], np.float32)[:, None]
        return [self.scaler_.transform(X), decoder_inputs(anchors, self.horizons_h), hs0]

    def _loss(self):
        if self.config.loss == "mse":
            return keras.losses.MeanSquaredError()
        if self.config.loss == "peak":
            return PeakWeightedLoss(self.threshold_, self.config.tau, self.config.beta, self.config.gamma)
        raise ValueError(f"loss must be 'peak' or 'mse', got {self.config.loss!r}")

    def fit(self, X: np.ndarray, Y: np.ndarray, anchors: pd.DatetimeIndex,
            X_es: np.ndarray, Y_es: np.ndarray, anchors_es: pd.DatetimeIndex, verbose: int = 2) -> pd.DataFrame:
        """Fit on (X, Y); early-stop on the separate (X_es, Y_es) block. Returns the training history."""
        cfg = self.config
        keras.utils.set_random_seed(cfg.seed)
        self.scaler_ = SequenceScaler().fit(X)
        self.threshold_ = float(np.quantile(Y, cfg.peak_quantile))
        self.model_ = build_seq2seq(X.shape[1], X.shape[2], len(self.horizons_h), 3, cfg.units,
                                    cfg.encoder_layers, cfg.bidirectional, cfg.attention, cfg.dropout, cfg.residual)
        self.model_.compile(optimizer=keras.optimizers.Adam(cfg.learning_rate, clipnorm=1.0), loss=self._loss(),
                            metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(cfg.patience // 3, 1),
                                              min_lr=1e-5),
        ]
        hist = self.model_.fit(
            self._inputs(X, anchors), np.asarray(Y, np.float32),
            validation_data=(self._inputs(X_es, anchors_es), np.asarray(Y_es, np.float32)),
            epochs=cfg.epochs, batch_size=cfg.batch_size, shuffle=True, callbacks=callbacks, verbose=verbose,
        )
        return pd.DataFrame(hist.history).rename_axis("epoch")

    def predict(self, X: np.ndarray, anchors: pd.DatetimeIndex, batch_size: int = 2048) -> np.ndarray:
        """Forecasts of shape (n, H) in metres, clipped at 0."""
        if self.model_ is None:
            raise RuntimeError("call fit first")
        p = self.model_.predict(self._inputs(X, anchors), batch_size=batch_size, verbose=0)
        return np.maximum(np.asarray(p, float), 0.0)

    def save(self, folder: Path | str) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.model_.save(folder / "model.keras")
        meta = {"horizons_h": list(self.horizons_h), "feature_names": self.feature_names,
                "target_index": self.target_index, "config": asdict(self.config),
                "scaler": self.scaler_.to_dict(), "threshold": self.threshold_}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, folder: Path | str) -> "Seq2SeqForecaster":
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        f = cls(tuple(meta["horizons_h"]), meta["feature_names"], meta["target_index"], Seq2SeqConfig(**meta["config"]))
        f.scaler_ = SequenceScaler.from_dict(meta["scaler"])
        f.threshold_ = meta["threshold"]
        f.model_ = keras.models.load_model(folder / "model.keras")
        return f
