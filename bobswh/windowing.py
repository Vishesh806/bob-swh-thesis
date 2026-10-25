"""Sliding-window sample generation and leakage-free chronological splits.

Notation: x_t is the feature vector at time t, y the target (Hs), L the
lookback, h_1..h_H the forecast horizons. A sample anchored at the forecast
origin t is

    X_t = [x_{t-L}, x_{t-L+s}, ..., x_t]      (inputs, all at or before t)
    Y_t = [y_{t+h_1}, ..., y_{t+h_H}]          (direct multi-step targets)

Every horizon is a target of its own (Direct Multi-Step / multi-output), so no
forecast is ever fed back in as an input.

A sample is kept only if every input and every target is finite, and (by
default) no target was imputed. Long buoy gaps therefore remove the samples
they touch instead of being bridged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from . import config
from .features import step_hours


def _to_steps(hours: float, dt_h: float, what: str) -> int:
    n = hours / dt_h
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"{what} = {hours} h is not a multiple of the {dt_h} h time step")
    return int(round(n))


@dataclass(frozen=True)
class WindowSpec:
    """Shape of the supervised samples.

    Attributes
    ----------
    target_col:
        Series to forecast, e.g. ``buoy_hs`` (or ``era5_swh`` for ERA5-only runs).
    input_cols:
        Exogenous inputs. The target's own history is prepended when
        ``include_target_history`` is True.
    lookback_h:
        L. The encoder sees [t - L, t], i.e. L / stride + 1 time steps.
    horizons_h:
        Forecast lead times.
    sequence_stride_h:
        Spacing of encoder steps (3 gives 25 steps for L = 72 h).
    anchor_stride_h:
        Spacing of forecast origins. Values > 1 thin highly autocorrelated
        neighbouring samples and cut memory.
    tabular_lags_h:
        Lags given to the columns in ``tabular_lag_cols`` in the flat (tree) layout.
    tabular_lag_cols:
        Columns that get the full lag set; every other input gets lag 0 only.
        None = the target column alone.
    """

    target_col: str = "buoy_hs"
    input_cols: tuple[str, ...] = ()
    include_target_history: bool = True
    lookback_h: int = config.LOOKBACK_H
    horizons_h: tuple[int, ...] = config.HORIZONS_H
    sequence_stride_h: int = 1
    anchor_stride_h: int = 1
    tabular_lags_h: tuple[int, ...] = (0, 1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 72)
    tabular_lag_cols: tuple[str, ...] | None = None

    @property
    def all_inputs(self) -> list[str]:
        cols = list(self.input_cols)
        if self.include_target_history and self.target_col not in cols:
            cols.insert(0, self.target_col)
        return cols

    @property
    def target_names(self) -> list[str]:
        return [f"{self.target_col}_t+{h}h" for h in self.horizons_h]


@dataclass
class WindowedData:
    """Supervised samples. X is (n, steps, features) for sequences or (n, features) for tables."""

    X: np.ndarray
    Y: np.ndarray
    anchors: pd.DatetimeIndex
    feature_names: list[str]
    target_names: list[str]

    def __len__(self) -> int:
        return len(self.anchors)

    def subset(self, idx: np.ndarray) -> "WindowedData":
        return WindowedData(self.X[idx], self.Y[idx], self.anchors[idx], self.feature_names, self.target_names)


class SlidingWindowGenerator:
    """Build (X, Y) samples from a regular, feature-engineered frame.

    Parameters
    ----------
    spec:
        Window definition.
    imputed:
        Boolean mask from preprocessing (same index as the data). Targets
        flagged as imputed are treated as missing unless
        ``allow_imputed_targets``, so models are never trained or scored
        against synthetic truth.
    dtype:
        float32 halves memory; 10 years x 73 steps x 30 features is ~0.8 GB
        in float32 with anchor_stride_h = 1.
    """

    def __init__(
        self,
        spec: WindowSpec,
        imputed: pd.DataFrame | None = None,
        allow_imputed_targets: bool = False,
        dtype: type = np.float32,
    ) -> None:
        self.spec = spec
        self.imputed = imputed
        self.allow_imputed_targets = allow_imputed_targets
        self.dtype = dtype

    def _prepare(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        missing = [c for c in self.spec.all_inputs + [self.spec.target_col] if c not in df]
        if missing:
            raise KeyError(f"columns not in frame: {missing}")
        dt = step_hours(df.index)
        values = df[self.spec.all_inputs].to_numpy(float)
        target = df[self.spec.target_col].to_numpy(float).copy()
        if self.imputed is not None and not self.allow_imputed_targets and self.spec.target_col in self.imputed:
            flag = self.imputed[self.spec.target_col].reindex(df.index, fill_value=False).to_numpy(bool)
            target[flag] = np.nan
        h = np.array([_to_steps(x, dt, "horizon") for x in self.spec.horizons_h])
        return values, target, h, dt

    def _anchors(self, n_rows: int, back: int, h_max: int, dt: float) -> np.ndarray:
        stride = _to_steps(self.spec.anchor_stride_h, dt, "anchor_stride_h")
        return np.arange(back, n_rows - h_max, stride)

    @staticmethod
    def _window_complete(values: np.ndarray, anchors: np.ndarray, back: int) -> np.ndarray:
        """True where every input is finite on every row of [t - back, t].

        The whole window is checked, not only the rows a layout samples, so the
        sequence and tabular layouts keep exactly the same forecast origins.
        """
        bad = np.r_[0, np.cumsum(~np.isfinite(values).all(axis=1))]
        return (bad[anchors + 1] - bad[anchors - back]) == 0

    def sequences(self, df: pd.DataFrame) -> WindowedData:
        """3-D samples for recurrent / Seq2Seq models: X has shape (n, L/stride + 1, F)."""
        values, target, h, dt = self._prepare(df)
        s = _to_steps(self.spec.sequence_stride_h, dt, "sequence_stride_h")
        back = _to_steps(self.spec.lookback_h, dt, "lookback_h")
        if back % s:
            raise ValueError("lookback_h must be a multiple of sequence_stride_h")
        offsets = np.arange(-back, 1, s)
        anchors = self._anchors(len(df), back, int(h.max()), dt)
        ok = self._window_complete(values, anchors, back)
        Y = target[anchors[:, None] + h[None, :]]
        ok &= np.isfinite(Y).all(axis=1)
        anchors = anchors[ok]
        X = values.astype(self.dtype)[anchors[:, None] + offsets[None, :]]
        return WindowedData(X, Y[ok].astype(self.dtype), df.index[anchors], self.spec.all_inputs, self.spec.target_names)

    def tabular(self, df: pd.DataFrame, require_complete_inputs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Flat lag features for gradient-boosted trees.

        Column ``{col}_lag{k}h`` holds x_{t-k}. Returns (X, Y) indexed by the
        forecast origin t. Gradient boosting tolerates missing inputs; keep
        ``require_complete_inputs`` True when comparing against the LSTM so
        that every model is trained and scored on the same samples (the same
        full-window completeness rule as ``sequences``).
        """
        values, target, h, dt = self._prepare(df)
        lag_cols = set(self.spec.tabular_lag_cols or (self.spec.target_col,))
        lag_steps = {k: _to_steps(k, dt, "tabular lag") for k in self.spec.tabular_lags_h}
        back = max(max(lag_steps.values()), _to_steps(self.spec.lookback_h, dt, "lookback_h"))
        anchors = self._anchors(len(df), back, int(h.max()), dt)
        cols: dict[str, np.ndarray] = {}
        for j, name in enumerate(self.spec.all_inputs):
            lags = self.spec.tabular_lags_h if name in lag_cols else (0,)
            for k in lags:
                cols[f"{name}_lag{k}h"] = values[anchors - lag_steps.get(k, _to_steps(k, dt, "lag")), j]
        index = df.index[anchors]
        X = pd.DataFrame(cols, index=index)
        Y = pd.DataFrame(target[anchors[:, None] + h[None, :]], index=index, columns=self.spec.target_names)
        ok = Y.notna().all(axis=1).to_numpy()
        if require_complete_inputs:
            ok &= self._window_complete(values, anchors, back)
        return X[ok].astype(self.dtype), Y[ok].astype(self.dtype)


def carve_early_stopping_block(
    splits: Mapping[str, tuple[str, str]],
    days: int = config.EARLY_STOPPING_DAYS,
    train: str = "train",
) -> dict[str, tuple[str, str]]:
    """Split the training block into ``fit`` and a trailing ``es`` block of ``days`` days.

    The other blocks are unchanged. The containment rule of
    ``ChronologicalSplitter`` then purges the fit/es boundary like any other.
    """
    start, end = (pd.Timestamp(t) for t in splits[train])
    es_start = end - pd.Timedelta(days=days) + pd.Timedelta(hours=1)
    if es_start <= start:
        raise ValueError("early-stopping block would cover the whole training period")
    out = {k: v for k, v in splits.items() if k != train}
    out["fit"] = (str(start), str(es_start - pd.Timedelta(hours=1)))
    out["es"] = (str(es_start), str(end))
    return out


@dataclass(frozen=True)
class ChronologicalSplitter:
    """Assign samples to train / val / test blocks with no information crossing a boundary.

    A sample with origin t belongs to block [start, end] only if all of its
    targets lie inside the block, t + max(h) <= end, and (``contain_inputs``)
    its whole input window does too, t - L >= start. The first condition
    removes the classic multi-step leak, where the last training samples
    have targets inside the validation period. The second also keeps
    validation inputs out of the training period, so the blocks are fully
    independent. About (L + max h) hours are lost at each boundary, i.e. 6 days
    for L = 72 h and h = 72 h, which is negligible over 10 years.
    """

    boundaries: Mapping[str, tuple[str, str]] = field(default_factory=lambda: dict(config.SPLITS))
    lookback_h: int = config.LOOKBACK_H
    max_horizon_h: int = max(config.HORIZONS_H)
    contain_inputs: bool = True

    def _intervals(self) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
        iv = sorted(((n, pd.Timestamp(a), pd.Timestamp(b)) for n, (a, b) in self.boundaries.items()), key=lambda r: r[1])
        for (n1, s1, e1), (n2, s2, _) in zip(iv, iv[1:]):
            if s2 <= e1:
                raise ValueError(f"split '{n2}' starts before '{n1}' ends")
        for n, s, e in iv:
            if e <= s:
                raise ValueError(f"split '{n}' ends before it starts")
        return iv

    def split(self, anchors: pd.DatetimeIndex) -> dict[str, np.ndarray]:
        """Integer positions into ``anchors`` for each block."""
        lb = pd.Timedelta(hours=self.lookback_h if self.contain_inputs else 0)
        hz = pd.Timedelta(hours=self.max_horizon_h)
        anchors = pd.DatetimeIndex(anchors)
        return {
            name: np.flatnonzero((anchors - lb >= start) & (anchors + hz <= end))
            for name, start, end in self._intervals()
        }

    def summary(self, anchors: pd.DatetimeIndex, splits: Mapping[str, np.ndarray]) -> pd.DataFrame:
        """Table of each block's sample count, first input time and last target time."""
        lb, hz = pd.Timedelta(hours=self.lookback_h), pd.Timedelta(hours=self.max_horizon_h)
        rows = []
        for name, idx in splits.items():
            a = pd.DatetimeIndex(anchors)[idx]
            rows.append({
                "split": name,
                "n_samples": len(a),
                "first_origin": a.min() if len(a) else pd.NaT,
                "last_origin": a.max() if len(a) else pd.NaT,
                "first_input_time": a.min() - lb if len(a) else pd.NaT,
                "last_target_time": a.max() + hz if len(a) else pd.NaT,
            })
        return pd.DataFrame(rows).set_index("split")
