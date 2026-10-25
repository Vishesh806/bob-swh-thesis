"""Reference forecasts that every learned model has to beat.

* Persistence:  Hs(t + h) = Hs(t). Hard to beat at short lead times because
  hourly Hs is strongly autocorrelated.
* Climatology:  Hs(t + h) = mean Hs on that calendar day in the training
  years. The skill floor at long lead times.

A model that does not beat persistence at 6-12 h and climatology at 72 h has
learned nothing useful, so the thesis reports skill relative to both.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


class PersistenceForecaster:
    """Repeat the value observed at the forecast origin for every horizon."""

    def __init__(self, horizons_h: Sequence[int]) -> None:
        self.horizons_h = tuple(horizons_h)

    def fit(self, *_, **__) -> "PersistenceForecaster":
        return self

    def predict(self, current: np.ndarray | pd.Series) -> np.ndarray:
        """``current``: Hs at each forecast origin (the ``{target}_lag0h`` column)."""
        x = np.asarray(current, float)
        return np.repeat(x[:, None], len(self.horizons_h), axis=1)


class ClimatologyForecaster:
    """Day-of-year mean of the training record, smoothed with a circular moving average.

    Parameters
    ----------
    smooth_days:
        Half-width of the smoothing window. +/- 15 days keeps the monsoon
        onset and withdrawal while averaging out individual storms.
    """

    def __init__(self, horizons_h: Sequence[int], smooth_days: int = 15) -> None:
        self.horizons_h = tuple(horizons_h)
        self.smooth_days = smooth_days
        self.table_: np.ndarray | None = None

    def fit(self, series: pd.Series) -> "ClimatologyForecaster":
        """``series``: target values on a DatetimeIndex, restricted to the training period."""
        s = series.dropna()
        doy = np.minimum(s.index.dayofyear.to_numpy(), 365) - 1      # fold 31 Dec of leap years into day 365
        sums = np.bincount(doy, weights=s.to_numpy(), minlength=365)
        counts = np.bincount(doy, minlength=365).astype(float)
        k = self.smooth_days
        kernel = np.ones(2 * k + 1)
        wrap = lambda a: np.convolve(np.r_[a[-k:], a, a[:k]], kernel, mode="valid")
        with np.errstate(invalid="ignore"):
            self.table_ = wrap(sums) / wrap(counts)
        return self

    def predict(self, anchors: pd.DatetimeIndex) -> np.ndarray:
        if self.table_ is None:
            raise RuntimeError("call fit first")
        out = np.empty((len(anchors), len(self.horizons_h)))
        for j, h in enumerate(self.horizons_h):
            valid = pd.DatetimeIndex(anchors) + pd.Timedelta(hours=h)
            out[:, j] = self.table_[np.minimum(valid.dayofyear.to_numpy(), 365) - 1]
        return out
