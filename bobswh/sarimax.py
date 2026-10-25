"""(S)ARIMA baseline for a single buoy, evaluated from many forecast origins.

Parameters are estimated once, on the training period only. At test time the
fitted state-space model is run through the full record with those
parameters frozen. The Kalman filter gives the filtered state a_{t|t}, which
uses observations up to t only, and the h-step forecast from origin t is

    a_{t+h|t} = c + T a_{t+h-1|t},    y_{t+h|t} = d + Z a_{t+h|t},

computed for all origins at once. This is the same forecast statsmodels
returns from a re-filtered model at each origin (checked in the tests), but
it costs one filter pass instead of one per origin, which makes scoring
tens of thousands of test origins feasible. Missing observations are handled
by the filter itself (the state is propagated without an update).

On exogenous regressors: SARIMAX can take ERA5 forcing as X, but a forecast
from origin t would then need the forcing at t + h, which is not known at
forecast time without an NWP forecast. Feeding it reanalysis values would be
a "perfect prognosis" experiment, not a forecast. The baseline is therefore
univariate. The tree and LSTM models use forcing only up to t.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .features import step_hours

log = logging.getLogger(__name__)

_FLOOR = 0.01   # m, floor before a log transform


@dataclass
class SARIMAXForecaster:
    """Univariate (S)ARIMA forecaster with frozen training-period parameters.

    Attributes
    ----------
    order, seasonal_order, trend:
        As in ``statsmodels.tsa.statespace.sarimax.SARIMAX``.
    fit_window_days:
        Fit on the last N days of the training series. Two years of hourly
        data are enough to pin down a low-order ARIMA, and fitting on the
        whole six-year training period would take far longer without
        changing the parameters much.
    log_transform:
        Model log(Hs). Stabilises the variance (Hs fluctuations grow with
        Hs); forecasts are back-transformed with exp, i.e. the median.
    """

    order: tuple[int, int, int] = (2, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
    trend: str | None = None
    fit_window_days: int | None = 730
    log_transform: bool = False
    maxiter: int = 200
    result_: object = field(default=None, init=False, repr=False)
    step_h_: float = field(default=np.nan, init=False)

    def _transform(self, y: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(y, _FLOOR)) if self.log_transform else y

    def _inverse(self, y: np.ndarray) -> np.ndarray:
        return np.exp(y) if self.log_transform else np.maximum(y, 0.0)

    def _model(self, y: np.ndarray) -> SARIMAX:
        return SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order, trend=self.trend,
                       enforce_stationarity=True, enforce_invertibility=True)

    def fit(self, series: pd.Series) -> "SARIMAXForecaster":
        """Estimate parameters on ``series`` (training period only, regular index)."""
        self.step_h_ = step_hours(series.index)
        if self.fit_window_days:
            series = series[series.index > series.index[-1] - pd.Timedelta(days=self.fit_window_days)]
        y = self._transform(series.to_numpy(float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result_ = self._model(y).fit(disp=False, maxiter=self.maxiter)
        log.info("SARIMAX%s%s: AIC %.1f on %d obs", self.order, self.seasonal_order, self.result_.aic, len(y))
        return self

    @property
    def aic(self) -> float:
        return float(self.result_.aic)

    def forecast_origins(self, series: pd.Series, anchors: pd.DatetimeIndex, horizons_h: Sequence[int]) -> np.ndarray:
        """Forecasts of shape (len(anchors), len(horizons_h)) from every origin in ``anchors``.

        ``series`` is the full record on the same regular grid used in ``fit``.
        Only values at or before each origin influence its forecast.
        """
        if self.result_ is None:
            raise RuntimeError("call fit first")
        if step_hours(series.index) != self.step_h_:
            raise ValueError("series step differs from the one used in fit")
        y = self._transform(series.to_numpy(float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self.result_.apply(y, refit=False)
        ssm = res.model.ssm
        Z = ssm["design"][:, :, 0] if ssm["design"].ndim == 3 else ssm["design"]
        T = ssm["transition"][:, :, 0] if ssm["transition"].ndim == 3 else ssm["transition"]
        c = np.ravel(ssm["state_intercept"][:, 0] if ssm["state_intercept"].ndim == 2 else ssm["state_intercept"])
        d = float(np.ravel(ssm["obs_intercept"])[0])
        pos = series.index.get_indexer(pd.DatetimeIndex(anchors))
        if (pos < 0).any():
            raise KeyError("some anchors are not in the series index")
        state = res.filter_results.filtered_state[:, pos]                 # a_{t|t}, shape (k, n)
        steps = [int(round(h / self.step_h_)) for h in horizons_h]
        out = np.empty((len(pos), len(steps)))
        for s in range(1, max(steps) + 1):
            state = c[:, None] + T @ state
            for j in [j for j, k in enumerate(steps) if k == s]:
                out[:, j] = d + (Z @ state)[0]
        return self._inverse(out)


def select_order(
    series: pd.Series,
    candidates: Iterable[tuple[int, int, int]] = ((1, 1, 0), (1, 1, 1), (2, 1, 1), (2, 1, 2), (3, 1, 1), (2, 0, 1)),
    **kwargs,
) -> tuple[SARIMAXForecaster, pd.DataFrame]:
    """Fit each candidate order on the training series and keep the lowest AIC.

    Returns (best fitted forecaster, table of candidates and their AIC).
    """
    rows, best = [], None
    for order in candidates:
        try:
            f = SARIMAXForecaster(order=tuple(order), **kwargs).fit(series)
        except Exception as exc:          # non-convergence of one candidate must not stop the search
            log.warning("order %s failed: %s", order, exc)
            continue
        rows.append({"order": str(tuple(order)), "aic": f.aic, "bic": float(f.result_.bic)})
        if best is None or f.aic < best.aic:
            best = f
    if best is None:
        raise RuntimeError("no candidate order could be fitted")
    return best, pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
