"""Forecast verification metrics, per horizon and pooled over horizons.

For observations o and predictions p at one horizon (n samples):

    RMSE  = sqrt(mean((p - o)^2))
    MAE   = mean(|p - o|)
    bias  = mean(p - o)                     (> 0: over-prediction)
    R^2   = 1 - sum((p - o)^2) / sum((o - mean(o))^2)
    SI    = RMSE / mean(o)                  (scatter index, the usual wave-model convention)
    SI_ub = sqrt(mean(((p - mean(p)) - (o - mean(o)))^2)) / mean(o)   (bias removed)
    CC    = Pearson correlation

With a peak threshold q, the same RMSE and bias are also computed on the
extreme subset o >= q (``rmse_peak``, ``bias_peak``), together with the
under-prediction rate there (``under_rate_peak``, the share of extreme cases
with p < o).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _as_2d(a) -> np.ndarray:
    a = np.asarray(a, float)
    return a[:, None] if a.ndim == 1 else a


def rmse(y_true, y_pred) -> np.ndarray:
    """RMSE per column."""
    e = _as_2d(y_pred) - _as_2d(y_true)
    return np.sqrt(np.mean(e ** 2, axis=0))


def multi_horizon_rmse(y_true, y_pred) -> float:
    """Pooled RMSE over all samples and horizons, sqrt(mean_h RMSE_h^2).

    This is the Optuna objective. Errors grow with lead time, so the long
    horizons carry most of the weight; that is deliberate, since the 48-72 h
    forecasts are the ones that matter for operational planning.
    """
    e = _as_2d(y_pred) - _as_2d(y_true)
    return float(np.sqrt(np.mean(e ** 2)))


def horizon_metrics(y_true, y_pred, horizons_h: Sequence[int], peak_threshold: float | None = None) -> pd.DataFrame:
    """Table indexed by horizon (h) with n, RMSE, MAE, bias, R^2, SI, SI_ub and CC,
    plus the extreme-subset scores when ``peak_threshold`` is given."""
    o, p = _as_2d(y_true), _as_2d(y_pred)
    if o.shape != p.shape or o.shape[1] != len(horizons_h):
        raise ValueError(f"shape mismatch: y_true {o.shape}, y_pred {p.shape}, {len(horizons_h)} horizons")
    rows = []
    for j, h in enumerate(horizons_h):
        oj, pj = o[:, j], p[:, j]
        ok = np.isfinite(oj) & np.isfinite(pj)
        oj, pj = oj[ok], pj[ok]
        e = pj - oj
        obar = oj.mean()
        sst = np.sum((oj - obar) ** 2)
        row = {
            "horizon_h": h,
            "n": int(ok.sum()),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)),
            "r2": float(1.0 - np.sum(e ** 2) / sst) if sst > 0 else np.nan,
            "si": float(np.sqrt(np.mean(e ** 2)) / obar),
            "si_ub": float(np.sqrt(np.mean(((pj - pj.mean()) - (oj - obar)) ** 2)) / obar),
            "cc": float(np.corrcoef(oj, pj)[0, 1]) if oj.std() > 0 and pj.std() > 0 else np.nan,
        }
        if peak_threshold is not None:
            peak = oj >= peak_threshold
            ep = e[peak]
            row.update({
                "n_peak": int(peak.sum()),
                "rmse_peak": float(np.sqrt(np.mean(ep ** 2))) if peak.any() else np.nan,
                "bias_peak": float(np.mean(ep)) if peak.any() else np.nan,
                "under_rate_peak": float(np.mean(ep < 0)) if peak.any() else np.nan,
            })
        rows.append(row)
    return pd.DataFrame(rows).set_index("horizon_h")
