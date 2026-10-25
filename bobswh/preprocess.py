"""Quality control, regularisation and short-gap imputation of time series.

Order of operations (``Preprocessor.run``):

1. Range check: values outside physically admissible bounds become NaN.
2. Regularisation onto a fixed time base. Bins are right-closed and labelled
   on the right, (t - dt, t] -> t, so a value stamped t never contains
   information from after t. Directions are averaged as unit vectors.
3. Isolated-spike removal (Hs only by default). Sustained peaks survive.
4. Imputation of gaps no longer than ``max_gap_h``. Longer gaps stay NaN,
   and the window generator later drops every sample that touches them.

On look-ahead: forward fill is causal. Linear, PCHIP and cubic interpolation
use the first valid value *after* the gap, so a model input at time t inside
an interpolated gap carries up to ``max_gap_h`` hours of future information.
The default is therefore ``ffill``; the interpolating methods are for
diagnostics and plots. Imputed values are flagged in ``PreprocessResult.imputed``
so that imputed targets can be excluded from training and scoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

from . import config

FILL_METHODS = ("ffill", "linear", "pchip", "cubic")
_CONTEXT_POINTS = 3   # valid points used on each side of a gap by pchip / cubic


@dataclass
class PreprocessResult:
    data: pd.DataFrame      # regular, QC'd, short gaps filled
    imputed: pd.DataFrame   # True where a value was filled in
    report: pd.DataFrame    # per-column QC counts


def drop_sparse_columns(df: pd.DataFrame, min_coverage: float = 0.9) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns observed in fewer than ``min_coverage`` of rows. Returns (data, dropped names).

    A variable present in only part of the files (e.g. one extra-variable
    month mixed with a longer product that lacks it) would otherwise make the
    window generator discard every sample outside that part.
    """
    coverage = df.notna().mean()
    dropped = sorted(coverage.index[coverage < min_coverage])
    return df.drop(columns=dropped), dropped


def range_qc(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Set values outside ``config.qc_bounds`` to NaN. Returns (data, removed count per column)."""
    out = df.copy()
    removed = {}
    for col in out.columns:
        bounds = config.qc_bounds(col)
        if bounds is None:
            removed[col] = 0
            continue
        bad = (out[col] < bounds[0]) | (out[col] > bounds[1])
        removed[col] = int(bad.sum())
        out.loc[bad, col] = np.nan
    return out, pd.Series(removed, dtype=int)


def regularise(df: pd.DataFrame, freq: str = config.BASE_FREQ, instantaneous: bool = False) -> pd.DataFrame:
    """Put the series on a regular time base of step ``freq``.

    Parameters
    ----------
    instantaneous:
        True for fields valid at an instant (ERA5 analyses): values are taken
        at the exact time steps and nothing is averaged. False for sensor
        records (buoys): all samples in (t - dt, t] are averaged into t.
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    full = pd.date_range(df.index[0].ceil(freq), df.index[-1].floor(freq), freq=freq, name=df.index.name)
    if instantaneous:
        return df.reindex(full)
    circular = [c for c in df.columns if config.is_circular(c)]
    work = df.drop(columns=circular)
    for c in circular:
        rad = np.deg2rad(df[c])
        work[f"__sin_{c}"] = np.sin(rad)
        work[f"__cos_{c}"] = np.cos(rad)
    agg = work.resample(freq, closed="right", label="right").mean()
    for c in circular:
        agg[c] = np.rad2deg(np.arctan2(agg.pop(f"__sin_{c}"), agg.pop(f"__cos_{c}"))) % 360.0
    return agg.reindex(full)[list(df.columns)]


def isolated_spike_mask(s: pd.Series, jump: float, neighbour_tol: float) -> pd.Series:
    """True where a sample departs from both neighbours by more than ``jump`` in the same sense
    while the neighbours agree with each other within ``neighbour_tol``."""
    prev, nxt = s.shift(1), s.shift(-1)
    up = (s - prev > jump) & (s - nxt > jump)
    down = (prev - s > jump) & (nxt - s > jump)
    return (up | down) & ((prev - nxt).abs() <= neighbour_tol)


def short_gaps(isna: np.ndarray, max_len: int, needs_right_anchor: bool) -> list[tuple[int, int]]:
    """(start, end_exclusive) of NaN runs of length <= max_len that have a valid value on the left
    (and on the right, if ``needs_right_anchor``)."""
    padded = np.r_[0, np.asarray(isna, np.int8), 0]
    d = np.diff(padded)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (ends - starts <= max_len) & (starts > 0)
    if needs_right_anchor:
        keep &= ends < len(isna)
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))


def _fill_array(y: np.ndarray, x: np.ndarray, gaps: list[tuple[int, int]], method: str) -> np.ndarray:
    out = y.copy()
    ok = np.isfinite(y)
    for st, en in gaps:
        if method == "ffill":
            out[st:en] = y[st - 1]
            continue
        left = [i for i in range(st - 1, max(st - 1 - _CONTEXT_POINTS, -1), -1) if ok[i]]
        right = [i for i in range(en, min(en + _CONTEXT_POINTS, len(y))) if ok[i]]
        if method == "linear":
            left, right = left[:1], right[:1]
        ctx = np.array(sorted(left + right))
        xs = x[st:en]
        if method == "linear" or len(ctx) < 3:
            out[st:en] = np.interp(xs, x[ctx], y[ctx])
        elif method == "pchip":
            out[st:en] = PchipInterpolator(x[ctx], y[ctx])(xs)
        else:
            out[st:en] = CubicSpline(x[ctx], y[ctx])(xs)
    return out


def fill_short_gaps(df: pd.DataFrame, max_gap_steps: int, method: str = config.FILL_METHOD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute NaN runs of at most ``max_gap_steps`` samples; longer runs stay NaN.

    Directions are filled through their sine and cosine components so that a
    gap between 350 and 10 degrees is bridged through north, not through 180.
    Non-negative variables are clipped at zero (cubic splines can overshoot).

    Returns (filled data, boolean imputed mask).
    """
    if method not in FILL_METHODS:
        raise ValueError(f"method must be one of {FILL_METHODS}, got {method!r}")
    filled = df.copy()
    imputed = pd.DataFrame(False, index=df.index, columns=df.columns)
    if max_gap_steps < 1 or df.empty:
        return filled, imputed
    x = ((df.index - df.index[0]) / pd.Timedelta(hours=1)).to_numpy(float)
    for col in df.columns:
        y = df[col].to_numpy(float)
        gaps = short_gaps(np.isnan(y), max_gap_steps, needs_right_anchor=(method != "ffill"))
        if not gaps:
            continue
        if config.is_circular(col):
            rad = np.deg2rad(y)
            s = _fill_array(np.sin(rad), x, gaps, method)
            c = _fill_array(np.cos(rad), x, gaps, method)
            new = np.rad2deg(np.arctan2(s, c)) % 360.0
            new[~np.isnan(y)] = y[~np.isnan(y)]
        else:
            new = _fill_array(y, x, gaps, method)
            if config.is_non_negative(col):
                new = np.where(np.isnan(new), np.nan, np.maximum(new, 0.0))
        mask = np.zeros(len(y), bool)
        for st, en in gaps:
            mask[st:en] = True
        filled[col] = new
        imputed[col] = mask
    return filled, imputed


@dataclass
class Preprocessor:
    """Range QC -> regularise -> despike -> fill short gaps.

    Attributes
    ----------
    freq:
        Target time step, e.g. "1h".
    max_gap_h:
        Longest gap (hours) that is imputed.
    fill_method:
        One of ``FILL_METHODS``; see the module docstring on look-ahead.
    instantaneous:
        True for ERA5 (sample at the step), False for buoys (average the bin).
    despike:
        Apply ``config.SPIKE_RULES``.
    """

    freq: str = config.BASE_FREQ
    max_gap_h: float = config.MAX_GAP_H
    fill_method: str = config.FILL_METHOD
    instantaneous: bool = False
    despike: bool = True

    def run(self, df: pd.DataFrame) -> PreprocessResult:
        n_raw = df.notna().sum()
        qc, n_range = range_qc(df)
        reg = regularise(qc, self.freq, self.instantaneous)
        n_spikes = pd.Series(0, index=reg.columns, dtype=int)
        if self.despike:
            for col in reg.columns:
                rule = config.spike_rule(col)
                if rule is None:
                    continue
                mask = isolated_spike_mask(reg[col], *rule)
                reg.loc[mask, col] = np.nan
                n_spikes[col] = int(mask.sum())
        step_h = pd.Timedelta(self.freq) / pd.Timedelta(hours=1)
        filled, imputed = fill_short_gaps(reg, int(self.max_gap_h // step_h), self.fill_method)
        report = pd.DataFrame({
            "n_raw": n_raw,
            "n_out_of_range": n_range,
            "n_spikes": n_spikes,
            "n_imputed": imputed.sum(),
            "n_missing_after": filled.isna().sum(),
            "coverage_after": filled.notna().mean().round(4),
        }).fillna(0)
        report.index.name = "column"
        return PreprocessResult(filled, imputed, report)
