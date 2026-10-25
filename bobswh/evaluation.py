"""Phase 5: common-sample verification of every model, skill scores, significance tests and storm windows.

All models are scored on exactly the same forecast origins: the predictions
of Phase 3 (baselines, trees) and Phase 4 (LSTM variants) are joined on the
origin time, and only origins present for every model are kept.

Skill relative to persistence:   SS = 1 - RMSE_model / RMSE_persistence.

Diebold-Mariano test (Diebold and Mariano 1995) of equal squared-error loss
between two models, with the Harvey-Leybourne-Newbold (1997) small-sample
correction. Errors of h-step forecasts issued every hour overlap, so the loss
differential is autocorrelated up to lag ~h; the long-run variance uses a
Newey-West (Bartlett) estimator with that many lags.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .metrics import horizon_metrics

MODEL_ORDER = ("persistence", "climatology", "sarimax", "lightgbm", "catboost", "lstm_mse", "lstm_peak")
MODEL_LABELS = {
    "persistence": "Persistence", "climatology": "Climatology", "sarimax": "ARIMA",
    "lightgbm": "LightGBM", "catboost": "CatBoost", "lstm_mse": "Seq2Seq LSTM (MSE)",
    "lstm_peak": "Seq2Seq LSTM (peak loss)",
}

# Tropical cyclones of the Bay of Bengal in the test period (2023-2024). Dates are
# approximate system lifetimes from the IMD best tracks, widened by a day; they
# only select windows for storm-period scoring and plotting.
TEST_CYCLONES: dict[str, tuple[str, str]] = {
    "Mocha": ("2023-05-08", "2023-05-15"),
    "Hamoon": ("2023-10-20", "2023-10-26"),
    "Midhili": ("2023-11-15", "2023-11-18"),
    "Michaung": ("2023-11-30", "2023-12-06"),
    "Remal": ("2024-05-23", "2024-05-28"),
    "Dana": ("2024-10-21", "2024-10-26"),
    "Fengal": ("2024-11-24", "2024-12-01"),
}


def load_predictions(buoy: str, results_dir: Path = config.RESULTS_DIR, split: str = "test") -> pd.DataFrame:
    """Wide frame indexed by origin: ``obs__t+{h}h`` and ``{model}__t+{h}h`` for every available model,
    restricted to origins common to all of them."""
    frames = []
    for phase in ("phase3", "phase4"):
        path = Path(results_dir) / phase / buoy / "predictions.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            frames.append(df[df["split"] == split].drop(columns="split"))
    if not frames:
        raise FileNotFoundError(f"no predictions for {buoy} under {results_dir}")
    out = frames[0]
    for other in frames[1:]:
        obs_cols = [c for c in other.columns if c.startswith("obs__")]
        common = out.index.intersection(other.index)
        diff = (out.loc[common, obs_cols] - other.loc[common, obs_cols]).abs().max().max()
        if diff > 1e-4:
            raise ValueError(f"{buoy}: observations differ between phases by {diff}")
        out = out.join(other.drop(columns=obs_cols), how="inner")
    return out.dropna()


def models_in(pred: pd.DataFrame) -> list[str]:
    names = {c.split("__")[0] for c in pred.columns if not c.startswith("obs__")}
    return [m for m in MODEL_ORDER if m in names] + sorted(names - set(MODEL_ORDER))


def arrays(pred: pd.DataFrame, model: str, horizons: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    obs = pred[[f"obs__t+{h}h" for h in horizons]].to_numpy()
    p = pred[[f"{model}__t+{h}h" for h in horizons]].to_numpy()
    return obs, p


def peak_threshold(buoy: str, target: str = "era5_swh", results_dir: Path = config.RESULTS_DIR) -> float:
    """q95 of training targets, as used by the peak-weighted loss (from the Phase 4 run), else from the features."""
    cfg = Path(results_dir) / "phase4" / buoy / "run_config.json"
    if cfg.exists():
        thr = json.loads(cfg.read_text()).get("peak_threshold_m")
        if thr is not None:
            return float(thr)
    df = pd.read_parquet(config.PROC_DIR / f"{buoy}_features.parquet", columns=[target])
    start, end = config.SPLITS["train"]
    return float(df.loc[start:end, target].quantile(0.95))


def evaluate(pred: pd.DataFrame, horizons: Sequence[int] = config.HORIZONS_H, threshold: float | None = None) -> pd.DataFrame:
    """Metrics per (model, horizon) plus skill vs persistence."""
    tables = []
    for m in models_in(pred):
        obs, p = arrays(pred, m, horizons)
        tables.append(horizon_metrics(obs, p, horizons, threshold).assign(model=m))
    out = pd.concat(tables).reset_index().set_index(["model", "horizon_h"])
    if "persistence" in out.index.get_level_values("model"):
        ref = out.xs("persistence", level="model")["rmse"]
        out["skill_vs_persistence"] = 1.0 - out["rmse"] / ref.reindex(out.index.get_level_values("horizon_h")).to_numpy()
    return out


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h_steps: int) -> tuple[float, float]:
    """DM statistic and two-sided p-value for equal mean squared error (HLN-corrected, Newey-West lags h-1..h)."""
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    n = d.size
    dc = d - d.mean()
    lags = max(int(h_steps), 1)
    gamma0 = np.dot(dc, dc) / n
    lrv = gamma0 + 2 * sum((1 - k / (lags + 1)) * np.dot(dc[k:], dc[:-k]) / n for k in range(1, lags + 1))
    if lrv <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h_steps + h_steps * (h_steps - 1) / n) / n)
    stat = dm * hln
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def dm_table(pred: pd.DataFrame, horizons: Sequence[int] = config.HORIZONS_H, reference: str | None = None) -> pd.DataFrame:
    """DM tests of every model against ``reference`` (default: lowest-RMSE model at each horizon)."""
    rows = []
    models = models_in(pred)
    for j, h in enumerate(horizons):
        obs = pred[f"obs__t+{h}h"].to_numpy()
        errs = {m: pred[f"{m}__t+{h}h"].to_numpy() - obs for m in models}
        ref = reference or min(models, key=lambda m: np.mean(errs[m] ** 2))
        for m in models:
            if m == ref:
                continue
            stat, p = diebold_mariano(errs[m], errs[ref], h)
            rows.append({"horizon_h": h, "reference": ref, "model": m, "dm_stat": stat, "p_value": p})
    return pd.DataFrame(rows)


def storm_mask(index: pd.DatetimeIndex, h: int, events: dict[str, tuple[str, str]] = TEST_CYCLONES) -> np.ndarray:
    """True where the forecast valid time (origin + h) falls inside a cyclone window."""
    valid = pd.DatetimeIndex(index) + pd.Timedelta(hours=h)
    mask = np.zeros(len(valid), bool)
    for start, end in events.values():
        mask |= (valid >= pd.Timestamp(start)) & (valid <= pd.Timestamp(end) + pd.Timedelta(hours=23))
    return mask


def storm_metrics(pred: pd.DataFrame, horizons: Sequence[int] = config.HORIZONS_H) -> pd.DataFrame:
    """RMSE and bias per model and horizon, restricted to cyclone windows."""
    rows = []
    for m in models_in(pred):
        for h in horizons:
            mask = storm_mask(pred.index, h)
            e = pred.loc[mask, f"{m}__t+{h}h"].to_numpy() - pred.loc[mask, f"obs__t+{h}h"].to_numpy()
            rows.append({"model": m, "horizon_h": h, "n": int(mask.sum()),
                         "rmse_storm": float(np.sqrt(np.mean(e ** 2))) if mask.any() else np.nan,
                         "bias_storm": float(np.mean(e)) if mask.any() else np.nan})
    return pd.DataFrame(rows).set_index(["model", "horizon_h"])


def largest_event(pred: pd.DataFrame, h: int = 24) -> tuple[str, pd.Timestamp]:
    """Named cyclone with the highest observed Hs at this buoy, and the time of that peak."""
    obs = pd.Series(pred[f"obs__t+{h}h"].to_numpy(), index=pred.index + pd.Timedelta(hours=h))
    best = None
    for name, (start, end) in TEST_CYCLONES.items():
        window = obs[start:pd.Timestamp(end) + pd.Timedelta(hours=23)]
        if len(window) and (best is None or window.max() > best[2]):
            best = (name, window.idxmax(), window.max())
    if best is None:
        return "largest event", obs.idxmax()
    return best[0], best[1]
