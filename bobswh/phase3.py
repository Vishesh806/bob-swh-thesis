"""Phase 3 orchestration: baselines and Direct Multi-Step tree models for one buoy.

Inputs:  ``{features_dir}/{buoy}_features.parquet`` and ``{buoy}_imputed.parquet`` (Phase 2).
Outputs in ``{out_dir}/{buoy}/``:

* ``split_summary.csv``         sample counts and time extent of fit / es / val / test
* ``metrics.csv``               RMSE, MAE, bias, R^2, SI, CC per model, split and horizon
* ``rmse_table.csv``            RMSE pivot (model x horizon) on the test block
* ``predictions.parquet``       observed and forecast Hs for val and test origins
* ``{backend}_params.json``     tuned (or default) hyperparameters and boosting rounds
* ``{backend}_importance.csv``  gain importance per horizon
* ``{backend}_optuna.csv``      all Optuna trials
* ``sarimax_orders.csv``        AIC table, when order selection is on
* ``models/{buoy}_{backend}.joblib``

Every model is scored on exactly the same forecast origins.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config
from .baselines import ClimatologyForecaster, PersistenceForecaster
from .features import FeatureEngineer
from .metrics import horizon_metrics
from .sarimax import SARIMAXForecaster, select_order
from .trees import DEFAULT_PARAMS, DirectMultiStepRegressor
from .tuning import MultiHorizonTuner
from .windowing import ChronologicalSplitter, SlidingWindowGenerator, WindowSpec, carve_early_stopping_block

log = logging.getLogger(__name__)

SCORED_SPLITS = ("val", "test")


def select_inputs(df: pd.DataFrame, target_col: str, include_buoy_exog: bool = False) -> list[str]:
    """Model inputs: every ERA5 and calendar feature, the target's own derived features
    (e.g. ``buoy_hs_tend3h``), and other buoy variables only if ``include_buoy_exog``.

    Buoy wind, pressure and SST have their own gaps; including them by default
    would drop every sample where any of them is missing.
    """
    cols = []
    for c in FeatureEngineer.feature_columns(df):
        if c == target_col:
            continue                                   # prepended by WindowSpec
        if c.startswith("buoy_") and not (include_buoy_exog or c.startswith(f"{target_col}_")):
            continue
        cols.append(c)
    return cols


@dataclass
class Phase3Runner:
    """Train and score persistence, climatology, SARIMAX and DMS tree models for one buoy.

    Attributes
    ----------
    lag_cols:
        Columns given the full lag set in the tabular layout (besides the target).
    n_trials:
        Optuna trials per backend; 0 trains with ``DEFAULT_PARAMS``.
    tune_stride:
        Use every n-th fit sample during tuning only (hourly origins are
        strongly autocorrelated, so thinning costs little information and
        cuts tuning time by the same factor). The final model uses all samples.
    sarimax_orders:
        Candidate (p, d, q) orders selected by AIC; None uses ``SARIMAXForecaster`` defaults.
    """

    buoy: str
    target_col: str = "buoy_hs"
    features_dir: Path = config.PROC_DIR
    out_dir: Path = config.RESULTS_DIR / "phase3"
    model_dir: Path = config.MODEL_DIR
    splits: Mapping[str, tuple[str, str]] = field(default_factory=lambda: dict(config.SPLITS))
    es_days: int = config.EARLY_STOPPING_DAYS
    horizons_h: tuple[int, ...] = config.HORIZONS_H
    lookback_h: int = config.LOOKBACK_H
    lag_cols: tuple[str, ...] = ("era5_swh", "era5_tau", "era5_wspd10", "era5_msl")
    include_buoy_exog: bool = False
    backends: tuple[str, ...] = ("lightgbm", "catboost")
    n_trials: int = 40
    timeout_s: float | None = None
    tune_stride: int = 3
    per_horizon: bool = False
    max_rounds: int = 5000
    early_stopping_rounds: int = 100
    sarimax_orders: Sequence[tuple[int, int, int]] | None = None
    sarimax_fit_days: int | None = 730
    run_sarimax: bool = True
    seed: int = 42
    n_jobs: int = -1

    # -- data ------------------------------------------------------------
    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_parquet(Path(self.features_dir) / f"{self.buoy}_features.parquet")
        imputed = pd.read_parquet(Path(self.features_dir) / f"{self.buoy}_imputed.parquet")
        return df, imputed

    def build_samples(self, df: pd.DataFrame, imputed: pd.DataFrame):
        """Tabular (X, Y) for all origins and the positions of each block."""
        inputs = select_inputs(df, self.target_col, self.include_buoy_exog)
        lag_cols = tuple(dict.fromkeys((self.target_col,) + tuple(c for c in self.lag_cols if c in df)))
        spec = WindowSpec(target_col=self.target_col, input_cols=tuple(inputs), lookback_h=self.lookback_h,
                          horizons_h=self.horizons_h, tabular_lag_cols=lag_cols)
        X, Y = SlidingWindowGenerator(spec, imputed=imputed).tabular(df)
        splitter = ChronologicalSplitter(carve_early_stopping_block(self.splits, self.es_days),
                                         self.lookback_h, max(self.horizons_h))
        idx = splitter.split(X.index)
        for name in ("fit", "es", *SCORED_SPLITS):
            if len(idx[name]) == 0:
                raise ValueError(f"block '{name}' has no complete samples; check the data coverage and SPLITS")
        return X, Y, idx, splitter.summary(X.index, idx)

    # -- models ----------------------------------------------------------
    def _train_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.splits["train"][1])

    def _baselines(self, df, imputed, X, blocks) -> dict[str, dict[str, np.ndarray]]:
        preds: dict[str, dict[str, np.ndarray]] = {}
        persist = PersistenceForecaster(self.horizons_h)
        preds["persistence"] = {s: persist.predict(X.iloc[blocks[s]][f"{self.target_col}_lag0h"]) for s in SCORED_SPLITS}

        observed = df[self.target_col].where(~imputed[self.target_col].reindex(df.index, fill_value=False))
        clim = ClimatologyForecaster(self.horizons_h).fit(observed[observed.index <= self._train_end()])
        preds["climatology"] = {s: clim.predict(X.index[blocks[s]]) for s in SCORED_SPLITS}

        if self.run_sarimax:
            series = df[self.target_col]
            train = series[series.index <= self._train_end()]
            if self.sarimax_orders:
                sar, table = select_order(train, self.sarimax_orders, fit_window_days=self.sarimax_fit_days)
                table.to_csv(self._out / "sarimax_orders.csv", index=False)
            else:
                sar = SARIMAXForecaster(fit_window_days=self.sarimax_fit_days).fit(train)
            last = max(X.index[blocks[s]].max() for s in SCORED_SPLITS)
            series = series[series.index <= last]
            preds["sarimax"] = {s: sar.forecast_origins(series, X.index[blocks[s]], self.horizons_h) for s in SCORED_SPLITS}
            log.info("SARIMAX order %s", sar.order)
        return preds

    def _trees(self, X, Y, blocks) -> dict[str, dict[str, np.ndarray]]:
        preds = {}
        Xb = {k: X.iloc[v] for k, v in blocks.items()}
        Yb = {k: Y.iloc[v].to_numpy() for k, v in blocks.items()}
        self.model_dir.mkdir(parents=True, exist_ok=True)
        for backend in self.backends:
            if self.n_trials > 0:
                tuner = MultiHorizonTuner(backend, self.horizons_h, n_trials=self.n_trials, timeout_s=self.timeout_s,
                                          per_horizon=self.per_horizon, max_rounds=self.max_rounds,
                                          early_stopping_rounds=self.early_stopping_rounds, seed=self.seed,
                                          n_jobs=self.n_jobs, study_name=f"{self.buoy}_{backend}")
                s = slice(None, None, max(self.tune_stride, 1))
                res = tuner.tune(Xb["fit"].iloc[s], Yb["fit"][s], Xb["es"], Yb["es"], Xb["val"], Yb["val"])
                res.trials.to_csv(self._out / f"{backend}_optuna.csv", index=False)
                params = res.params
            else:
                params = DEFAULT_PARAMS[backend]
            model = DirectMultiStepRegressor(backend, self.horizons_h, params, self.max_rounds,
                                             self.early_stopping_rounds, self.seed, self.n_jobs)
            model.fit(Xb["fit"], Yb["fit"], Xb["es"], Yb["es"])
            model.save(self.model_dir / f"{self.buoy}_{backend}.joblib")
            model.feature_importance().to_csv(self._out / f"{backend}_importance.csv")
            with open(self._out / f"{backend}_params.json", "w") as fh:
                json.dump({"params": params, "best_iterations": model.best_iterations_,
                           "n_features": len(model.feature_names_)}, fh, indent=2)
            preds[backend] = {s: model.predict(Xb[s]) for s in SCORED_SPLITS}
        return preds

    # -- main ------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Run every model; return the long metrics table (model, split, horizon)."""
        self._out = Path(self.out_dir) / self.buoy
        self._out.mkdir(parents=True, exist_ok=True)
        df, imputed = self.load()
        X, Y, blocks, summary = self.build_samples(df, imputed)
        summary.to_csv(self._out / "split_summary.csv")
        log.info("samples per block: %s", {k: len(v) for k, v in blocks.items()})

        preds = {**self._baselines(df, imputed, X, blocks), **self._trees(X, Y, blocks)}

        tables, frames = [], []
        for split in SCORED_SPLITS:
            obs = Y.iloc[blocks[split]].to_numpy()
            frame = pd.DataFrame(obs, index=X.index[blocks[split]],
                                 columns=[f"obs__t+{h}h" for h in self.horizons_h])
            for name, p in preds.items():
                tables.append(horizon_metrics(obs, p[split], self.horizons_h).assign(model=name, split=split))
                for j, h in enumerate(self.horizons_h):
                    frame[f"{name}__t+{h}h"] = p[split][:, j]
            frames.append(frame.assign(split=split))
        metrics = pd.concat(tables).reset_index().set_index(["model", "split", "horizon_h"]).sort_index()
        metrics.to_csv(self._out / "metrics.csv")
        pd.concat(frames).to_parquet(self._out / "predictions.parquet")
        rmse_table = metrics.xs("test", level="split")["rmse"].unstack("horizon_h")
        rmse_table.to_csv(self._out / "rmse_table.csv")
        with open(self._out / "run_config.json", "w") as fh:
            json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}, fh, indent=2, default=str)
        return metrics
