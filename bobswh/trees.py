"""Direct Multi-Step (DMS) gradient-boosted tree forecasters.

One independent regressor f_h is trained for each lead time h:

    Hs_hat(t + h) = f_h(X_t),   h in {6, 12, 24, 48, 72} h,

with X_t containing information up to the origin t only. No forecast is fed
back as an input, so errors cannot accumulate across horizons the way they do
in the recursive strategy. Each f_h can also settle on its own effective
complexity (number of boosting rounds via early stopping, and optionally its
own hyperparameters).

Backends: LightGBM, CatBoost and XGBoost, behind one interface.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BACKENDS = ("lightgbm", "catboost", "xgboost")

# Sensible untuned starting points (overridden by Optuna in Phase 3 tuning).
DEFAULT_PARAMS: dict[str, dict] = {
    "lightgbm": {"learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 50, "subsample": 0.8,
                 "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    "catboost": {"learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 3.0, "bootstrap_type": "Bernoulli",
                 "subsample": 0.8, "random_strength": 1.0},
    "xgboost": {"learning_rate": 0.03, "max_depth": 7, "min_child_weight": 5.0, "subsample": 0.8,
                "colsample_bytree": 0.8, "reg_lambda": 1.0},
}


def make_estimator(backend: str, params: Mapping, max_rounds: int, early_stopping_rounds: int | None,
                   seed: int, n_jobs: int):
    """Build an untrained regressor for one horizon."""
    params = dict(params)
    if backend == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=max_rounds, random_state=seed, n_jobs=n_jobs, verbose=-1, **params)
    if backend == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=max_rounds, loss_function="RMSE", random_seed=seed, thread_count=n_jobs,
                                 verbose=False, allow_writing_files=False, **params)
    if backend == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=max_rounds, tree_method="hist", eval_metric="rmse", random_state=seed,
                            n_jobs=n_jobs, early_stopping_rounds=early_stopping_rounds, **params)
    raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")


def fit_estimator(est, backend: str, X, y, X_es=None, y_es=None, early_stopping_rounds: int | None = None):
    """Fit with early stopping on (X_es, y_es) when given. Returns (estimator, best iteration)."""
    use_es = X_es is not None and early_stopping_rounds
    if backend == "lightgbm":
        import inspect

        import lightgbm as lgb
        if use_es:
            # LightGBM >= 4.6 takes eval_X / eval_y; older versions only eval_set
            if "eval_X" in inspect.signature(est.fit).parameters:
                es_kw = {"eval_X": (X_es,), "eval_y": (y_es,)}
            else:
                es_kw = {"eval_set": [(X_es, y_es)]}
            est.fit(X, y, eval_metric="rmse", callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
                    **es_kw)
            return est, int(est.best_iteration_)
        est.fit(X, y)
        return est, int(est.n_estimators)
    if backend == "catboost":
        if use_es:
            est.fit(X, y, eval_set=(X_es, y_es), early_stopping_rounds=early_stopping_rounds, use_best_model=True)
            return est, int(est.get_best_iteration()) + 1
        est.fit(X, y)
        return est, int(est.tree_count_)
    if backend == "xgboost":
        if use_es:
            est.fit(X, y, eval_set=[(X_es, y_es)], verbose=False)
            return est, int(est.best_iteration) + 1
        est.set_params(early_stopping_rounds=None)
        est.fit(X, y, verbose=False)
        return est, int(est.n_estimators)
    raise ValueError(backend)


class DirectMultiStepRegressor:
    """One gradient-boosted model per forecast horizon.

    Parameters
    ----------
    backend:
        "lightgbm", "catboost" or "xgboost".
    horizons_h:
        Lead times, in the column order of Y.
    params:
        Hyperparameters shared by all horizons (a dict), or one dict per
        horizon (a sequence), e.g. from per-horizon tuning.
    max_rounds, early_stopping_rounds:
        Upper bound on boosting rounds, and patience on the early-stopping set.
    """

    def __init__(
        self,
        backend: str,
        horizons_h: Sequence[int],
        params: Mapping | Sequence[Mapping] | None = None,
        max_rounds: int = 5000,
        early_stopping_rounds: int | None = 100,
        seed: int = 42,
        n_jobs: int = -1,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.backend = backend
        self.horizons_h = tuple(horizons_h)
        if params is None:
            params = DEFAULT_PARAMS[backend]
        self.params = [dict(params)] * len(self.horizons_h) if isinstance(params, Mapping) else [dict(p) for p in params]
        if len(self.params) != len(self.horizons_h):
            raise ValueError("need one parameter dict per horizon")
        self.max_rounds = max_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.n_jobs = n_jobs
        self.models_: list = []
        self.best_iterations_: list[int] = []
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame | np.ndarray,
            X_es: pd.DataFrame | None = None, Y_es: pd.DataFrame | np.ndarray | None = None) -> "DirectMultiStepRegressor":
        """Train f_h for every horizon. (X_es, Y_es) is the early-stopping set; it must not be
        the set used to select hyperparameters or to report results."""
        Y = np.asarray(Y, float)
        Y_es = None if Y_es is None else np.asarray(Y_es, float)
        self.feature_names_ = list(X.columns)
        self.models_, self.best_iterations_ = [], []
        for j, h in enumerate(self.horizons_h):
            est = make_estimator(self.backend, self.params[j], self.max_rounds, self.early_stopping_rounds,
                                 self.seed + j, self.n_jobs)
            est, best = fit_estimator(est, self.backend, X, Y[:, j], X_es, None if Y_es is None else Y_es[:, j],
                                      self.early_stopping_rounds)
            self.models_.append(est)
            self.best_iterations_.append(best)
            log.info("%s t+%dh: %d rounds", self.backend, h, best)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Forecasts of shape (n, n_horizons), clipped at 0 m."""
        if not self.models_:
            raise RuntimeError("call fit first")
        X = X[self.feature_names_]
        return np.maximum(np.column_stack([m.predict(X) for m in self.models_]), 0.0)

    def feature_importance(self) -> pd.DataFrame:
        """Gain-based importance, normalised to sum to 1 per horizon (features x horizons)."""
        cols = {}
        for h, m in zip(self.horizons_h, self.models_):
            if self.backend == "lightgbm":
                imp = m.booster_.feature_importance(importance_type="gain")
            elif self.backend == "catboost":
                imp = m.get_feature_importance()
            else:
                score = m.get_booster().get_score(importance_type="gain")
                imp = np.array([score.get(f, 0.0) for f in self.feature_names_])
            imp = np.asarray(imp, float)
            cols[f"t+{h}h"] = imp / imp.sum() if imp.sum() > 0 else imp
        return pd.DataFrame(cols, index=self.feature_names_)

    def save(self, path: Path | str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str) -> "DirectMultiStepRegressor":
        return joblib.load(path)
