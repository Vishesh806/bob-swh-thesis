"""Bayesian hyperparameter optimisation (Optuna, TPE) of the DMS tree models.

Three disjoint, chronologically ordered sample blocks are used:

    fit  -> trains the boosters
    es   -> early stopping (chooses the number of rounds), the last part of the training period
    val  -> the Optuna objective (chooses the hyperparameters)

The test block is never touched. Keeping early stopping and hyperparameter
selection on different blocks stops the validation score from being
optimised twice, which would make it an optimistic estimate.

Objective (shared mode): the pooled multi-horizon RMSE on val,
sqrt(mean_h RMSE_h^2), over one set of hyperparameters for all horizons.
Horizons are fitted in order and the running objective is reported after each
one, so the median pruner can stop a poor trial after its first horizons.
Per-horizon mode instead runs one study per horizon on RMSE_h.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import rmse
from .trees import BACKENDS, fit_estimator, make_estimator

log = logging.getLogger(__name__)


def suggest_params(trial, backend: str) -> dict:
    """Search space per backend (log scales for rates and regularisation)."""
    if backend == "lightgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if backend == "catboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 200, log=True),
        }
    if backend == "xgboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")


@dataclass
class TuningResult:
    params: dict | list[dict]          # shared dict, or one dict per horizon
    best_value: float | list[float]
    trials: pd.DataFrame


@dataclass
class MultiHorizonTuner:
    """Optuna search for DMS hyperparameters.

    Attributes
    ----------
    per_horizon:
        False: one parameter set for all horizons, objective = pooled RMSE.
        True: an independent study per horizon, objective = RMSE_h.
    n_trials, timeout_s:
        Budget per study (whichever is reached first).
    storage:
        e.g. "sqlite:///optuna.db" to resume an interrupted search (Colab).
    """

    backend: str
    horizons_h: Sequence[int]
    n_trials: int = 40
    timeout_s: float | None = None
    per_horizon: bool = False
    max_rounds: int = 5000
    early_stopping_rounds: int = 100
    seed: int = 42
    n_jobs: int = -1
    storage: str | None = None
    study_name: str | None = None
    studies_: list = field(default_factory=list, init=False, repr=False)

    def _fit_score(self, params, j, X_fit, Y_fit, X_es, Y_es, X_val, Y_val) -> float:
        est = make_estimator(self.backend, params, self.max_rounds, self.early_stopping_rounds, self.seed + j, self.n_jobs)
        est, _ = fit_estimator(est, self.backend, X_fit, Y_fit[:, j], X_es, Y_es[:, j], self.early_stopping_rounds)
        return float(rmse(Y_val[:, j], np.maximum(est.predict(X_val), 0.0))[0])

    def _study(self, name: str):
        import optuna
        return optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
            storage=self.storage,
            study_name=name,
            load_if_exists=self.storage is not None,
        )

    def tune(self, X_fit: pd.DataFrame, Y_fit, X_es: pd.DataFrame, Y_es, X_val: pd.DataFrame, Y_val) -> TuningResult:
        import optuna

        Y_fit, Y_es, Y_val = (np.asarray(a, float) for a in (Y_fit, Y_es, Y_val))
        data = (X_fit, Y_fit, X_es, Y_es, X_val, Y_val)
        base = self.study_name or f"{self.backend}_dms"
        self.studies_ = []

        if not self.per_horizon:
            def objective(trial) -> float:
                params = suggest_params(trial, self.backend)
                sq = []
                for j in range(len(self.horizons_h)):
                    sq.append(self._fit_score(params, j, *data) ** 2)
                    running = float(np.sqrt(np.mean(sq)))
                    trial.report(running, step=j)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    trial.set_user_attr(f"rmse_t+{self.horizons_h[j]}h", float(np.sqrt(sq[-1])))
                return running

            study = self._study(base)
            study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout_s, gc_after_trial=True)
            self.studies_.append(study)
            return TuningResult(self._complete(study, study.best_params), study.best_value, study.trials_dataframe())

        params, values, tables = [], [], []
        for j, h in enumerate(self.horizons_h):
            def objective(trial, j=j) -> float:
                return self._fit_score(suggest_params(trial, self.backend), j, *data)

            study = self._study(f"{base}_t+{h}h")
            study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout_s, gc_after_trial=True)
            self.studies_.append(study)
            params.append(self._complete(study, study.best_params))
            values.append(study.best_value)
            tables.append(study.trials_dataframe().assign(horizon_h=h))
        return TuningResult(params, values, pd.concat(tables, ignore_index=True))

    def _complete(self, study, best: dict) -> dict:
        """Add back the fixed (non-searched) entries of the search space."""
        import optuna
        fixed = suggest_params(optuna.trial.FixedTrial(best), self.backend)
        return {**fixed, **best}
