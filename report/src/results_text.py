"""Numbers and sentences derived from the Phase 5 outputs, shared by the report and the slides.

Nothing here is typed by hand: every figure quoted in the text is computed from results/phase5.
"""
import json

import numpy as np
import pandas as pd

from common import RES

P5 = RES / "phase5"
H = (6, 12, 24, 48, 72)
LABEL = {"persistence": "Persistence", "climatology": "Climatology", "sarimax": "ARIMA", "lightgbm": "LightGBM",
         "catboost": "CatBoost", "lstm_mse": "LSTM (MSE)", "lstm_peak": "LSTM (peak loss)"}
ORDER = ["persistence", "climatology", "sarimax", "lightgbm", "catboost", "lstm_mse", "lstm_peak"]
ML = ["lightgbm", "catboost", "lstm_mse", "lstm_peak"]

rmse = pd.read_csv(P5 / "rmse_test.csv", index_col=[0, 1])
rmse.columns = [int(c) for c in rmse.columns]
metrics = pd.read_csv(P5 / "metrics_test.csv", index_col=[0, 1, 2])
pooled = pd.read_csv(P5 / "metrics_test_pooled.csv", index_col=[0, 1])
storm = pd.read_csv(P5 / "storm_metrics_test.csv", index_col=[0, 1, 2])
dm = pd.read_csv(P5 / "diebold_mariano_test.csv")
summary = json.loads((P5 / "summary.json").read_text())

BUOYS = list(dict.fromkeys(rmse.index.get_level_values(0)))
MODELS = [m for m in ORDER if m in set(rmse.index.get_level_values(1))]
mean_rmse = rmse.groupby(level=1).mean().reindex(MODELS)            # model x horizon, mean over buoys


def best(h, among=None):
    cands = [m for m in (among or MODELS) if m != "climatology"]
    return mean_rmse.loc[cands, h].idxmin()


def pct_gain(model, ref, h):
    return 100.0 * (1.0 - mean_rmse.loc[model, h] / mean_rmse.loc[ref, h])


BEST = {h: best(h) for h in H}
BEST_ML = {h: best(h, [m for m in ML if m in MODELS]) for h in H}
best_tree = {h: min([m for m in ("lightgbm", "catboost") if m in MODELS], key=lambda m: mean_rmse.loc[m, h]) for h in H}
has_lstm = all(m in MODELS for m in ("lstm_mse", "lstm_peak"))


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


# ---------------------------------------------------------------- headline numbers
gain_pers = {h: pct_gain(BEST_ML[h], "persistence", h) for h in H}
gain_arima = {h: pct_gain(BEST_ML[h], "sarimax", h) for h in H} if "sarimax" in MODELS else {}
r2 = metrics["r2"].groupby(level=[1, 2]).mean()
si = metrics["si"].groupby(level=[1, 2]).mean()
cc = metrics["cc"].groupby(level=[1, 2]).mean()

if has_lstm:
    pk = pooled.loc["lstm_peak"]
    ms = pooled.loc["lstm_mse"]
    under = {h: (ms.loc[h, "under_rate_peak"], pk.loc[h, "under_rate_peak"]) for h in H}
    biasp = {h: (ms.loc[h, "bias_peak"], pk.loc[h, "bias_peak"]) for h in H}
    rmsep = {h: (ms.loc[h, "rmse_peak"], pk.loc[h, "rmse_peak"]) for h in H}
    cost = {h: 100.0 * (pk.loc[h, "rmse"] / ms.loc[h, "rmse"] - 1.0) for h in H}

storm_mean = storm["rmse_storm"].groupby(level=[1, 2]).mean()
storm_bias = storm["bias_storm"].groupby(level=[1, 2]).mean()
STORM_BEST = {h: storm_mean.xs(h, level=1).drop(["climatology"], errors="ignore").idxmin() for h in H}

dm_sig = dm.assign(sig=dm["p_value"] < 0.05)
dm_frac = float(dm_sig["sig"].mean())

zoom = {b: (v["zoom_event"], v["zoom_peak_time"][:10], v["zoom_peak_hs_m"]) for b, v in summary["buoys"].items()}
n_test = {b: v["n_test_origins"] for b, v in summary["buoys"].items()}
q95 = {b: v["peak_threshold_m"] for b, v in summary["buoys"].items()}


# ---------------------------------------------------------------- tables
def rmse_table_rows(buoy):
    rows = []
    for m in MODELS:
        vals = rmse.loc[(buoy, m)]
        cells = []
        for h in H:
            mark = "**" if m == best(h) and buoy is None else ""
            cells.append(f3(vals[h]))
        rows.append([LABEL[m]] + cells)
    return rows


def rmse_rows_with_best(buoy):
    rows = []
    col_best = {h: rmse.xs(buoy, level=0).drop("climatology", errors="ignore")[h].idxmin() for h in H}
    for m in MODELS:
        rows.append([LABEL[m]] + [("**" + f3(rmse.loc[(buoy, m), h]) + "**") if m == col_best[h] else f3(rmse.loc[(buoy, m), h])
                                  for h in H])
    return rows


def mean_rows(stat):
    tab = metrics[stat].groupby(level=[1, 2]).mean().unstack(1).reindex(MODELS)
    return [[LABEL[m]] + [f3(tab.loc[m, h]) for h in H] for m in MODELS]


def skill_rows():
    tab = metrics["skill_vs_persistence"].groupby(level=[1, 2]).mean().unstack(1).reindex(MODELS).drop("persistence")
    return [[LABEL[m]] + [f"{tab.loc[m, h]:+.2f}" for h in H] for m in tab.index]


def storm_rows():
    rows = []
    for m in MODELS:
        if m == "climatology":
            continue
        rows.append([LABEL[m]] + [f"{storm_mean.loc[(m, h)]:.3f} ({storm_bias.loc[(m, h)]:+.2f})" for h in (24, 48, 72)])
    return rows


def peak_rows():
    rows = []
    for m in [x for x in ("persistence", "sarimax", "lightgbm", "catboost", "lstm_mse", "lstm_peak") if x in MODELS]:
        r = pooled.loc[m]
        rows.append([LABEL[m]] + [f"{r.loc[h, 'rmse_peak']:.2f} / {r.loc[h, 'bias_peak']:+.2f} / {100 * r.loc[h, 'under_rate_peak']:.0f}%"
                                  for h in (24, 72)])
    return rows


DM_MODELS = [m for m in ("sarimax", "lightgbm", "catboost", "lstm_mse", "lstm_peak") if m in MODELS]


def _p(p):
    return "< 0.001" if p < 1e-3 else f"{p:.3f}"


def dm_rows(horizons=(24, 72)):
    """One row per (site, lead time): the best model there and the DM p-value of every other learned model."""
    rows = []
    for b in BUOYS:
        for h in horizons:
            sub = dm[(dm.buoy == b) & (dm.horizon_h == h)].set_index("model")
            ref = sub["reference"].iloc[0]
            rows.append([b, f"{h} h", LABEL[ref]] + ["best" if m == ref else _p(sub.loc[m, "p_value"]) for m in DM_MODELS])
    return rows


DM_HEADER = ["Site", "Lead", "Best model"] + [LABEL[m] for m in DM_MODELS]
dm_learned = dm[dm.model.isin(DM_MODELS) & dm.reference.isin(DM_MODELS)]
dm_frac_learned = float((dm_learned["p_value"] < 0.05).mean())
dm_trees_ns = float((dm[dm.model.isin(["lightgbm", "catboost"]) & dm.reference.isin(["lightgbm", "catboost"])]["p_value"] >= 0.05).mean())

# training facts quoted in the text
_imp6 = pd.concat([pd.read_csv(RES / "phase3" / b / "catboost_importance.csv", index_col=0)["t+6h"] for b in BUOYS], axis=1).mean(axis=1)
IMP6_RECENT = float(_imp6[[f"era5_swh_lag{k}h" for k in (0, 1, 2, 3)]].sum())
LSTM_BEST_EPOCH = max(int(pd.read_csv(RES / "phase4" / b / f"history_{v}.csv", index_col=0)["val_loss"].idxmin()) + 1
                      for b in BUOYS for v in ("mse", "peak"))
TRIALS = {(b, m): len(pd.read_csv(RES / "phase3" / b / f"{m}_optuna.csv")) for b in BUOYS for m in ("lightgbm", "catboost")}


# ---------------------------------------------------------------- abstract
def _abstract():
    b24, b72 = BEST_ML[24], BEST_ML[72]
    s = (
        "Accurate forecasts of significant wave height (H_{s}) 6 to 72 hours ahead are essential for marine safety in "
        "the Bay of Bengal, where tropical cyclones raise H_{s} from 1–2 m to more than 10 m within a day. This report "
        "develops a data-driven alternative to numerical wave models and compares classical, tree-based and deep-learning "
        "forecasters under a strictly leakage-free protocol. Ten years (2015–2024) of hourly ECMWF ERA5 data at the "
        "positions of the INCOIS OMNI buoys BD08, BD10 and BD11 were aligned with a KD-tree on the sphere and enriched "
        "with physically based predictors: wind stress, pressure tendency and anomaly, the air–sea temperature "
        "difference and cyclically encoded directions. Direct multi-step LightGBM and CatBoost models, tuned by "
        "Bayesian optimisation on a pooled multi-horizon RMSE, and an encoder–decoder bidirectional LSTM with attention "
        "were trained on 2015–2020 and tested on 2023–2024, a period containing cyclones Mocha, Michaung, Remal and "
        "Dana; every model was scored on the same forecast origins. ")
    s += (f"With ERA5 H_{{s}} as the target, the best learned model reduced the test RMSE of persistence by "
          f"{gain_pers[24]:.0f}% at 24 h ({LABEL[b24]}, {f3(mean_rmse.loc[b24, 24])} m) and by {gain_pers[72]:.0f}% at "
          f"72 h ({LABEL[b72]}, {f3(mean_rmse.loc[b72, 72])} m)")
    if gain_arima:
        s += f", and the ARIMA baseline by {gain_arima[24]:.0f}% and {gain_arima[72]:.0f}% respectively"
    s += ". "
    if has_lstm:
        s += (f"A peak-weighted asymmetric loss lowered the share of under-predicted extreme sea states (above the "
              f"training 95th percentile) at 24 h from {100 * under[24][0]:.0f}% to {100 * under[24][1]:.0f}%, at a cost "
              f"of {cost[24]:.0f}% in overall RMSE. ")
    s += ("All forecasters lose skill at long lead times during cyclones, which points to forecast forcing as the "
          "next step. The pipeline is ready to take the buoy observations as the target once they are available.")
    return s


ABSTRACT = _abstract()
