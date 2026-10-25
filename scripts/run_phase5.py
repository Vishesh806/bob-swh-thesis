"""Phase 5: evaluation tables and thesis figures from the Phase 3 and Phase 4 test predictions.

    python scripts/run_phase5.py                    # every buoy with results
    python scripts/run_phase5.py --buoys BD08

Writes results/phase5/*.csv, results/phase5/summary.json and figures/*.png.
Only test-period (2023-2024) forecasts are used, on origins common to every model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bobswh import config, plots  # noqa: E402
from bobswh.evaluation import (dm_table, evaluate, largest_event, load_predictions, models_in,  # noqa: E402
                               peak_threshold, storm_metrics)

H = config.HORIZONS_H
LEARNED = ("sarimax", "lightgbm", "catboost", "lstm_mse", "lstm_peak")


def busiest_window(pred: pd.DataFrame, days: int, avoid: pd.Timestamp, h: int = 24) -> pd.Timestamp:
    """Start of the 30-day window with the most variable observed Hs, at least 20 days from ``avoid``."""
    obs = pd.Series(pred[f"obs__t+{h}h"].to_numpy(), index=pred.index + pd.Timedelta(hours=h)).asfreq("1h")
    std = obs.rolling(f"{days}D", min_periods=days * 20).std()
    centre = std.index - pd.Timedelta(days=days / 2)
    std = std[np.abs((centre - avoid) / pd.Timedelta(days=1)) > 20]
    return std.idxmax() - pd.Timedelta(days=days)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buoys", nargs="+", default=list(config.BUOYS))
    args = ap.parse_args()
    out = config.RESULTS_DIR / "phase5"
    figs = config.PROJECT_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(exist_ok=True)

    metrics, storms, dms, preds, summary = [], [], [], {}, {"buoys": {}}
    for b in args.buoys:
        try:
            pred = load_predictions(b)
        except FileNotFoundError:
            print(f"{b}: no results yet, skipped")
            continue
        preds[b] = pred
        thr = peak_threshold(b)
        m = evaluate(pred, H, thr)
        metrics.append(m.assign(buoy=b))
        storms.append(storm_metrics(pred, H).assign(buoy=b))
        dms.append(dm_table(pred, H).assign(buoy=b))
        models = models_in(pred)
        learned = [x for x in LEARNED if x in models]
        name, peak_t = largest_event(pred)
        summary["buoys"][b] = {"n_test_origins": len(pred), "peak_threshold_m": thr, "models": models,
                               "zoom_event": name, "zoom_peak_time": str(peak_t),
                               "zoom_peak_hs_m": float(pd.Series(pred["obs__t+24h"].to_numpy(),
                                                                 index=pred.index + pd.Timedelta(hours=24))[peak_t])}

        trio = [x for x in ("lightgbm", "catboost", "lstm_peak") if x in models]
        start = busiest_window(pred, 30, peak_t)
        plots.timeline(pred, start, 30, trio, (24, 72),
                       f"{b}: forecast vs ERA5 $H_s$, {start:%d %b %Y} + 30 days (test period)",
                       figs / f"fig_timeline_{b}.png")
        plots.timeline(pred, peak_t - pd.Timedelta(days=5), 9, learned, (24, 72),
                       f"{b}: cyclone {name}, peak $H_s$ {summary['buoys'][b]['zoom_peak_hs_m']:.1f} m on {peak_t:%d %b %Y}",
                       figs / f"fig_cyclone_{b}.png")
        best = [x for x in ("catboost", "lstm_peak") if x in models]
        plots.scatter(pred, ["sarimax"] + best, 24, f"{b}: forecast vs observed $H_s$ at t + 24 h (test)",
                      figs / f"fig_scatter_{b}.png")
        hist = {v: pd.read_csv(config.RESULTS_DIR / "phase4" / b / f"history_{v.split('_')[1]}.csv", index_col=0)
                for v in ("lstm_mse", "lstm_peak") if (config.RESULTS_DIR / "phase4" / b / f"history_{v.split('_')[1]}.csv").exists()}
        if hist:
            plots.history(hist, f"{b}: Seq2Seq LSTM training", figs / f"fig_lstm_history_{b}.png")
        print(f"{b}: {len(pred)} common test origins, models {models}, zoom on {name} ({peak_t:%Y-%m-%d})")

    if not metrics:
        sys.exit("no results to evaluate")
    allm = pd.concat(metrics).reset_index().set_index(["buoy", "model", "horizon_h"]).sort_index()
    allm.to_csv(out / "metrics_test.csv")
    pd.concat(storms).reset_index().set_index(["buoy", "model", "horizon_h"]).to_csv(out / "storm_metrics_test.csv")
    pd.concat(dms).to_csv(out / "diebold_mariano_test.csv", index=False)
    rmse = allm["rmse"].unstack("horizon_h")
    rmse.to_csv(out / "rmse_test.csv")
    allm["skill_vs_persistence"].unstack("horizon_h").to_csv(out / "skill_test.csv")

    common_models = [m for m in LEARNED if all(m in models_in(p) for p in preds.values())]
    plots.bars(rmse, common_models, (24, 72), "Test RMSE at t + 24 h and t + 72 h (2023–2024)", figs / "fig_rmse_bars.png")
    every = [m for m in ("persistence", "sarimax", "lightgbm", "catboost", "lstm_mse", "lstm_peak")
             if all(m in models_in(p) for p in preds.values())]
    plots.rmse_vs_horizon(allm, every, "Test RMSE by lead time", figs / "fig_rmse_horizon.png")
    plots.rmse_vs_horizon(allm, [m for m in every if m != "persistence"], "Skill relative to persistence",
                          figs / "fig_skill_horizon.png", value="skill_vs_persistence", ylabel="Skill score (1 − RMSE / RMSE$_{pers}$)")

    pooled = pd.concat(preds.values())
    thr_all = float(np.mean([summary["buoys"][b]["peak_threshold_m"] for b in preds]))
    pooled_m = evaluate(pooled, H, thr_all)
    pooled_m.to_csv(out / "metrics_test_pooled.csv")
    peak_models = [m for m in ("catboost", "lstm_mse", "lstm_peak") if m in models_in(pooled)]
    plots.peak_bias(pooled_m, peak_models, H, "Extreme sea states (observed $H_s$ ≥ training q95), all buoys",
                    figs / "fig_peak_bias.png")

    summary["rmse_test"] = {f"{b}|{m}": rmse.loc[(b, m)].round(4).to_dict() for b, m in rmse.index}
    summary["pooled"] = pooled_m.round(4).reset_index().to_dict(orient="records")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print("\nTest RMSE (m):")
        print(rmse)


if __name__ == "__main__":
    main()
