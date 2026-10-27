"""Build the small, self-contained data bundle the dashboard reads (app/data/), from the project's results.

    .venv/bin/python app/prepare_data.py

Run after Phases 1-5. The bundle (a few MB) is committed so that the dashboard can be deployed on its own,
without the 400 MB of raw data, the feature tables or the trained models.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bobswh import config  # noqa: E402
from bobswh.evaluation import TEST_CYCLONES, load_predictions  # noqa: E402

OUT = ROOT / "app" / "data"
RES = config.RESULTS_DIR
BUOYS = list(config.BUOYS)
SERIES = ["era5_swh", "era5_mwp", "era5_wspd10", "era5_msl", "era5_sst", "era5_t2m", "era5_mwd"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # hourly ERA5 series at each site (the forecast target and the main drivers)
    frames = []
    for b in BUOYS:
        df = pd.read_parquet(config.PROC_DIR / f"{b}_features.parquet", columns=SERIES)
        frames.append(df.astype("float32").assign(buoy=b))
    pd.concat(frames).rename_axis("time").reset_index().to_parquet(OUT / "series.parquet", compression="zstd", index=False)

    # test-period forecasts of every model, on the common origins
    frames = []
    for b in BUOYS:
        p = load_predictions(b).astype("float32")
        frames.append(p.rename_axis("origin").reset_index().assign(buoy=b))
    pd.concat(frames).to_parquet(OUT / "predictions_test.parquet", compression="zstd", index=False)

    # tables produced by Phase 5
    for name in ("metrics_test.csv", "metrics_test_pooled.csv", "storm_metrics_test.csv", "diebold_mariano_test.csv",
                 "diebold_mariano_vs_arima.csv", "summary.json"):
        shutil.copy(RES / "phase5" / name, OUT / name)

    # model internals: importance, Optuna histories, chosen parameters, LSTM training curves
    imp, opt, hist, params, splits = [], [], [], {}, []
    for b in BUOYS:
        for m in ("lightgbm", "catboost"):
            i = pd.read_csv(RES / "phase3" / b / f"{m}_importance.csv", index_col=0)
            imp.append(i.rename_axis("feature").reset_index().assign(buoy=b, model=m))
            o = pd.read_csv(RES / "phase3" / b / f"{m}_optuna.csv")
            opt.append(o[["number", "value", "state"]].assign(buoy=b, model=m))
            params[f"{b}|{m}"] = json.loads((RES / "phase3" / b / f"{m}_params.json").read_text())
        for v in ("mse", "peak"):
            h = pd.read_csv(RES / "phase4" / b / f"history_{v}.csv", index_col=0)
            hist.append(h.rename_axis("epoch").reset_index().assign(buoy=b, model=f"lstm_{v}"))
        splits.append(pd.read_csv(RES / "phase3" / b / "split_summary.csv").assign(buoy=b))
    pd.concat(imp).to_csv(OUT / "importance.csv", index=False)
    pd.concat(opt).to_csv(OUT / "optuna_trials.csv", index=False)
    pd.concat(hist).to_csv(OUT / "lstm_history.csv", index=False)
    pd.concat(splits).to_csv(OUT / "split_summary.csv", index=False)
    (OUT / "params.json").write_text(json.dumps(params, indent=1))
    sens = RES / "phase4_sensitivity" / "BD08_summary.csv"
    if sens.exists():
        shutil.copy(sens, OUT / "lstm_sensitivity.csv")
    shutil.copy(config.PROC_DIR / "era5_node_alignment.csv", OUT / "node_alignment.csv")
    (OUT / "cyclones.json").write_text(json.dumps(TEST_CYCLONES, indent=1))

    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"wrote {len(list(OUT.iterdir()))} files, {total / 1e6:.1f} MB, to {OUT}")


if __name__ == "__main__":
    main()
