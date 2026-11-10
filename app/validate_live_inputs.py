"""How well do live Open-Meteo inputs stand in for ERA5? Writes app/data/live_validation_*.csv.

    .venv/bin/python app/validate_live_inputs.py

Over October-December 2024 (test period; cyclones Dana and Fengal) at the three sites:
1. agreement of each Open-Meteo input variable with ERA5 (correlation, bias, RMSE);
2. forecast RMSE of the saved tree models fed with Open-Meteo inputs vs. with ERA5 inputs, both scored against
   ERA5 Hs at t + h, on the same 3-hourly origins.
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bobswh import config  # noqa: E402
from bobswh.features import FeatureEngineer  # noqa: E402
from bobswh.live import MARINE_URL, MARINE_VARS, WEATHER_VARS, frame_from_json, inputs_at  # noqa: E402
from bobswh.trees import DirectMultiStepRegressor  # noqa: E402

HIST_WEATHER_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
START, END, WARMUP = "2024-10-01", "2024-12-31", "2024-08-20"
H = (6, 12, 24, 48, 72)
RAW = ["era5_u10", "era5_v10", "era5_msl", "era5_t2m", "era5_sst", "era5_swh", "era5_mwd", "era5_mwp"]


def get(url, params):
    q = urllib.parse.urlencode({k: (",".join(v) if isinstance(v, list) else v) for k, v in params.items()})
    with urllib.request.urlopen(f"{url}?{q}", timeout=120) as r:
        return json.loads(r.read().decode())


def main():
    out = ROOT / "app" / "data"
    agree, skill = [], []
    for b, site in config.BUOYS.items():
        common = {"latitude": site.lat, "longitude": site.lon, "start_date": WARMUP, "end_date": END, "timezone": "GMT"}
        live = frame_from_json(get(MARINE_URL, {**common, "hourly": list(MARINE_VARS)}),
                               get(HIST_WEATHER_URL, {**common, "hourly": list(WEATHER_VARS), "wind_speed_unit": "ms"}))
        era5 = pd.read_parquet(config.PROC_DIR / f"{b}_features.parquet", columns=RAW).loc[WARMUP:END]
        live = live.reindex(era5.index).ffill(limit=3)
        win = slice(START, END)
        for v in RAW:
            a, e = live.loc[win, v], era5.loc[win, v]
            if v == "era5_mwd":     # circular: compare the smallest angle difference
                d = ((a - e + 180) % 360) - 180
                agree.append({"buoy": b, "variable": v, "corr": np.nan, "bias": float(d.mean()), "rmse": float(np.sqrt((d ** 2).mean()))})
            else:
                agree.append({"buoy": b, "variable": v, "corr": float(a.corr(e)), "bias": float((a - e).mean()),
                              "rmse": float(np.sqrt(((a - e) ** 2).mean()))})
        f_live, f_era5 = FeatureEngineer().transform(live), FeatureEngineer().transform(era5)
        origins = pd.date_range(START, pd.Timestamp(END) - pd.Timedelta(hours=72), freq="3h")
        target = era5["era5_swh"]
        for m in ("lightgbm", "catboost"):
            model = DirectMultiStepRegressor.load(config.MODEL_DIR / f"{b}_{m}.joblib")
            X_live = pd.concat([inputs_at(f_live, model.feature_names_, o) for o in origins])
            X_era5 = pd.concat([inputs_at(f_era5, model.feature_names_, o) for o in origins])
            ok = X_live.notna().all(axis=1) & X_era5.notna().all(axis=1)
            p_live, p_era5 = model.predict(X_live[ok]), model.predict(X_era5[ok])
            for j, h in enumerate(H):
                y = target.reindex(X_live.index[ok] + pd.Timedelta(hours=h)).to_numpy()
                skill.append({"buoy": b, "model": m, "horizon_h": h, "n": int(ok.sum()),
                              "rmse_era5_inputs": float(np.sqrt(np.nanmean((p_era5[:, j] - y) ** 2))),
                              "rmse_live_inputs": float(np.sqrt(np.nanmean((p_live[:, j] - y) ** 2))),
                              "rmse_persistence": float(np.sqrt(np.nanmean((X_era5[ok]["era5_swh_lag0h"].to_numpy() - y) ** 2)))})
        print(b, "done")
    pd.DataFrame(agree).to_csv(out / "live_validation_inputs.csv", index=False)
    pd.DataFrame(skill).to_csv(out / "live_validation_skill.csv", index=False)
    s = pd.DataFrame(skill).groupby("horizon_h")[["rmse_era5_inputs", "rmse_live_inputs", "rmse_persistence"]].mean().round(3)
    print(pd.DataFrame(agree).groupby("variable")[["corr", "bias", "rmse"]].mean().round(3))
    print(s)


if __name__ == "__main__":
    main()
