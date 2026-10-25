"""Phase 5 tests: common-sample join, skill, Diebold-Mariano test, cyclone windows."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bobswh.evaluation import diebold_mariano, dm_table, evaluate, load_predictions, storm_mask

H = (6, 12, 24, 48, 72)


def _pred_frame(index, models, rng, noise):
    obs = rng.uniform(0.5, 3.0, (len(index), len(H)))
    df = pd.DataFrame({f"obs__t+{h}h": obs[:, j] for j, h in enumerate(H)}, index=index)
    for m in models:
        for j, h in enumerate(H):
            df[f"{m}__t+{h}h"] = obs[:, j] + rng.normal(0, noise[m] * (1 + j), len(index))
    return df


def test_load_predictions_joins_on_common_origins(tmp_path):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-04", periods=200, freq="1h")
    p3 = _pred_frame(idx, ["persistence", "lightgbm"], rng, {"persistence": 0.3, "lightgbm": 0.1})
    p4 = p3[[c for c in p3 if c.startswith("obs__")]].iloc[10:].copy()
    p4[[f"lstm_peak__t+{h}h" for h in H]] = p3[[f"lightgbm__t+{h}h" for h in H]].iloc[10:].to_numpy()
    for phase, df in (("phase3", p3), ("phase4", p4)):
        (tmp_path / phase / "SYN").mkdir(parents=True)
        df.assign(split="test").to_parquet(tmp_path / phase / "SYN" / "predictions.parquet")
    out = load_predictions("SYN", tmp_path)
    assert len(out) == 190 and out.index[0] == idx[10]                       # only origins common to both phases
    m = evaluate(out, H)
    assert m.loc[("persistence", 24), "skill_vs_persistence"] == pytest.approx(0.0)
    assert (m.xs("lightgbm", level="model")["skill_vs_persistence"] > 0).all()


def test_load_predictions_rejects_mismatched_observations(tmp_path):
    rng = np.random.default_rng(1)
    idx = pd.date_range("2023-01-04", periods=50, freq="1h")
    p3 = _pred_frame(idx, ["lightgbm"], rng, {"lightgbm": 0.1})
    p4 = p3.copy()
    p4["obs__t+6h"] += 1.0
    for phase, df in (("phase3", p3), ("phase4", p4.rename(columns=lambda c: c.replace("lightgbm", "lstm_mse")))):
        (tmp_path / phase / "SYN").mkdir(parents=True)
        df.assign(split="test").to_parquet(tmp_path / phase / "SYN" / "predictions.parquet")
    with pytest.raises(ValueError, match="observations differ"):
        load_predictions("SYN", tmp_path)


def test_diebold_mariano_detects_a_worse_model_only():
    rng = np.random.default_rng(2)
    good = rng.normal(0, 1.0, 5000)
    bad = rng.normal(0, 1.5, 5000)
    stat, p = diebold_mariano(bad, good, h_steps=6)
    assert stat > 0 and p < 1e-6
    twin = rng.normal(0, 1.0, 5000)
    _, p_same = diebold_mariano(twin, good, h_steps=6)
    assert p_same > 0.01


def test_dm_table_uses_best_model_as_reference():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2023-01-04", periods=3000, freq="1h")
    pred = _pred_frame(idx, ["persistence", "lightgbm", "catboost"], rng,
                       {"persistence": 0.4, "lightgbm": 0.1, "catboost": 0.2})
    t = dm_table(pred, H)
    assert (t["reference"] == "lightgbm").all() and (t["p_value"] < 0.05).all()


def test_storm_mask_uses_valid_time():
    idx = pd.DatetimeIndex(["2023-05-07 00:00", "2023-05-12 00:00", "2023-06-20 00:00"])
    assert storm_mask(idx, 24).tolist() == [True, True, False]               # 7 May + 24 h = 8 May, inside Mocha
    assert storm_mask(idx[:1], 6).tolist() == [False]
