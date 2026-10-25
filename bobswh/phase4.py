"""Phase 4 orchestration: Seq2Seq LSTM for one buoy, with and without the peak-weighted loss.

Inputs:  ``{features_dir}/{buoy}_features.parquet`` and ``{buoy}_imputed.parquet`` (Phase 2).
Outputs in ``{out_dir}/{buoy}/``:

* ``split_summary.csv``          sample counts and time extent of fit / es / val / test
* ``metrics.csv``                per variant, split and horizon, including extreme-subset scores
* ``predictions.parquet``        observed and forecast Hs for val and test origins
* ``history_{variant}.csv``      loss and RMSE per epoch
* ``models/{buoy}_lstm_{variant}/``  model.keras + meta.json (scaler, threshold, config)

The blocks are the same as in Phase 3 (fit / es / val / test, with the same
containment rule), so the LSTM is trained, early-stopped and scored on the
same periods as the tree models.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping

import pandas as pd

from . import config
from .metrics import horizon_metrics
from .seq2seq import Seq2SeqConfig, Seq2SeqForecaster
from .windowing import ChronologicalSplitter, SlidingWindowGenerator, WindowSpec, carve_early_stopping_block

log = logging.getLogger(__name__)

SCORED_SPLITS = ("val", "test")

# Encoder inputs besides the target's own history.
INPUT_SETS: dict[str, tuple[str, ...]] = {
    # the variables named in the thesis objective: wind vector, pressure, wave period
    "core": ("era5_u10", "era5_v10", "era5_msl", "era5_mwp"),
    # core + the Phase 2 physics features (stress, SST, swell direction, pressure tendency, ...)
    "extended": ("era5_u10", "era5_v10", "era5_msl", "era5_mwp", "era5_swh", "era5_tau", "era5_tau_mean24h",
                 "era5_msl_tend3h", "era5_msl_tend24h", "era5_msl_anom", "era5_sst", "era5_air_sea_dt",
                 "era5_mwd_sin", "era5_mwd_cos", "era5_wind_wave_cos", "doy_sin", "doy_cos"),
}


@dataclass
class Phase4Runner:
    """Train and score Seq2Seq LSTM variants for one buoy.

    Attributes
    ----------
    input_set:
        Key of ``INPUT_SETS``; columns missing from the frame are skipped.
    variants:
        Loss variants to train: "peak" (peak-weighted) and/or "mse" (the ablation).
    sequence_stride_h:
        Spacing of encoder steps; 3 gives 25 steps for a 72 h lookback.
    """

    buoy: str
    target_col: str = "buoy_hs"
    features_dir: Path = config.PROC_DIR
    out_dir: Path = config.RESULTS_DIR / "phase4"
    model_dir: Path = config.MODEL_DIR
    splits: Mapping[str, tuple[str, str]] = field(default_factory=lambda: dict(config.SPLITS))
    es_days: int = config.EARLY_STOPPING_DAYS
    horizons_h: tuple[int, ...] = config.HORIZONS_H
    lookback_h: int = config.LOOKBACK_H
    sequence_stride_h: int = 1
    input_set: str = "core"
    variants: tuple[str, ...] = ("peak", "mse")
    model_config: Seq2SeqConfig = field(default_factory=Seq2SeqConfig)
    verbose: int = 2

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_parquet(Path(self.features_dir) / f"{self.buoy}_features.parquet")
        imputed = pd.read_parquet(Path(self.features_dir) / f"{self.buoy}_imputed.parquet")
        return df, imputed

    def build_samples(self, df: pd.DataFrame, imputed: pd.DataFrame):
        inputs = tuple(c for c in INPUT_SETS[self.input_set] if c in df and c != self.target_col)
        spec = WindowSpec(target_col=self.target_col, input_cols=inputs, lookback_h=self.lookback_h,
                          horizons_h=self.horizons_h, sequence_stride_h=self.sequence_stride_h)
        data = SlidingWindowGenerator(spec, imputed=imputed).sequences(df)
        splitter = ChronologicalSplitter(carve_early_stopping_block(self.splits, self.es_days),
                                         self.lookback_h, max(self.horizons_h))
        idx = splitter.split(data.anchors)
        for name in ("fit", "es", *SCORED_SPLITS):
            if len(idx[name]) == 0:
                raise ValueError(f"block '{name}' has no complete samples; check the data coverage and SPLITS")
        return data, idx, splitter.summary(data.anchors, idx)

    def run(self) -> pd.DataFrame:
        """Train every variant; return the long metrics table (model, split, horizon)."""
        out = Path(self.out_dir) / self.buoy
        out.mkdir(parents=True, exist_ok=True)
        df, imputed = self.load()
        data, idx, summary = self.build_samples(df, imputed)
        summary.to_csv(out / "split_summary.csv")
        blocks = {k: data.subset(v) for k, v in idx.items()}
        log.info("samples per block: %s; encoder input %s", {k: len(v) for k, v in blocks.items()}, data.X.shape[1:])

        tables, preds, threshold = [], {}, None
        for variant in self.variants:
            name = f"lstm_{variant}"
            fc = Seq2SeqForecaster(self.horizons_h, data.feature_names, 0, replace(self.model_config, loss=variant))
            fit, es = blocks["fit"], blocks["es"]
            history = fc.fit(fit.X, fit.Y, fit.anchors, es.X, es.Y, es.anchors, verbose=self.verbose)
            history.to_csv(out / f"history_{variant}.csv")
            fc.save(Path(self.model_dir) / f"{self.buoy}_{name}")
            threshold = fc.threshold_
            preds[name] = {s: fc.predict(blocks[s].X, blocks[s].anchors) for s in SCORED_SPLITS}

        frames = []
        for split in SCORED_SPLITS:
            obs = blocks[split].Y
            frame = pd.DataFrame(obs, index=blocks[split].anchors, columns=[f"obs__t+{h}h" for h in self.horizons_h])
            for name, p in preds.items():
                tables.append(horizon_metrics(obs, p[split], self.horizons_h, threshold).assign(model=name, split=split))
                for j, h in enumerate(self.horizons_h):
                    frame[f"{name}__t+{h}h"] = p[split][:, j]
            frames.append(frame.assign(split=split))
        metrics = pd.concat(tables).reset_index().set_index(["model", "split", "horizon_h"]).sort_index()
        metrics.to_csv(out / "metrics.csv")
        pd.concat(frames).to_parquet(out / "predictions.parquet")
        with open(out / "run_config.json", "w") as fh:
            json.dump({**{k: v for k, v in asdict(self).items() if k != "model_config"},
                       "model_config": asdict(self.model_config), "peak_threshold_m": threshold,
                       "feature_names": data.feature_names}, fh, indent=2, default=str)
        return metrics
