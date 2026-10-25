"""Chapter 6 (Conclusions and future work) and the appendices."""
import json

from common import *
from results_text import *


def _params_rows():
    rows = []
    for b in BUOYS:
        for m in ("lightgbm", "catboost"):
            p = RES / "phase3" / b / f"{m}_params.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            names = {"learning_rate": "learning rate", "num_leaves": "leaves", "depth": "depth",
                     "min_child_samples": "min. leaf samples", "min_data_in_leaf": "min. leaf samples", "subsample": "row subsample"}
            desc = "; ".join(f"{names[k]} {d['params'][k]:.3g}" if isinstance(d['params'][k], float) else f"{names[k]} {d['params'][k]}"
                             for k in names if k in d["params"])
            rows.append([b, LABEL[m], desc, ", ".join(str(x) for x in d["best_iterations"])])
    return rows


def _conclusions():
    items = [
        "A leakage-free pipeline was built from ERA5 time series at three OMNI buoy positions: quality control, KD-tree "
        "alignment on the sphere, physically based features (wind stress with a saturating drag coefficient, pressure "
        "tendency and anomaly, air–sea temperature difference, sine/cosine directions) and a containment rule that keeps "
        "every sample's inputs and targets inside one chronological block. Automated tests verify each guardrail.",
        f"The tree models and the MSE-trained LSTM beat every baseline at every lead time on the unseen 2023–2024 "
        f"test period. Averaged over the sites, the best of them reduced the persistence RMSE by {gain_pers[24]:.0f}% at 24 h and {gain_pers[72]:.0f}% "
        f"at 72 h" + (f", and the ARIMA RMSE by {gain_arima[24]:.0f}% and {gain_arima[72]:.0f}%." if gain_arima else "."),
        f"Direct multi-step gradient boosting and the Seq2Seq Bi-LSTM reached broadly similar accuracy; the most accurate "
        f"model was {LABEL[BEST[6]]} at 6 h, {LABEL[BEST[24]]} at 24 h and {LABEL[BEST[72]]} at 72 h. Feature importance "
        f"shows a physically consistent shift: persistence and short-term tendency at 6 h; seasonal phase, SST, pressure "
        f"and the 24 h pressure tendency at 72 h, where the model acts as a state-conditioned climatology.",
    ]
    if has_lstm:
        items.append(
            f"The peak-weighted loss reduced the share of under-predicted extreme sea states at 24 h from "
            f"{100 * under[24][0]:.0f}% to {100 * under[24][1]:.0f}% at the cost of {cost[24]:.0f}% higher overall RMSE, a "
            f"trade-off suited to warning applications.")
    items.append(
        "Inside cyclone windows the errors roughly double and storms are under-predicted on average; three days "
        "ahead, the information available at the origin is not sufficient to forecast a cyclone peak at a point.")
    return items


BLOCKS = [
    C("ch:conclusion", "Conclusions and Future Work"),
    H2("sec:conclusions", "Conclusions"),
    N(_conclusions()),
    H2("sec:future", "Future work"),
    B(["**Buoy targets.** Train on ERA5 and fine-tune on the OMNI observations, or use the buoy record directly as the "
       "target, to verify the models against measurements and to correct the smoothing of ERA5 cyclone peaks.",
       "**Forecast forcing.** Add NWP wind and pressure forecasts for t + h as decoder inputs of the Seq2Seq network; the "
       "architecture already separates known-future inputs from the history.",
       "**Spatial context.** Replace the point inputs by fields around the site (ConvLSTM or graph networks), so that "
       "approaching cyclones and remote swell are seen at their source.",
       "**Probabilistic output.** Quantile or distributional losses would give calibrated uncertainty bands, which are "
       "more useful for go/no-go decisions in marine operations than a single value.",
       "**Larger searches and ensembles.** Longer hyperparameter searches, LSTM tuning and an ensemble of the tree and "
       "LSTM forecasts are inexpensive next steps on a GPU."]),

    ("references",),
    ("appendix_mode",),
    C("ch:appA", "Hyperparameter Search Spaces and Selected Values"),
    T("tab:space", "Optuna search spaces (ranges sampled on a log scale are marked “log”)", ["Hyperparameter", "LightGBM", "CatBoost"],
      [["Learning rate", "0.01–0.2 (log)", "0.01–0.2 (log)"], ["Tree size", "leaves 15–255 (log)", "depth 4–10"],
       ["Minimum samples per leaf", "10–300 (log)", "1–200 (log)"],
       ["Row / column subsampling", "rows 0.5–1, columns 0.4–1", "rows 0.5–1 (Bernoulli)"],
       ["Regularisation", "L1 and L2: 10^{−8}–10 (log)", "L2: 1–30 (log); random strength 0.1–10 (log)"]], [1.8, 2.1, 2.1]),
    T("tab:params", "Selected hyperparameters and boosting rounds per horizon (6, 12, 24, 48, 72 h)",
      ["Buoy", "Model", "Selected parameters", "Boosting rounds"], _params_rows(), [0.6, 0.85, 3.2, 1.35]),
    C("ch:appB", "Software and Reproducibility"),
    P("All code is in the Python package *bobswh* (modules config, era5, buoy, alignment, preprocess, features, windowing, "
      "baselines, sarimax, trees, tuning, seq2seq, evaluation, plots) with command-line scripts for each phase. The "
      "experiments of this report are reproduced with:"),
    B(["Data: `scripts/download_era5_timeseries.py` retrieves the ERA5 time series for the three sites (three CDS requests).",
       "Phases 1 and 2: `scripts/build_features.py --era5-only` builds the quality-controlled feature tables.",
       "Phases 3 and 4: `scripts/run_era5_only.sh`, run with PHASE=3 and PHASE=4, trains the tuned trees and the two LSTM variants.",
       "Phase 5: `scripts/run_phase5.py` produces all tables and figures of {ch:results}."]),
    P("The automated test suite (pytest) checks, among others, that features are causal, that windows and splits contain "
      "no look-ahead, that the fast ARIMA forecasts equal the statsmodels reference, and that the peak-weighted loss has "
      "the stated weights. Software: Python 3.12, pandas 2.2, xarray 2023.6, SciPy 1.13, statsmodels 0.14, LightGBM 4.7, "
      "CatBoost 1.2, Optuna 5.0, TensorFlow 2.17 with Keras 3."),
]
