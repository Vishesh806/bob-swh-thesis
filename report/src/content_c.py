"""Chapter 5 (Results and discussion), generated from the Phase 5 outputs."""
import pandas as _pd

from common import *
from results_text import *


def _day(iso):
    return _pd.Timestamp(iso).strftime("%-d %B %Y")

HDR = ["Model", "6 h", "12 h", "24 h", "48 h", "72 h"]
W6 = [1.5, 0.9, 0.9, 0.9, 0.9, 0.9]


def _rmse_tables():
    out = []
    for b in BUOYS:
        out.append(T(f"tab:rmse_{b}", f"Test RMSE (m) at {b}, 2023–2024 ({n_test[b]:,} common forecast origins; best learned or "
                     f"baseline model per lead time in bold)", HDR, rmse_rows_with_best(b), W6))
    return out


def _trees_vs_lstm():
    parts = [f"{LABEL[BEST[h]]} at {h} h ({f3(mean_rmse.loc[BEST[h], h])} m)" for h in (6, 24, 72)]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _peak_paragraph():
    if not has_lstm:
        return P("The Seq2Seq LSTM results were not available when this report was generated.")
    return P(
        f"{{fig:peak}} and {{tab:peak}} isolate the effect of the peak-weighted loss on the extreme subset (observed H_{{s}} at "
        f"or above the training 95th percentile, {min(q95.values()):.2f}–{max(q95.values()):.2f} m depending on the site). "
        f"Trained with the MSE, the LSTM under-predicted {100 * under[24][0]:.0f}% of these cases at 24 h and "
        f"{100 * under[72][0]:.0f}% at 72 h, with a mean error of {biasp[24][0]:+.2f} m and {biasp[72][0]:+.2f} m. The "
        f"peak-weighted loss reduced the under-prediction rate to {100 * under[24][1]:.0f}% and {100 * under[72][1]:.0f}% and "
        f"moved the mean error at the peaks to {biasp[24][1]:+.2f} m and {biasp[72][1]:+.2f} m. The RMSE on the extreme "
        f"subset changed from {rmsep[24][0]:.2f} to {rmsep[24][1]:.2f} m at 24 h and from {rmsep[72][0]:.2f} to "
        f"{rmsep[72][1]:.2f} m at 72 h. The price is paid in ordinary conditions: over all samples the RMSE of the "
        f"peak-loss network is {cost[24]:+.0f}% at 24 h and {cost[72]:+.0f}% at 72 h relative to the MSE network, because "
        f"the loss tolerates over-prediction near the threshold. This is the intended trade-off for a warning system, in "
        f"which a missed peak is far costlier than a false alarm, but it means the two losses answer different questions "
        f"and should be chosen by the application rather than by RMSE alone.")


def _sensitivity_blocks():
    import pandas as pd
    path = RES / "phase4_sensitivity" / "BD08_summary.csv"
    if not path.exists():
        return [H2("sec:res_sensitivity", "LSTM sensitivity"),
                P("A sensitivity check of the LSTM training settings was not available when this report was generated.")]
    s = pd.read_csv(path)
    base = rmse.loc[("BD08", "lstm_mse")]
    rows = [["Baseline (core inputs, lr 10^{−3}, dropout 0.1)", str(LSTM_BEST_EPOCH)] + [f3(base[h]) for h in H]]
    names = {"core_lr3e-4_do0.2": "Core inputs, lr 3×10^{−4}, dropout 0.2",
             "extended_lr3e-4_do0.2": "Extended inputs (17), lr 3×10^{−4}, dropout 0.2"}
    for r in s.itertuples():
        rows.append([names.get(r.config, r.config), str(r.best_epoch)] + [f3(getattr(r, f"rmse_{h}h")) for h in H])
    core = s[s.config.str.startswith("core")].iloc[0]
    ext = s[s.config.str.startswith("extended")].iloc[0]
    dc = [100 * (core[f"rmse_{h}h"] / base[h] - 1) for h in H]
    de = [100 * (ext[f"rmse_{h}h"] / base[h] - 1) for h in H]
    core_lo, core_hi, ext_lo, ext_hi = min(dc), max(dc), min(de), max(de)
    return [
        H2("sec:res_sensitivity", "LSTM sensitivity"),
        T("tab:sens", "LSTM (MSE) sensitivity to training settings at BD08|LSTM (MSE) sensitivity at BD08: best early-stopping epoch and test RMSE (m)",
          ["Configuration", "Best epoch", "6 h", "12 h", "24 h", "48 h", "72 h"], rows, [2.3, 0.6, 0.62, 0.62, 0.62, 0.62, 0.62]),
        P(f"Because the LSTM stopped improving after very few epochs, the MSE variant was retrained at BD08 with a lower "
          f"learning rate and stronger dropout, with the core inputs and with an extended input set of 17 variables "
          f"(core plus wind stress, pressure tendencies and anomaly, SST, air–sea ΔT, wave direction and seasonal phase). "
          f"{{tab:sens}} shows the result. With the core inputs, the slower training moves the best epoch later but leaves "
          f"the test RMSE practically unchanged ({core_lo:+.0f}% to {core_hi:+.0f}%), so the LSTM results of this chapter "
          f"are not an artefact of the training settings. The extended inputs improve the RMSE at every lead time "
          f"({ext_lo:+.0f}% to {ext_hi:+.0f}%), to {f3(ext['rmse_24h'])} m at 24 h and {f3(ext['rmse_72h'])} m at 72 h "
          f"against {f3(rmse.loc[('BD08', 'lightgbm'), 24])} and {f3(rmse.loc[('BD08', 'lightgbm'), 72])} m for LightGBM: "
          f"the engineered physical features help the network too, and narrow but do not close the gap to the tree "
          f"models. A full hyperparameter search of the network, with the extended inputs, remains future work."),
    ]


BLOCKS = [
    C("ch:results", "Results and Discussion"),
    P("All results in this chapter are for the test period 2023–2024, which no model saw during training, early stopping "
      "or hyperparameter selection. The target is ERA5 H_{s} at the buoy positions ({sec:buoydata}), and every "
      "model is scored on exactly the same forecast origins at each site."),

    H2("sec:res_overall", "Overall accuracy by lead time"),
    P(f"{{tab:rmse_BD08}}–{{tab:rmse_BD11}} give the RMSE of every model at every lead time, and {{fig:rmse_h}} shows "
      f"the same results graphically. Averaged over the three sites, the most accurate model is {_trees_vs_lstm()}. "
      f"Errors grow with lead time for every model, from about {f2(mean_rmse.loc[BEST[6], 6])} m at 6 h to "
      f"{f2(mean_rmse.loc[BEST[72], 72])} m at 72 h for the best model, against {f2(mean_rmse.loc['persistence', 6])} m "
      f"and {f2(mean_rmse.loc['persistence', 72])} m for persistence."),
    *_rmse_tables(),
    F("fig:rmse_h", "fig_rmse_horizon.png", 6.0, "Test RMSE against lead time at the three sites"),
    P(f"Persistence is a strong reference at short lead times because hourly H_{{s}} is highly autocorrelated. ARIMA "
      f"improves on it at 6–24 h (mean RMSE {f3(mean_rmse.loc['sarimax', 6])} against {f3(mean_rmse.loc['persistence', 6])} m "
      f"at 6 h) but converges to persistence by 48–72 h ({f3(mean_rmse.loc['sarimax', 72])} against "
      f"{f3(mean_rmse.loc['persistence', 72])} m at 72 h), as expected of a univariate linear model whose forecasts "
      f"revert towards the recent level. Climatology does not depend on the lead time ({f3(mean_rmse.loc['climatology', 24])} m) "
      f"and is the weakest model up to 24 h. At 48 h it roughly matches persistence (better at BD08, worse at BD10 and "
      f"BD11), and at 72 h it beats both persistence and ARIMA at every site: three days ahead, the seasonal mean is a "
      f"better guess than the current state. A useful forecaster at long lead times must "
      f"therefore beat climatology, not only persistence. The learned models do both: at 24 h the best of them reduces "
      f"the persistence RMSE by {gain_pers[24]:.0f}%, and at 72 h by {gain_pers[72]:.0f}%, which is "
      f"{pct_gain(BEST[72], 'climatology', 72):.0f}% below climatology ({{fig:skill}}, {{tab:skill}})."),
    F("fig:skill", "fig_skill_horizon.png", 6.0, "Skill relative to persistence, SS = 1 − RMSE/RMSE_{pers}"),
    T("tab:skill", "Skill score relative to persistence, mean over the three sites", HDR, skill_rows(), W6),
    P(f"The R² of the best model decreases from {r2.loc[(BEST[6], 6)]:.2f} at 6 h to {r2.loc[(BEST[24], 24)]:.2f} at 24 h "
      f"and {r2.loc[(BEST[72], 72)]:.2f} at 72 h, and its scatter index rises from {si.loc[(BEST[6], 6)]:.2f} to "
      f"{si.loc[(BEST[72], 72)]:.2f} ({{tab:r2}}, {{tab:si}}). A scatter index below about 0.2 is usually regarded as good "
      f"agreement in wave-model verification [@mentaschi], which the learned models reach up to about 24 h."),
    T("tab:r2", "Coefficient of determination R², mean over the three sites", HDR, mean_rows("r2"), W6),
    T("tab:si", "Scatter index SI = RMSE / mean observation, mean over the three sites", HDR, mean_rows("si"), W6),
    F("fig:bars", "fig_rmse_bars.png", 6.0, "Test RMSE of ARIMA, the tree models and the LSTMs at 24 h and 72 h|Test RMSE of ARIMA, LightGBM, CatBoost and the two LSTM variants at 24 h and 72 h"),
    F("fig:scatter", "fig_scatter_BD08.png", 6.0, "Forecast against observed H_{s} at 24 h at BD08|Forecast against observed H_{s} at 24 h at BD08 (hexagonal bins, log density)"),

    H2("sec:res_significance", "Statistical significance"),
    P(f"{{tab:dm}} gives, for each site at 24 h and 72 h, the model with the lowest RMSE and the Diebold–Mariano p-values "
      f"of the other learned models against it. Two results are robust. First, testing directly against ARIMA (the "
      f"strongest classical baseline up to 48 h), both tree models and the MSE-trained LSTM are significantly better at "
      f"24 h and 72 h at BD08 and BD10 (p < 0.01). At BD11, where the sea is calmer and ARIMA relatively stronger, "
      f"CatBoost is significantly better at both lead times (p < 0.03), and LightGBM and the LSTM at 72 h only (p < 0.01); "
      f"at 24 h their gains there are not significant (p = 0.06 and 0.11). The improvement over the baselines is "
      f"therefore not a sampling artefact, but at the calmest site it is clear only at long lead times. Second, the differences between LightGBM and CatBoost are mostly not "
      f"significant ({100 * dm_trees_ns:.0f}% of their pairwise tests have p ≥ 0.05), and at 72 h the MSE-trained LSTM is "
      f"statistically indistinguishable from the best tree model at all three sites: the ranking among the strong models "
      f"is within the sampling uncertainty. Over all comparisons among learned models, {100 * dm_frac_learned:.0f}% are "
      f"significant at the 5% level. The Newey–West correction matters here: forecasts issued every hour overlap, which "
      f"reduces the effective sample size far below the 17,400 origins per site."),
    T("tab:dm", "Diebold–Mariano p-values against the best model at each site and lead time (small p: the model is significantly worse than the best)",
      DM_HEADER, dm_rows(), [0.5, 0.45, 1.0] + [0.81] * len(DM_MODELS)),

    H2("sec:res_timeline", "Forecasts over a continuous month"),
    P(f"{{fig:timeline}} compares the forecasts with ERA5 over the 30-day window of the test period with the most variable "
      f"sea state at BD08. At 24 h all three learned models follow the growth and decay of individual events closely, "
      f"with small phase errors at the onset of rapid growth. At 72 h the forecasts are smoother than the observations: "
      f"they keep the level of the monsoon sea but damp, and often delay, short events, which is the usual behaviour of point "
      f"forecasters trained with squared-error losses when the forcing that will drive an event is not yet visible at the "
      f"origin."),
    F("fig:timeline", "fig_timeline_BD08.png", 6.0, "Forecasts at BD08 over a 30-day test window|Forecast and observed H_{s} at BD08 over a 30-day test window, at 24 h and 72 h"),

    H2("sec:res_cyclones", "Performance during tropical cyclones"),
    P(f"{{fig:cyclone}} zooms on cyclone {zoom['BD08'][0]} at BD08 (peak H_{{s}} {zoom['BD08'][2]:.1f} m on "
      f"{_day(zoom['BD08'][1])}). At 24 h the learned models start the rise on time, because the growing wind stress and "
      f"falling pressure are already visible at the origin, but the trees and the MSE-trained LSTM level off near 4 m, "
      f"about two metres below the peak. The peak-loss LSTM comes closest to the peak. ARIMA, in contrast, reproduces the "
      f"shape of the event about a day late, as a persistence-like forecaster must. At 72 h no model reproduces the "
      f"peak: three days before it the cyclone is still far from the buoy and leaves little trace in the local wind, "
      f"pressure and wave history. {{tab:storm}} quantifies this over all named cyclones of the test period."),
    P(f"Inside the cyclone windows the RMSE of the best model at 24 h is {storm_mean.loc[(STORM_BEST[24], 24)]:.2f} m "
      f"against {mean_rmse.loc[STORM_BEST[24], 24]:.2f} m over all test samples, and at 72 h "
      f"{storm_mean.loc[(STORM_BEST[72], 72)]:.2f} m against {mean_rmse.loc[STORM_BEST[72], 72]:.2f} m, i.e. about twice "
      f"the all-sample error at both lead times. The trees reduce the storm-window RMSE of persistence and ARIMA by a "
      f"quarter to a third at 24–72 h, but their mean errors inside the windows are negative: storms are under-predicted on "
      f"average, increasingly so with lead time. The peak-loss LSTM is the exception, with a mean error within "
      f"±0.07 m at every lead time inside the cyclone windows."),
    F("fig:cyclone", "fig_cyclone_BD08.png", 6.0, f"Forecasts during cyclone {zoom['BD08'][0]} at BD08, at 24 h and 72 h"),
    T("tab:storm", "RMSE and mean error inside the cyclone windows of 2023–2024|RMSE (m) and mean error (m, in brackets) inside the windows of the named cyclones of 2023–2024, mean over the three sites",
      ["Model", "24 h", "48 h", "72 h"], storm_rows(), [1.8, 1.4, 1.4, 1.4]),
    F("fig:cyclone11", "fig_cyclone_BD11.png", 6.0, f"Forecasts during cyclone {zoom['BD11'][0]} at BD11, at 24 h and 72 h"),

    H2("sec:res_peak", "Effect of the peak-weighted loss"),
    _peak_paragraph(),
    F("fig:peak", "fig_peak_bias.png", 6.0, "Mean error and under-prediction rate on extreme sea states (observed H_{s} ≥ training q95), all sites"),
    T("tab:peak", "Scores on extreme sea states (observed H_{s} ≥ q95), all sites|Extreme subset (obs ≥ q95), all sites: RMSE (m) / mean error (m) / share under-predicted",
      ["Model", "24 h", "72 h"], peak_rows(), [1.8, 2.1, 2.1]),

    H2("sec:res_importance", "What the tree models learned"),
    P("{fig:importance} shows the CatBoost feature importance at 6 h and 72 h. At 6 h the forecast is dominated by the most "
      "recent values of H_{s} (the current value and the three preceding hours carry " + f"{100 * IMP6_RECENT:.0f}" + "% of the "
      "importance) and its short-term tendency, i.e. "
      "by persistence corrected for the current growth or decay. At 72 h the weight moves to the slow, large-scale state: "
      "the seasonal phase, SST, the mean sea-level pressure and its 24 h tendency, with the current H_{s} still "
      "contributing. SST in the bay follows the monsoon cycle closely, and the pressure level separates the monsoon trough "
      "from fair-weather conditions, so the 72 h model behaves like a climatology conditioned on the present large-scale "
      "state rather than like an extrapolation of the local sea. This is consistent with {sec:res_overall}, where "
      "climatology overtakes persistence at 48–72 h, and with the cyclone results: a storm that is not yet visible in these "
      "slow variables at the origin cannot be forecast three days ahead from local inputs. The 24 h pressure tendency is "
      "the main short-lived signal that survives at 72 h, which supports the inclusion of pressure features."),
    F("fig:importance", "fig_importance.png", 5.6, "CatBoost feature importance at 6 h and 72 h (top 15 features, mean over the sites)"),

    H2("sec:res_lstm", "Trees versus the Seq2Seq LSTM"),
    P("The gradient-boosted trees and the LSTM reach broadly similar accuracy, with the ranking depending on the lead time "
      "and the site. Three factors favour the trees in this setting. First, the trees see 78 engineered predictors, "
      "including wind stress, pressure tendency and direction encodings, whereas the LSTM was given the five core "
      "variables of the thesis objective; the feature engineering therefore does part of the work that the network has to "
      "learn. Second, the training set (about 43,000 hourly origins, strongly autocorrelated) is modest for a recurrent "
      "network: the early-stopping loss was lowest after only " + f"{LSTM_BEST_EPOCH}" + " epochs at every site, after which "
      "the network began to over-fit ({fig:history}), so the recurrent layers were only lightly trained. Third, the "
      "LSTM was trained with fixed hyperparameters, whereas the trees were tuned ({sec:res_sensitivity} reports "
      "a first sensitivity check). Even so, the MSE-trained LSTM is the most accurate model at 48 h and 72 h at BD11, "
      "and statistically equivalent to the best tree model at 72 h at every site. The LSTM, on the other hand, produces all five horizons "
      "from one jointly trained network, which keeps the forecast trajectory consistent across lead times, and it is the "
      "only architecture in which the peak-weighted loss could be applied directly."),
    F("fig:history", "fig_lstm_history_BD08.png", 6.0, "Training and early-stopping loss of the two LSTM variants at BD08"),

    *_sensitivity_blocks(),
    H2("sec:limitations", "Limitations"),
    B(["**Target.** The models are verified against ERA5 H_{s}, not against buoy measurements. ERA5 smooths cyclone "
       "peaks and is itself a model product, so the absolute errors against observations will be larger; the relative "
       "ranking of the forecasters and the effect of the loss are expected to carry over, but this must be confirmed with "
       "the OMNI data.",
       "**Forcing.** Only information up to the forecast origin is used. In operational practice NWP wind forecasts for "
       "t + h would be available and would raise the skill at 48–72 h, particularly in cyclones.",
       "**Point models.** Each site uses a 1.5° box around the buoy. Swell generated elsewhere in the bay is seen only "
       "through its local signature (period, direction), not at its source.",
       "**Computing budget.** Hyperparameter searches were limited to 15 minutes per study on a laptop CPU, and the LSTM "
       "was not tuned. Larger searches would probably narrow some of the differences between models."]),
]
