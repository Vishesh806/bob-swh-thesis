"""Chapters 1 (Introduction) and 2 (Literature review)."""
from common import *

BLOCKS = [
    C("ch:intro", "Introduction"),
    H2("sec:background", "Background"),
    P("The Bay of Bengal is a semi-enclosed basin of the north-eastern Indian Ocean bounded by India, Bangladesh, Myanmar and "
      "Sri Lanka. Its sea state is governed by the reversing monsoon: south-westerly winds from June to September generate "
      "a persistent wind sea and long-period swell, while weaker north-easterlies prevail from November to February. "
      "Superimposed on this seasonal cycle are tropical cyclones, which form mainly in the pre-monsoon (April–May) and "
      "post-monsoon (October–December) seasons and raise the significant wave height from about 1–2 m to more than 8–10 m "
      "within one or two days [@imd]. The coasts that surround the bay are among the most densely populated in the world, "
      "and its ports, fishing fleets, offshore oil and gas installations and coastal communities all depend on timely "
      "warnings of hazardous seas."),
    P("The **significant wave height** H_{s}, defined as four times the square root of the zeroth moment of the wave "
      "spectrum and close to the mean height of the highest third of the waves [@holthuijsen], is the single most widely "
      "used descriptor of the sea state. Operational forecasts of H_{s} are produced by third-generation spectral wave "
      "models such as WAM [@wamdi] and WAVEWATCH III [@ww3], which integrate the wave action balance equation forced by "
      "numerical weather prediction (NWP) winds. These models are physically complete but computationally expensive: a "
      "basin-scale ensemble run needs a supercomputer, a high-quality wind forcing and a long chain of pre- and "
      "post-processing. For an operator who needs a forecast at a fixed location, for example a mooring, a port approach "
      "or an offshore platform, a data-driven model that maps recent observations and forcing to future H_{s} can be "
      "orders of magnitude cheaper to run and can be retrained as new data arrive."),
    H2("sec:problem", "Problem statement"),
    P("Forecasting H_{s} at a point 6 to 72 hours ahead is a **multi-step time-series problem**. The simplest approach "
      "trains a one-step model and feeds its own predictions back as inputs to reach longer lead times (the recursive "
      "strategy). Errors then accumulate with every step, and the model is never trained on the inputs it receives at "
      "long lead times [@taieb]. A sounder approach is the **direct multi-step** (DMS) strategy, in which a separate "
      "mapping is learned for every lead time, or a sequence-to-sequence network that emits all lead times jointly "
      "without feedback. A second difficulty is that the most important forecasts are the rarest: tropical cyclones make "
      "up a small fraction of any record, and a model that minimises the mean squared error over all samples tends to "
      "under-predict exactly the peaks that matter for safety. A third is **information leakage**: sliding windows over "
      "an autocorrelated series make it easy to let future values reach the training set, which inflates every reported "
      "skill score. This thesis addresses all three."),
    H2("sec:objectives", "Objectives"),
    N(["To build a reproducible, leakage-free pipeline that ingests ECMWF ERA5 reanalysis and INCOIS OMNI buoy data, "
       "aligns them in space and time and engineers physically meaningful predictors (wind stress, pressure tendency, "
       "air–sea temperature difference, cyclically encoded directions).",
       "To forecast H_{s} at lead times of 6, 12, 24, 48 and 72 h with direct multi-step gradient-boosted trees "
       "(LightGBM, CatBoost) tuned by Bayesian optimisation on a multi-horizon objective.",
       "To design an encoder–decoder (sequence-to-sequence) bidirectional LSTM with attention for the same task, and a "
       "peak-weighted loss that penalises under-prediction during extreme sea states.",
       "To compare all models against persistence, climatology and ARIMA on an unseen test period (2023–2024) that "
       "contains several tropical cyclones, using common samples, standard wave-verification metrics and formal "
       "significance tests."]),
    H2("sec:scope", "Scope and contributions"),
    P("The study covers ten years (2015–2024) of hourly data at the positions of three OMNI moored buoys, BD08, BD10 and "
      "BD11, which sample the northern, central and south-western bay. Because the INCOIS buoy records had not been "
      "received when the experiments were run, all models in this report are trained and verified against **ERA5 "
      "significant wave height at the buoy positions**. The pipeline is written so that the buoy record can replace the "
      "ERA5 target without code changes, and {sec:limitations} discusses what this choice implies. The main "
      "contributions are:"),
    B(["A leakage-free windowing and splitting scheme in which every sample's inputs and all of its targets lie inside "
       "one chronological block, verified by automated tests.",
       "A fair, common-sample comparison of classical, tree-based and deep-learning forecasters at five lead times and "
       "three locations, including skill during the named cyclones of the test period.",
       "A peak-weighted asymmetric loss for the Seq2Seq LSTM, evaluated against an otherwise identical network trained "
       "with the mean squared error.",
       "An open, modular Python implementation with 44 automated tests, from which every table and figure in this report "
       "is regenerated."]),
    H2("sec:organisation", "Organisation of the report"),
    P("{ch:lit} reviews numerical and data-driven wave forecasting. {ch:data} describes the study area, the ERA5 and buoy "
      "data, and the preprocessing. {ch:method} presents the feature engineering, the forecasting strategy, the models and "
      "the verification framework. {ch:results} reports and discusses the results, and {ch:conclusion} draws the "
      "conclusions and outlines future work."),

    C("ch:lit", "Literature Review"),
    H2("sec:lit_numerical", "Numerical wave models and reanalyses"),
    P("Third-generation spectral models solve the action balance equation for the directional wave spectrum, with source "
      "terms for wind input, non-linear four-wave interactions and whitecapping dissipation [@holthuijsen]. WAM introduced "
      "this approach for global forecasting [@wamdi], and WAVEWATCH III extended it with improved physics and numerics "
      "[@ww3]. Their skill is limited mainly by the quality of the forcing winds, particularly inside tropical cyclones "
      "where the wind field is poorly resolved."),
    P("Reanalyses combine such a model with data assimilation over decades. ERA5, the fifth-generation ECMWF reanalysis, "
      "couples the IFS atmospheric model with WAM, assimilates satellite altimeter wave heights, and provides hourly fields "
      "from 1940 to the present: atmospheric variables on a 0.25° grid and wave variables on a 0.5° grid [@era5]. Its "
      "homogeneous, gap-free record makes it the standard training source for data-driven models where in-situ data are "
      "short or interrupted."),
    H2("sec:lit_ann", "Data-driven wave forecasting"),
    P("Neural networks were applied to wave forecasting from the late 1990s. Deo and Naidu forecast wave heights from "
      "preceding observations with a feed-forward network [@deo1998], and Deo et al. showed that networks driven by wind "
      "inputs can match empirical and numerical approaches at short lead times [@deo2001]. Londhe and Panchang extended "
      "point forecasts to 24 h with buoy data off the United States coast [@londhe], and Mandal and Prabaharan used "
      "recurrent networks to capture the temporal dependence of the wave record [@mandal]. Long short-term memory (LSTM) "
      "networks [@hochreiter], whose gated memory cells mitigate the vanishing-gradient problem of simple recurrent "
      "networks, have since become the default deep-learning model for H_{s} series; Fan et al. reported clear gains of "
      "LSTM over conventional networks for multi-hour forecasts [@fan]."),
    P("Sequence-to-sequence (Seq2Seq) encoder–decoder networks [@sutskever] map an input sequence to an output sequence of "
      "different length and are a natural fit for multi-horizon forecasting. Bidirectional encoders read the history in "
      "both directions [@schuster], and attention mechanisms let each output step weigh the relevant part of the input "
      "[@bahdanau]. In parallel, gradient-boosted decision trees [@friedman], in their modern implementations XGBoost "
      "[@xgboost], LightGBM [@lightgbm] and CatBoost [@catboost], have become the strongest general-purpose learners for "
      "tabular data. They need little preprocessing, handle heterogeneous predictors and are fast to train, and they can "
      "be applied to time series through lagged features."),
    H2("sec:lit_multistep", "Multi-step strategies and verification"),
    P("Ben Taieb et al. compared recursive, direct and multiple-output strategies and found that direct and "
      "multiple-output approaches avoid the error accumulation of recursion at long horizons [@taieb]. For verification, "
      "the wave-modelling community reports RMSE, bias, correlation and the scatter index; Mentaschi et al. showed that "
      "RMSE-based indices can favour models that under-predict variability, and recommended reporting bias-removed "
      "measures alongside them [@mentaschi]. Differences between forecasters evaluated on the same sample should be tested "
      "for significance, for which the Diebold–Mariano test [@dm] with the small-sample correction of Harvey et al. [@hln] "
      "is the standard tool."),
    H2("sec:gap", "Research gap"),
    P("Three gaps motivate this work. First, most data-driven studies forecast one lead time or use recursion, and few "
      "compare direct multi-step trees with Seq2Seq networks on identical samples. Second, published skill scores rarely "
      "document how look-ahead bias was excluded, although overlapping windows make leakage easy. Third, models are "
      "usually trained with a symmetric loss and scored on all samples, so their behaviour in cyclones, the cases of "
      "greatest consequence in the Bay of Bengal, is seldom examined separately. This thesis addresses each of these."),
]
