"""Chapters 3 (Study area and data) and 4 (Methodology). Data tables are computed from the processed files."""
import pandas as pd

from common import *

PROC = ROOT / "data" / "processed"
BUOYS = {"BD08": (18.15, 89.67, "northern bay"), "BD10": (16.32, 87.98, "central bay"), "BD11": (13.58, 84.11, "south-western bay")}


def _stats_rows():
    rows = []
    for b in BUOYS:
        df = pd.read_parquet(PROC / f"{b}_features.parquet", columns=["era5_swh", "era5_mwp", "era5_wspd10", "era5_msl"])
        hs = df["era5_swh"]
        rows.append([b, f"{hs.mean():.2f}", f"{hs.std():.2f}", f"{hs.quantile(0.95):.2f}", f"{hs.quantile(0.99):.2f}",
                     f"{hs.max():.2f}", f"{df['era5_mwp'].mean():.1f}", f"{df['era5_wspd10'].mean():.1f}", f"{df['era5_msl'].min():.0f}"])
    return rows


def _node_rows():
    a = pd.read_csv(PROC / "era5_node_alignment.csv").drop_duplicates(["site", "grid_resolution_deg"])
    rows = []
    for b, (lat, lon, region) in BUOYS.items():
        atm = a[(a.site == b) & (a.grid_resolution_deg == 0.25)].iloc[0]
        wav = a[(a.site == b) & (a.grid_resolution_deg == 0.5)].iloc[0]
        rows.append([b, region, f"{lat:.2f} N, {lon:.2f} E", f"{atm.node_lat:.2f} N, {atm.node_lon:.2f} E ({atm.distance_km:.0f} km)",
                     f"{wav.node_lat:.2f} N, {wav.node_lon:.2f} E ({wav.distance_km:.0f} km)"])
    return rows


def _event_rows(n=4):
    rows = []
    for b in BUOYS:
        df = pd.read_parquet(PROC / f"{b}_features.parquet", columns=["era5_swh", "era5_msl"])
        hs = df["era5_swh"].sort_values(ascending=False)
        picked = []
        for t, v in hs.items():
            if all(abs((t - p).total_seconds()) > 5 * 86400 for p, _ in picked):
                picked.append((t, v))
            if len(picked) == n:
                break
        for t, v in picked:
            p = df["era5_msl"][t - pd.Timedelta("2D"):t + pd.Timedelta("2D")].min()
            rows.append([b, f"{t:%d %b %Y %H}:00", f"{v:.1f}", f"{p:.0f}"])
    return rows


VARS = [
    ["u_{10}, v_{10}", "10 m wind components", "m/s", "0.25°"],
    ["T_{2m}", "2 m air temperature", "K → °C", "0.25°"],
    ["MSLP", "Mean sea-level pressure", "Pa → hPa", "0.25°"],
    ["SST", "Sea surface temperature", "K → °C", "0.25°"],
    ["H_{s}", "Significant height of combined wind waves and swell", "m", "0.5°"],
    ["θ_{w}", "Mean wave direction (coming from)", "°", "0.5°"],
    ["T_{m}", "Mean wave period", "s", "0.5°"],
]

FEATURES = [
    ["Wind speed U_{10}", "√(u_{10}² + v_{10}²)", "Forcing of wind-sea growth"],
    ["Wind direction θ_{U}", "atan2(−u, −v) mod 360°", "Same convention as θ_{w}"],
    ["Air density ρ_{a}", "p / (R_{d} T)", "Scales the momentum flux"],
    ["Wind stress τ", "ρ_{a} C_{d} |U| (u, v)", "Momentum input to waves"],
    ["Stress history", "mean τ, max U_{10} over 6–24 h", "Duration-limited growth"],
    ["Pressure tendency", "p(t) − p(t − h), h = 3–24 h", "Approaching depression"],
    ["Pressure anomaly", "p − 30-day trailing mean", "Synoptic signal"],
    ["Air–sea ΔT", "T_{2m} − SST", "Surface-layer stability"],
    ["Direction encodings", "sin θ, cos θ (θ_{w}, θ_{U})", "No 359° → 0° jump"],
    ["Wind–wave alignment", "cos(θ_{U} − θ_{w})", "Wind sea vs. crossing swell"],
    ["Wave steepness", "H_{s} / L_{0}, L_{0} = gT_{m}²/2π", "Wind sea vs. swell"],
    ["H_{s} tendency", "H_{s}(t) − H_{s}(t − h), h = 3–12 h", "Growth or decay phase"],
    ["Seasonal phase", "sin, cos of 2π·doy/365.25", "Monsoon cycle"],
]

BLOCKS = [
    C("ch:data", "Study Area and Data"),
    H2("sec:area", "The Bay of Bengal and the buoy sites"),
    P("The study area is the Bay of Bengal between 4° and 23° N and 79° and 100° E. Three sites of the INCOIS Ocean Moored "
      "buoy Network for the northern Indian Ocean (OMNI) [@incois] were selected to sample different wave regimes: BD08 in "
      "the northern bay, on the track of most cyclones that recurve towards Odisha, West Bengal, Bangladesh and Myanmar; "
      "BD10 in the central bay; and BD11 in the south-western bay off the Andhra Pradesh and Tamil Nadu coasts, which is "
      "exposed to post-monsoon cyclones that move west. {tab:sites} lists their nominal positions and the ERA5 grid nodes "
      "matched to them ({sec:align})."),
    T("tab:sites", "Buoy sites and the ERA5 nodes matched to them (distance from the buoy in brackets)",
      ["Buoy", "Region", "Nominal position", "Atmospheric node (0.25°)", "Wave node (0.5°)"], _node_rows(),
      [0.55, 1.05, 1.3, 1.55, 1.55]),
    H2("sec:era5", "ERA5 reanalysis"),
    P("ERA5 [@era5] supplies hourly fields of the variables in {tab:vars} for 1 January 2015 to 31 December 2024, i.e. "
      "87,672 hours at every site. The data were retrieved from the Copernicus Climate Data Store (CDS) through its "
      "time-series service [@era5ts], which serves the same reanalysis values as the gridded product for a box of at most "
      "2° × 2° over any period. One request per buoy (a 1.5° × 1.5° box) replaced the 60 two-month requests that the size "
      "limit of the gridded product would have required; the values were checked to agree with the gridded download for "
      "January 2015 to within 0.0002 in every variable. Every series is complete: there are no missing hours in the ten "
      "years at any site."),
    T("tab:vars", "ERA5 variables, units and native grid|ERA5 variables used, their units (as delivered → as used) and native grid", ["Symbol", "Variable", "Units", "Grid"],
      VARS, [1.0, 2.9, 1.3, 0.8]),
    P("Summary statistics are given in {tab:stats}. Mean H_{s} is about 1.5 m at all three sites, but the tails differ: "
      "H_{s} exceeded 10 m at BD10 during cyclone Amphan (2020) and at BD08 during Yaas (2021), while the maximum at BD11 "
      "is about 7 m. The seasonal cycle ({fig:season}) peaks in July at the two northern sites, with median H_{s} above "
      "2 m during the south-west monsoon, whereas BD11 has a flatter cycle with relatively higher seas during the "
      "north-east monsoon (November–January). The largest events at each site ({tab:events}) all coincide with named tropical "
      "cyclones in the IMD records [@imd], which confirms that the reanalysis resolves cyclone-driven seas at these "
      "positions, if with some smoothing of the peak."),
    T("tab:stats", "Statistics of ERA5 at the buoy sites, 2015–2024",
      ["Buoy", "Mean H_{s} (m)", "Std (m)", "q95 (m)", "q99 (m)", "Max (m)", "Mean T_{m} (s)", "Mean U_{10} (m/s)", "Min MSLP (hPa)"],
      _stats_rows(), [0.55, 0.65, 0.55, 0.6, 0.6, 0.6, 0.75, 0.85, 0.85]),
    T("tab:events", "Largest wave events at each site (distinct events at least five days apart)",
      ["Buoy", "Time of peak (UTC)", "Peak H_{s} (m)", "Min MSLP within ±2 days (hPa)"], _event_rows(), [0.8, 1.9, 1.3, 2.0]),
    F("fig:record", "fig_record.png", 5.4,
      "ERA5 significant wave height at the three sites, 2015–2024|ERA5 significant wave height at the three sites, 2015–2024, with the fit, early-stopping, validation and test blocks shaded"),
    F("fig:season", "fig_season.png", 5.0, "Monthly distribution of H_{s} at the three sites (median and 5–95 % range)"),
    H2("sec:buoydata", "INCOIS OMNI buoy data"),
    P("The OMNI buoys measure H_{s}, wave period and direction, wind, pressure, air temperature and SST, typically every "
      "hour or three hours. The pipeline includes a reader for INCOIS CSV/TXT exports that maps the varying column headers "
      "to canonical names, removes sentinel codes and converts timestamps to UTC, together with range checks, an "
      "isolated-spike test and short-gap filling ({sec:preprocess}). The buoy records had not been received when "
      "the experiments in this report were run, so ERA5 H_{s} at the buoy positions serves as the forecast target "
      "throughout; the buoy data can be substituted without changes to the code."),

    C("ch:method", "Methodology"),
    P("The methodology is organised in five phases ({fig:flow}): data ingestion and preprocessing, feature engineering and "
      "sample generation, baseline and tree-based models, the Seq2Seq LSTM, and evaluation. Every phase is a module of the "
      "Python package written for this work, and an automated test suite checks each guardrail described below."),
    F("fig:flow", "fig_flowchart.png", 4.6, "Methodology: the five phases of the forecasting framework"),
    H2("sec:preprocess", "Phase 1: ingestion, quality control and alignment"),
    H3("sec:qc", "Quality control and regularisation"),
    P("Values outside physically admissible ranges (for example 0 ≤ H_{s} ≤ 20 m and 870 ≤ p ≤ 1060 hPa) are rejected. "
      "Only negative sentinel codes (−999, −99.9, …) are removed as codes, because 999 hPa is a valid pressure inside a "
      "cyclone. Sensor records are averaged onto an hourly grid in right-closed bins, (t − 1 h, t] ↦ t, so that a value "
      "labelled t never contains later information; directions are averaged as unit vectors. An isolated spike is a "
      "sample that departs from both neighbours by more than 1.5 m in the same sense while the neighbours agree within "
      "0.5 m; sustained storm peaks pass this test, unlike a standard-deviation filter, which would clip them."),
    H3("sec:gaps", "Gap filling"),
    P("Gaps of at most three hours are filled by forward fill, which is causal. Linear, PCHIP and cubic interpolation are "
      "available but use the first value after the gap, which would leak up to three hours of future information into the "
      "inputs. Filled values are flagged, and a filled value is never used as a training or verification target. Longer "
      "gaps are left missing and remove every sample that touches them."),
    H3("sec:align", "Spatial alignment"),
    P("Grid nodes and buoy positions are mapped to Cartesian coordinates on the unit sphere and indexed with a KD-tree. The "
      "chord distance c = 2 sin(d/2R) increases monotonically with the great-circle distance:"),
    EQ("eq:haversine", r"d = 2R\,\arcsin\sqrt{\sin^2\frac{\Delta\varphi}{2} + \cos\varphi_1\cos\varphi_2\,\sin^2\frac{\Delta\lambda}{2}},\qquad R = 6371\ \mathrm{km}"),
    P("Hence the KD-tree nearest neighbour is exactly the haversine nearest neighbour. Only *valid* nodes enter the tree: "
      "nodes where any field is undefined (land in the wave grid) are excluded, so a coastal buoy is never matched to a "
      "land point. The atmospheric (0.25°) and wave (0.5°) grids are matched separately ({tab:sites})."),

    H2("sec:features", "Phase 2: feature engineering"),
    P("The predictors, summarised in {tab:features}, encode the physics of wave generation."),
    P("The surface wind stress and the air density are:"),
    EQ("eq:stress", r"\vec{\tau} = \rho_a\, C_d(U_{10})\, U_{10}\, (u_{10},\, v_{10}),\qquad \rho_a = \frac{p}{R_d\,T}"),
    P("Here τ is the momentum flux from the wind to the sea surface. The drag coefficient follows Large and Pond "
      "[@largepond]:"),
    EQ("eq:cd", r"10^3\,C_d = 1.2\quad (U_{10} < 11\ \mathrm{m/s}),\qquad 10^3\,C_d = 0.49 + 0.065\,U_{10}\quad (11 \leq U_{10} \leq 25\ \mathrm{m/s})"),
    P("Above 25 m/s the coefficient is held at its 25 m/s value, because field measurements in tropical cyclones show "
      "C_{d} levelling off "
      "at high wind speeds [@powell]; an alternative scheme after Wu [@wu] is also implemented. Waves are duration-limited, "
      "so the stress is also averaged over the preceding 6, 12 and 24 h. A falling pressure is the earliest sign of an "
      "approaching depression, and the air–sea temperature difference measures the stability of the surface layer. "
      "Directions are encoded by their sine and cosine to remove the 359° → 0° discontinuity, and raw degree values are "
      "excluded from the model inputs. Every feature at time t uses data at times ≤ t only (backward differences, trailing "
      "windows); a test confirms that changing future values leaves all earlier features unchanged."),
    T("tab:features", "Engineered features", ["Feature", "Definition", "Physical rationale"], FEATURES, [1.65, 2.35, 2.0]),

    H2("sec:windows", "Forecasting strategy and sample generation"),
    P("For a forecast origin t, look-back L = 72 h and horizons h ∈ {6, 12, 24, 48, 72} h, a sample is:"),
    EQ("eq:window", r"\mathbf{X}_t = [\mathbf{x}_{t-L},\ \ldots,\ \mathbf{x}_t],\qquad \mathbf{Y}_t = [H_s(t+6),\ H_s(t+12),\ H_s(t+24),\ H_s(t+48),\ H_s(t+72)]"),
    P("Each horizon is a target of its own (direct multi-step), so no forecast is ever fed back as an input. The tree models "
      "receive a flat layout of lags H_{s}(t − k), k ∈ {0, 1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 72} h, of H_{s}, ERA5 wind "
      "stress, wind speed and pressure, plus all other features at t (78 inputs). The LSTM receives the sequence itself. A "
      "sample is kept only if every input over [t − L, t] and every target is available, and both layouts keep exactly "
      "the same origins."),
    H3("sec:splits", "Chronological blocks without leakage"),
    T("tab:splits", "Chronological blocks", ["Block", "Period", "Use", "Cyclones"],
      [["Fit", "2015–2019", "Training", "Fani (2019) and others"],
       ["Early stopping", "2020", "Number of boosting rounds / epochs", "Amphan"],
       ["Validation", "2021–2022", "Hyperparameter selection", "Yaas, Asani, Sitrang"],
       ["Test", "2023–2024", "Reported results only", "Mocha, Michaung, Remal, Dana, Fengal"]], [1.2, 1.0, 2.0, 1.8]),
    P("The record is divided into four blocks ({tab:splits}). A sample belongs to block [T_{s}, T_{e}] only if:"),
    EQ("eq:contain", r"t - L \geq T_s \quad\mathrm{and}\quad t + \max h \leq T_e"),
    P("The second condition removes the usual multi-step leak, in which the last training samples have targets inside the "
      "validation period; the first keeps the blocks fully independent. Early stopping uses the 2020 block and "
      "hyperparameter selection the 2021–2022 block, so the validation score is optimised only once, and the test block "
      "is used for reporting only. Every block contains named cyclones."),

    H2("sec:baselines", "Phase 3a: reference forecasts"),
    P("**Persistence** forecasts Ĥ_{s}(t + h) = H_{s}(t), which is hard to beat at short lead times. **Climatology** "
      "forecasts the training-period mean for the calendar day of t + h, smoothed over ±15 days. **ARIMA**(p, d, q) "
      "[@box] models the differenced series as:"),
    EQ("eq:arima", r"\phi(B)\,(1 - B)^d\, H_{s,t} = \theta(B)\,\varepsilon_t"),
    P("Here B is the backshift operator. The order (2, 1, 1) was chosen by AIC, and the parameters were estimated on the last "
      "two years of the training period with statsmodels [@statsmodels]. At test time the state-space form [@durbin] is "
      "filtered through the record with fixed parameters, and the h-step forecast from every origin is obtained from the "
      "filtered state, a_{t+h|t} = T a_{t+h−1|t}, using information up to t only. This reproduces the statsmodels "
      "forecast at each origin to machine precision but needs a single filter pass for all 17,400 test origins. The "
      "baseline is univariate because forcing at t + h is not known at forecast time."),

    H2("sec:trees", "Phase 3b: direct multi-step gradient boosting"),
    P("For each horizon, an independent ensemble of regression trees is trained [@friedman]:"),
    EQ("eq:gbdt", r"\hat{H}_s(t+h) = f_h(\mathbf{x}_t) = \sum_{m=1}^{M_h} \eta\, g_{h,m}(\mathbf{x}_t)"),
    P("LightGBM [@lightgbm] grows trees leaf-wise on histogram-binned features; CatBoost [@catboost] uses symmetric "
      "(oblivious) trees with ordered boosting, which reduces the prediction shift of classical gradient boosting. The "
      "number of rounds M_{h} is set by early stopping on the 2020 block for each horizon separately. Hyperparameters "
      "were searched with the tree-structured Parzen estimator [@tpe] of Optuna [@optuna], with a median pruner, "
      "minimising the pooled multi-horizon RMSE on the validation block:"),
    EQ("eq:obj", r"J(\vartheta) = \sqrt{\frac{1}{5}\sum_{h} \mathrm{RMSE}_h(\vartheta)^2}"),
    P("One parameter set is shared by all horizons. Every third training origin was used during the search (adjacent "
      "hourly origins are nearly collinear), and the final models were refitted on all origins. Within the computing "
      "budget of this project (a laptop CPU), each search was limited to 15 minutes or 40 trials. LightGBM completed 40 "
      "trials at every site (about a third pruned after the first horizons), CatBoost, whose deep symmetric trees are "
      "slower to fit, only 8–27; the search spaces are listed in {ch:appA}."),

    H2("sec:lstm", "Phase 4: Seq2Seq LSTM with attention"),
    P("An LSTM cell [@hochreiter] updates a memory c_{t} through input, forget and output gates:"),
    EQ("eq:lstm", r"\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tanh(W_c[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c),\qquad \mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)"),
    P("Here f_{t}, i_{t} and o_{t} are sigmoid gates of [h_{t−1}, x_{t}]. The forecaster ({fig:arch}) has three parts. A "
      "**bidirectional LSTM encoder** [@schuster] with 64 units per direction reads the history of H_{s}, u_{10}, v_{10}, "
      "MSLP and T_{m} over 72 h at a 3-hourly step (25 steps); its final states initialise the decoder. An **LSTM "
      "decoder** runs over five steps, one per horizon, whose inputs are only quantities known at forecast time: the "
      "normalised lead time h/72 and the seasonal phase at t + h. **Additive attention** [@bahdanau] forms, for each "
      "decoder state s_{j}, a context vector over the encoder outputs e_{k}:"),
    EQ("eq:attn", r"\mathbf{c}_j = \sum_k \alpha_{jk}\,\mathbf{e}_k,\qquad \alpha_{jk} = \mathrm{softmax}_k\left(\mathbf{v}^\top \tanh(W_1\mathbf{s}_j + W_2\mathbf{e}_k)\right)"),
    P("A dense head then predicts the change from the value at the origin, Ĥ_{s}(t + h_{j}) = H_{s}(t) + Δ_{j}, so the "
      "untrained network starts from persistence. The decoder is non-autoregressive: it never receives its own "
      "predictions, so there is no error feedback between horizons and no exposure bias. Inputs are standardised with "
      "statistics of the fit block only. Training used Adam [@kingma] (learning rate 10^{−3}, gradient clipping at 1, "
      "batch 256), early stopping on the 2020 block with patience 6, and learning-rate halving on plateaus [@tensorflow]."),
    F("fig:arch", "fig_architecture.png", 6.0, "Architecture of the Seq2Seq Bi-LSTM forecaster with attention"),
    H3("sec:peakloss", "Peak-weighted loss"),
    P("Let e = y − ŷ (e > 0 is an under-prediction) and s(y) = σ((y − q)/τ_{s}) a soft indicator of an extreme sea state, "
      "with q the 95th percentile of the training targets and τ_{s} = 0.25 m. The loss is:"),
    EQ("eq:peak", r"\mathcal{L} = \frac{1}{H}\sum_{j=1}^{H}\left[1 + s(y_j)\left(\beta + \gamma\,\mathbf{1}[e_j > 0]\right)\right] e_j^2"),
    P("In ordinary seas (s ≈ 0) it reduces to the MSE; in extreme seas (s ≈ 1) every error weighs 1 + β = 2 times as much "
      "and an under-prediction 1 + β + γ = 6 times as much. The loss accepts some positive bias at the peaks in return for "
      "fewer missed extremes. An otherwise identical network trained with the MSE isolates the effect."),

    H2("sec:verification", "Phase 5: verification"),
    P("All models are scored on the same test origins. For observations o_{i} and forecasts p_{i} at one horizon:"),
    EQ("eq:metrics", r"\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_i (p_i - o_i)^2},\quad \mathrm{MAE} = \frac{1}{n}\sum_i |p_i - o_i|,\quad R^2 = 1 - \frac{\sum_i (p_i - o_i)^2}{\sum_i (o_i - \bar{o})^2},\quad \mathrm{SI} = \frac{\mathrm{RMSE}}{\bar{o}}"),
    P("The bias and the correlation coefficient are reported alongside these [@mentaschi], together with the skill "
      "relative to persistence:"),
    EQ("eq:skill", r"SS = 1 - \frac{\mathrm{RMSE}_{\mathrm{model}}}{\mathrm{RMSE}_{\mathrm{persistence}}}"),
    P("Scores are also computed on the extreme subset (o ≥ q) and within the windows of the named cyclones of 2023–2024. "
      "Differences in squared error between models are tested with the Diebold–Mariano statistic [@dm] with the "
      "small-sample correction of Harvey et al. [@hln]; because forecasts issued every hour overlap, the long-run variance "
      "uses a Newey–West estimator with h lags."),
]
