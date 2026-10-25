# Phase 3 methods notes (draft material for the Methodology chapter)

## 1 Reference forecasts

| Model | Forecast of Hs(t + h) | Role |
|---|---|---|
| Persistence | Hs(t) | Skill floor at 6–12 h, where hourly Hs is strongly autocorrelated |
| Climatology | Mean Hs on the calendar day of t + h over the training years, smoothed ±15 days | Skill floor at 48–72 h; carries the monsoon cycle |
| SARIMA(p, d, q) | Kalman-filter forecast from the origin, parameters frozen from training | Classical statistical benchmark |

**SARIMA estimation.** Parameters are estimated by maximum likelihood on the last two years of the
training block. Higher-order candidates are compared by AIC (`--sarimax-select`). At evaluation,
the state-space form

  yₜ = d + Z aₜ,  aₜ₊₁ = c + T aₜ + R ηₜ

is filtered through the complete record with the parameters held fixed. The filtered state
a_{t|t} depends on observations up to t only, and the h-step forecast follows from
a_{t+h|t} = c + T a_{t+h−1|t}, ŷ_{t+h|t} = d + Z a_{t+h|t}. This reproduces the statsmodels forecast
from each individual origin to machine precision, which a test checks. It needs a single filter
pass, so all ~17 000 test origins can be scored. Missing observations are handled by the filter,
which propagates the state without an update.

**Why the baseline has no exogenous regressors.** A SARIMAX with ERA5 forcing would require the
forcing at t + h when forecasting from t. That forcing is unknown at forecast time unless an NWP
forecast is available. Supplying reanalysis values would turn the baseline into a
perfect-prognosis experiment and favour it unfairly.

## 2 Direct Multi-Step gradient boosting

For each horizon h ∈ {6, 12, 24, 48, 72} h an independent regressor is trained:

  Ĥs(t + h) = f_h(**x**ₜ),  f_h = Σₘ₌₁^{M_h} η · g_{h,m}(**x**ₜ),

where the g_{h,m} are regression trees fitted to the negative gradient of the squared error. The
input vector **x**ₜ holds lags of Hs, ERA5 Hs, wind stress, wind speed and MSLP at
k ∈ {0, 1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 72} h, together with the remaining Phase 2 features at t.
No forecast is used as an input, so errors do not propagate between horizons. Each f_h selects
its own number of boosting rounds M_h by early stopping.

Three implementations are compared under the same interface:

* LightGBM: leaf-wise growth with histogram-binned features.
* CatBoost: symmetric (oblivious) trees with ordered boosting, which reduces the prediction shift
  of classical gradient boosting.
* XGBoost: depth-wise growth with a second-order approximation of the loss.

Forecasts are clipped at 0 m.

## 3 Sample blocks and hyperparameter optimisation

| Block | Period | Used for |
|---|---|---|
| fit | 2015 – 2019 | Training the boosters |
| es | 2020 | Early stopping (choice of M_h) |
| val | 2021 – 2022 | Optuna objective (choice of hyperparameters) |
| test | 2023 – 2024 | Reported results only |

Every block obeys the containment rule of Phase 2 (t − L ≥ start, t + 72 h ≤ end). Early
stopping and hyperparameter selection use different blocks, so the validation score is optimised
only once and remains an unbiased estimate for model selection.

The search uses the Tree-structured Parzen Estimator (TPE) with a median pruner. The objective is
the pooled multi-horizon RMSE on the validation block,

  J(θ) = √( (1/H) Σ_h RMSE_h(θ)² ),

with one parameter vector θ shared by all horizons. Horizons are fitted in order, and the running
value of J is reported after each one, so an unpromising trial is stopped early. In the
alternative per-horizon mode, a separate study minimises each RMSE_h. During the search, every
third training origin is used. Adjacent hourly origins are almost collinear, so this costs little
information. The final models use all origins.

Search spaces (log-uniform where marked *):

| LightGBM | CatBoost | XGBoost |
|---|---|---|
| learning rate 0.01–0.2* | learning rate 0.01–0.2* | learning rate 0.01–0.2* |
| num_leaves 15–255* | depth 4–10 | max_depth 3–10 |
| min_child_samples 10–300* | min_data_in_leaf 1–200* | min_child_weight 1–100* |
| subsample 0.5–1, colsample 0.4–1 | subsample 0.5–1 (Bernoulli) | subsample 0.5–1, colsample 0.4–1 |
| L1, L2 1e−8–10* | l2_leaf_reg 1–30*, random_strength 0.1–10* | L1, L2 1e−8–10* |

## 4 Verification

All models are scored on the same forecast origins with RMSE, MAE, bias, R², the scatter index
SI = RMSE / ō, its bias-removed form, and the correlation coefficient, for each horizon
(`bobswh/metrics.py`). Phase 5 adds skill scores relative to persistence and climatology, scores
stratified by cyclone and non-cyclone conditions, and Diebold–Mariano significance tests.
