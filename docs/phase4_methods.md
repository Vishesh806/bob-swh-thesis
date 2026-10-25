# Phase 4 methods notes (draft material for the Methodology chapter)

## 1 Problem formulation

Given the multivariate history **X**ₜ = [**x**ₜ₋L, …, **x**ₜ] ∈ ℝ^((L/s+1)×F) and the known-future
covariates **D**ₜ = [**d**ₜ₊ₕ₁, …, **d**ₜ₊ₕ_H], the network outputs all horizons jointly,

  [Ĥs(t+6), Ĥs(t+12), Ĥs(t+24), Ĥs(t+48), Ĥs(t+72)] = g_θ(**X**ₜ, **D**ₜ, Hs(t)).

The input set is chosen per experiment:

* *Core*: Hs, u₁₀, v₁₀, MSLP and mean wave period.
* *Extended*: core plus wind stress and its 24 h mean, pressure tendencies and anomaly, SST and
  the air–sea temperature difference, mean wave direction (sin, cos), wind–wave alignment and
  the seasonal phase.

Inputs are standardised with means and standard deviations estimated on the fit block alone.
Targets remain in metres.

## 2 Architecture

**Encoder.** A bidirectional LSTM reads the history. The final forward and backward hidden and
cell states are concatenated, hₑ = [h→; h←] and cₑ = [c→; c←], and initialise the decoder. The
full sequence of encoder outputs {**e**ₖ} is kept for attention.

**Decoder.** An LSTM with 2U units runs over H = 5 steps, one per horizon, starting from
(hₑ, cₑ). Its input at step j is **d**ⱼ = [hⱼ / 72, sin φ(t+hⱼ), cos φ(t+hⱼ)], with φ the day-of-year
phase. These are the only quantities known at forecast time; no reanalysis value after t enters the
model. The horizons are irregularly spaced, so the lead time is given to the decoder explicitly.

**Attention.** For each decoder state **s**ⱼ, additive (Bahdanau) attention forms a context vector
**c**ⱼ = Σₖ αⱼₖ **e**ₖ, with αⱼₖ = softmax_k(**v**ᵀ tanh(W₁**s**ⱼ + W₂**e**ₖ)). Each horizon can then
weight the part of the history most relevant to it.

**Head.** A dense layer (ReLU) and a linear output per step. With the residual connection,
Ĥs(t+hⱼ) = Hs(t) + Δⱼ, so the untrained network starts from persistence and learns corrections
to it. Predictions are clipped at 0 m.

The decoder is **non-autoregressive**: it never receives its own predictions, so there is no
error feedback between horizons, no exposure bias from teacher forcing, and training and
inference run the same computation. This satisfies the Direct Multi-Step requirement within one
jointly trained network.

## 3 Peak-weighted loss

Let e = y − ŷ, where e > 0 is an under-prediction, and let s(y) = σ((y − q)/τ) be a soft indicator
of an extreme sea state. q is the 95th percentile of the fit-block targets and τ = 0.25 m. Then

  ℒ = (1/H) Σⱼ [1 + s(yⱼ)(β + γ·𝟙{eⱼ > 0})] eⱼ².

* Ordinary sea states (s ≈ 0): ℒ reduces to the MSE.
* Extreme sea states (s ≈ 1): over-prediction is weighted 1 + β, under-prediction 1 + β + γ.
  The defaults β = 1, γ = 4 give weights 2 and 6.

The sigmoid makes the weight a smooth function of the observed Hs. ℒ stays continuous in e,
because the asymmetric factor switches only at e = 0, where e² = 0. Under-forecasting a cyclone
peak is the operationally costly error, so the loss deliberately trades a small positive bias at
the peaks for fewer missed extremes. To isolate this effect, an otherwise identical network is
trained with the plain MSE (ablation). The two are compared on the extreme subset (obs ≥ q) by
RMSE, bias and under-prediction rate, and on all samples by RMSE.

## 4 Training

| Setting | Value |
|---|---|
| Optimiser | Adam, learning rate 10⁻³, gradient clipping ‖g‖ ≤ 1 |
| Batch size | 256, shuffled within the fit block |
| Early stopping | On the es block (2020), patience 10 epochs, best weights restored |
| Learning-rate schedule | Halved after 3 epochs without improvement, floor 10⁻⁵ |
| Regularisation | Dropout 0.1 in the LSTMs and the head |
| Seed | Fixed (Python, NumPy, TensorFlow) |

Training, early stopping, model selection and testing use the same fit / es / val / test blocks
as Phase 3, under the same containment rule, so every model is assessed on identical periods.
