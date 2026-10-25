# Phase 1–2 methods notes (draft material for the Methodology chapter)

## 1 Data sources

**ERA5.** Hourly single-level reanalysis from ECMWF, 2015–2024 (120 months), over 4–23° N, 79–100° E.
Atmospheric fields (u10, v10, msl, sst, t2m) are on a 0.25° grid. Wave fields from the coupled
WAM model (swh, mwd, mwp, pp1d, shww, shts) are on a 0.5° grid. ERA5 wave heights assimilate
satellite altimeter data, but they remain model output, so the INCOIS buoys are the reference truth.

**INCOIS OMNI buoys.** Moored observations of Hs, wave period and direction, wind, pressure and
SST at BD08, BD10 and BD11. All timestamps are converted to UTC.

## 2 Quality control

1. *Range check.* Values outside the admissible interval (e.g. 0 ≤ Hs ≤ 20 m, 870 ≤ p ≤ 1060 hPa)
   are rejected. Only negative sentinel codes (−999, −99.9, …) are removed as codes, because
   999 hPa is a physically valid pressure inside a tropical cyclone.
2. *Regularisation.* Records are averaged into right-closed bins, (t − Δt, t] ↦ t, so a value
   labelled t contains no later information. Directions are averaged as unit vectors:
   θ̄ = atan2( Σ sin θᵢ , Σ cos θᵢ ).
3. *Isolated-spike test.* A sample xₜ is rejected if
   |xₜ − xₜ₋₁| > J and |xₜ − xₜ₊₁| > J with the same sign, while |xₜ₋₁ − xₜ₊₁| ≤ τ
   (J = 1.5 m, τ = 0.5 m for hourly Hs). A storm peak lasting two or more samples fails the last
   condition and is kept. A standard deviation or MAD filter would instead clip real cyclone peaks.
4. *Gap filling.* NaN runs of at most G = 3 h are filled. Longer runs are left missing and later
   remove every sample that touches them. Forward fill is the default because it is causal.
   Linear, PCHIP and local cubic interpolation are available, but they use the first observation
   after the gap and so leak up to G hours of future information into the inputs. Imputed values
   are flagged, and imputed targets are excluded from training and scoring.

## 3 Spatial alignment

Grid nodes and buoy positions are mapped to the unit sphere,
**r** = (cos φ cos λ, cos φ sin λ, sin φ), and indexed with a KD-tree. The chord distance
c = 2 sin(d / 2R) increases strictly with the great-circle distance

d = 2R arcsin √( sin²(Δφ/2) + cos φ₁ cos φ₂ sin²(Δλ/2) ),  R = 6371.009 km,

so the KD-tree nearest neighbour is exactly the haversine nearest neighbour, at O(log N) cost
per query. Only *valid* nodes enter the tree: nodes where any field is undefined (land in the WAM
grid, land in SST) are excluded, so that a coastal buoy is never matched to a land node. Each
grid (0.25° atmosphere, 0.5° waves) is matched separately. The default is the single nearest
node. An optional inverse-distance weighting over k nodes uses wᵢ = dᵢ⁻ᵖ / Σⱼ dⱼ⁻ᵖ, with
directions averaged as unit vectors.

## 4 Feature engineering

| Feature | Definition | Physical rationale |
|---|---|---|
| Wind speed U₁₀ | √(u₁₀² + v₁₀²) | Primary forcing of wind-sea growth |
| Wind direction θ_w | atan2(−u, −v) mod 360°, "from" convention | Same convention as ERA5 mwd and buoy records |
| Air density ρₐ | p / (R_d T), R_d = 287.05 J kg⁻¹ K⁻¹ | Momentum flux scales with ρₐ, which is about 4 % lower in warm tropical air than at 15 °C |
| Drag coefficient C_d | Large & Pond (1981): 1.2×10⁻³ for U < 11 m s⁻¹; (0.49 + 0.065U)×10⁻³ for 11–25 m s⁻¹, held constant above | Above ~30 m s⁻¹ C_d saturates (Powell et al. 2003); unbounded linear growth would overstate cyclone stress |
| Wind stress **τ** | ρₐ C_d \|U\| (u, v), and \|τ\| | Momentum flux into the wave field; the energy input grows roughly as U³, which raw U does not represent linearly |
| Stress history | mean \|τ\| over (t−w, t], w = 6, 12, 24 h; max U₁₀ over the same windows | Waves are duration-limited: Hs depends on how long the wind has blown, not only its current speed |
| Pressure tendency | p(t) − p(t−h), h = 3, 12, 24 h | Rapid pressure fall marks an approaching depression or cyclone before the waves arrive |
| Pressure anomaly | p(t) − mean of p over the past 30 days | Separates synoptic depressions from the seasonal (monsoon) pressure cycle |
| Air–sea ΔT | T₂ₘ − SST | Unstable stratification (ΔT < 0) increases the transfer of momentum to the waves; SST above ~26.5 °C also favours cyclogenesis |
| Direction encodings | (sin θ, cos θ) for mwd, wind and buoy directions | Removes the 359°→0° discontinuity; raw degrees are excluded from the model inputs |
| Wind–wave alignment | cos(θ_w − θ_mwd) | ≈ +1 for a locally generated wind sea, ≤ 0 for swell crossing or opposing the wind, a frequent regime in the bay outside the monsoon |
| Wave steepness | Hs / L₀, L₀ = gT²/2π | Distinguishes young, steep wind sea from long-period swell |
| Hs tendency | Hs(t) − Hs(t−h), h = 3, 6, 12 h | Growth or decay phase of the sea state |
| Seasonal phase | sin, cos of 2π·doy/365.25 | Southwest / northeast monsoon cycle |

Every feature at time t depends only on data at times ≤ t (backward differences, trailing
windows). A test checks this directly: changing future values leaves all earlier features unchanged.

## 5 Supervised samples (Direct Multi-Step)

For a forecast origin t, lookback L and horizons H = {6, 12, 24, 48, 72} h:

**X**ₜ = [**x**ₜ₋L, **x**ₜ₋L₊ₛ, …, **x**ₜ] ∈ ℝ^((L/s+1)×F),  **Y**ₜ = [Hs(t+6), Hs(t+12), Hs(t+24), Hs(t+48), Hs(t+72)].

Each horizon is a target of its own. No prediction is fed back as an input, so errors do not
accumulate as they do in a recursive strategy. The tree models receive a flattened layout,
Hs(t−k) for k ∈ {0, 1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 72} h plus the current exogenous features.
Both layouts keep exactly the same origins: a sample is retained only if every input over
[t − L, t] and every target is observed.

## 6 Chronological split without leakage

A sample belongs to block [Tₛ, Tₑ] only if t − L ≥ Tₛ and t + max H ≤ Tₑ. The second condition
removes the usual multi-step leak, in which the final training samples have targets that lie inside
the validation period. The first keeps the blocks fully independent. Blocks: train 2015–2020
(Fani, Amphan; the final year, 2020, serves as the early-stopping block), validation 2021–2022
(Yaas, Sitrang), test 2023–2024 (Mocha, Michaung, Remal).
Scalers and any other fitted transformations are estimated on the training block alone (Phase 3).

## References

- Large, W. G. and Pond, S. (1981). *J. Phys. Oceanogr.*, 11, 324–336.
- Wu, J. (1982). *J. Geophys. Res.*, 87(C12), 9704–9706.
- Powell, M. D., Vickery, P. J. and Reinhold, T. A. (2003). *Nature*, 422, 279–283.
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Q. J. R. Meteorol. Soc.*, 146, 1999–2049.
