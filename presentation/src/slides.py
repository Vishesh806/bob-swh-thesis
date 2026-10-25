"""The thesis presentation: 21 slides. Every number is taken from report/src/results_text.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "report" / "src"))
from deckkit import *  # noqa: E402,F403
import re as _re  # noqa: E402
from results_text import (BEST, BEST_ML, LABEL, BUOYS, biasp, cost, f2, f3, gain_arima, gain_pers, has_lstm,  # noqa: E402
                          mean_rmse, n_test, storm_mean, under, zoom)

FIG = Path("/Users/visheshagarwal/Desktop/bob_swh_thesis/figures")
LOGO = Path("/Users/visheshagarwal/Desktop/bob_swh_thesis/assets/iitkgp_logo.png")
STUDENT, ROLL, SUPERVISOR = "Vishesh Bhagwandas Agarwal", "Roll No. 22NA30045", "Professor Swapnadip De Chowdhury"
N_SLIDES = 21


def logo(x=0.3, y=0.25, h=0.95, card=False):
    """IIT Kharagpur logo, top-left. On dark slides it sits on a white card so the blue stays legible."""
    els = []
    if card:
        els.append(rect(x - 0.08, y - 0.06, h * 250 / 281 + 0.16, h + 0.12, WHITE, name="logo card"))
    return els + [image(LOGO, x, y, h=h, name="IIT Kharagpur logo")]


NBSP = "\u00a0"


def glue(t):
    """Keep a number and its unit on one line: '24 h' -> '24\u00a0h'."""
    return _re.sub(r"(\d) (?=(h|m|km|hPa|m/s|s|°C|%)\b)", "\\1" + NBSP, t)


def frame(n, title, message=None, source=None):
    els = logo() + [text(1.35, 0.32, 11.35, 0.75, title, size=30, color=NAVY, face=HEAD, bold=True, name="title")]
    if message:
        els.append(text(1.35, 1.05, 11.35, 0.5, message, size=16, color=GREY, name="key message"))
    els.append(text(11.9, 6.95, 0.9, 0.35, f"{n} / {N_SLIDES}", size=11, color=GREY, align="r", name="slide number"))
    if source:
        els.append(text(0.6, 6.95, 10.8, 0.35, source, size=10, color=GREY, name="source"))
    return els


def stat(x, y, w, big, label, color=TEAL, h=1.55, big_size=40, fill=TINT):
    return [rect(x, y, w, h, fill, name="stat card"),
            text(x + 0.15, y + 0.08, w - 0.3, 0.92, big, size=big_size, color=color, face=BODY, bold=True, name="stat value"),
            text(x + 0.15, y + 0.92, w - 0.3, h - 0.97, glue(label), size=13, color=INK, name="stat label")]


def card(x, y, w, h, num, head, body, fill=TINT, body_size=14):
    return [rect(x, y, w, h, fill, name="card"), circle(x + 0.2, y + 0.22, 0.5, TEAL, str(num), size=15),
            text(x + 0.85, y + 0.17, w - 1.0, 0.55, head, size=16, bold=True, color=NAVY, name="card heading"),
            text(x + 0.85, y + 0.68, w - 1.0, h - 0.8, body, size=body_size, color=INK, name="card body")]


TEST_SRC = "Test period 2023–2024, unseen in training, early stopping and tuning; target ERA5 Hs at the buoy positions"
SLIDES = []

# 1 ---------------------------------------------------------------- title
SLIDES.append(dict(bg=NAVY, title_text="Multi-step significant wave height forecasting in the Bay of Bengal", els=[
    *logo(0.5, 0.25, 0.85, card=True),
    text(0.8, 1.2, 11.7, 0.45, "B.TECH–M.TECH DUAL DEGREE PROJECT · OCEAN ENGINEERING AND NAVAL ARCHITECTURE", size=13, color="9FC9CE", bold=True, name="kicker"),
    text(0.8, 1.7, 11.7, 2.25, "Multi-Step Significant Wave Height Forecasting in the Bay of Bengal", size=40, color=WHITE, face=HEAD, bold=True, name="title"),
    text(0.8, 3.95, 11.7, 0.6, "Hybrid machine-learning and deep-learning frameworks, 6 to 72 hours ahead", size=20, color="CFE6E8", italic=True, name="subtitle"),
    rect(0.8, 4.95, 5.6, 1.6, "13355C", name="presenter card"),
    text(1.0, 5.08, 5.3, 1.4, [[(STUDENT, {"bold": True, "size": 18})], ROLL, f"Supervisor: {SUPERVISOR}"], size=15, color=WHITE, spacing=4, name="presenter"),
    text(7.0, 5.08, 5.6, 1.4, ["Department of Ocean Engineering and Naval Architecture", "Indian Institute of Technology Kharagpur", "10 November 2026"], size=15, color="CFE6E8", spacing=4, name="institute"),
], notes="One sentence: a data-driven model that forecasts wave height up to three days ahead at buoy locations in the Bay of Bengal, built and tested so that no future information leaks into training."))

# 2 ---------------------------------------------------------------- motivation
SLIDES.append(dict(els=frame(2, "Cyclones lift a 1.5 m sea to 10 m within a day",
    "The Bay of Bengal needs fast, reliable wave forecasts at fixed points: ports, platforms, moorings",
    "ERA5 Hs at the three OMNI buoy positions, 2015–2024 (Hersbach et al. 2020); cyclone identification from IMD reports") + [
    image(FIG / "fig_record.png", 0.6, 1.65, w=7.6, h=5.2),
    *stat(8.6, 1.75, 4.1, "10.7 m", "Peak Hs at BD10 in cyclone Amphan, May 2020", color=AMBER),
    *stat(8.6, 3.45, 4.1, "≈ 1.5 m", "Mean Hs at all three sites"),
    text(8.6, 5.15, 4.1, 1.7, ["Numerical models (WAM, WAVEWATCH III) need supercomputers and NWP forcing",
                               "A point forecaster can be cheap, fast and retrained as data arrive"], size=14, bullets=True, name="takeaways"),
], notes="Point to the spikes: each one is a named cyclone (Amphan, Yaas, Fani, Mocha). Between them the monsoon cycle dominates. The engineering need is a forecast at a fixed location, which is exactly what a data-driven model can give cheaply."))

# 3 ---------------------------------------------------------------- objectives
SLIDES.append(dict(els=frame(3, "Objectives and contributions",
    "Forecast Hs at 6, 12, 24, 48 and 72 h, without leakage, and without feeding forecasts back") + [
    rect(0.6, 1.75, 5.2, 5.05, TINT, name="objectives card"),
    text(0.85, 1.9, 4.8, 0.5, "Objectives", size=19, bold=True, color=NAVY, face=HEAD, name="objectives heading"),
    text(0.85, 2.45, 4.8, 4.25, ["Leakage-free pipeline: ERA5 + INCOIS buoys, aligned and physically featured",
                                 "Direct multi-step LightGBM and CatBoost, Bayesian-tuned",
                                 "Seq2Seq Bi-LSTM with attention and a peak-weighted loss",
                                 "Fair comparison with persistence, climatology and ARIMA, including cyclones"], size=15, bullets=True, spacing=9, name="objectives"),
    text(6.2, 1.8, 6.5, 0.5, "My contributions", size=19, bold=True, color=NAVY, face=HEAD, name="contrib heading"),
    *[e for i, (h, b) in enumerate([
        ("Containment rule for splits", "Inputs and all targets of a sample inside one block"),
        ("Physics features", "Wind stress, pressure tendency, air–sea ΔT, sin/cos directions"),
        ("Fast all-origin ARIMA", "One Kalman pass gives the statsmodels forecast at every origin"),
        ("Peak-weighted loss", "Under-predicting a storm costs 6× an ordinary error"),
        ("Tested, reproducible code", "Automated tests for every guardrail; all tables regenerate")]) for e in [
        circle(6.2, 2.4 + i * 0.88, 0.48, TEAL, str(i + 1), size=14),
        text(6.85, 2.31 + i * 0.88, 5.9, 0.43, h, size=15, bold=True, color=INK, name="contribution"),
        text(6.85, 2.68 + i * 0.88, 5.9, 0.36, b, size=13, color=GREY, name="contribution detail")]],
], notes="These five items are what I designed and built myself, beyond applying existing libraries."))

# 4 ---------------------------------------------------------------- data
SLIDES.append(dict(els=frame(4, "Data: ten years of hourly ERA5 at three buoy sites",
    "87,672 hours per site, no gaps; one CDS time-series request per buoy instead of 60 gridded requests",
    "ERA5 (ECMWF / Copernicus C3S, CC-BY-4.0); OMNI buoys BD08, BD10, BD11 (INCOIS)") + [
    *stat(0.6, 1.8, 2.85, "87,672", "Hours per site, 2015–2024"),
    *stat(3.6, 1.8, 2.85, "8", "ERA5 variables per site", big_size=40),
    *stat(0.6, 3.55, 2.85, "8–25 km", "Buoy to matched ERA5 node (KD-tree on the sphere)", big_size=32),
    *stat(3.6, 3.55, 2.85, "0", "Missing hours after quality control"),
    text(0.6, 5.3, 5.85, 1.5, ["Buoy reader, spike test and gap rules are built and tested; OMNI data were not yet received",
                               "So the target here is ERA5 Hs at the buoy positions"], size=14, bullets=True, name="data notes"),
    image(FIG / "fig_season.png", 6.9, 1.8, w=5.8, h=2.8),
    table(6.9, 4.85, [1.0, 1.8, 2.95], [
        ["Block", "Years", "Use"], ["Fit", "2015–2019", "Training"], ["Early stop", "2020", "Rounds / epochs"],
        ["Validation", "2021–2022", "Hyperparameters"], ["Test", "2023–2024", "Reported results only"]], size=12, row_h=0.36),
], notes="Explain why ERA5: complete, homogeneous, and the CDS time-series service delivered ten years per site in minutes. Seasonal cycle: monsoon peak in July at the northern sites, relatively higher winter seas at BD11. Be upfront that the buoy data had not arrived, so ERA5 is the target."))

# 5 ---------------------------------------------------------------- methodology
SLIDES.append(dict(els=frame(5, "Methodology: five phases, one guardrail",
    "Every input at or before the forecast origin; every block holds all inputs and targets of its samples") + [
    image(FIG / "fig_flowchart.png", 3.3, 1.6, w=6.7, h=5.3),
], notes="Walk top to bottom. The guardrail at the bottom is the thread through all phases: it is what makes the test scores honest."))

# 6 ---------------------------------------------------------------- leakage
SLIDES.append(dict(els=frame(6, "Zero look-ahead, by construction and by test",
    "The common multi-step leak: the last training samples have targets inside the validation period") + [
    *card(0.6, 1.75, 5.9, 1.55, 1, "Containment rule", "A sample with origin t belongs to block [Ts, Te] only if t − 72 h ≥ Ts and t + 72 h ≤ Te"),
    *card(6.8, 1.75, 5.9, 1.55, 2, "Causal features", "Backward differences and trailing windows only; a test changes the future and checks the past is unchanged"),
    *card(0.6, 3.5, 5.9, 1.55, 3, "Causal gap filling", "Forward fill only, because interpolation would read past the gap. Filled values are never targets"),
    *card(6.8, 3.5, 5.9, 1.55, 4, "Train-only statistics", "Scalers, peak threshold and climatology are computed on the fit block alone"),
    rect(0.6, 5.25, 12.1, 1.5, NAVY, name="split banner"),
    text(0.9, 5.35, 3.2, 1.3, "3 blocks", size=40, color=AMBER, face=BODY, bold=True, valign="m", name="split stat"),
    text(4.0, 5.35, 8.5, 1.3, "Early stopping (2020), hyperparameter selection (2021–22) and reporting (2023–24) never share data, so the validation score is optimised only once and the test score is untouched.", size=15, color=WHITE, valign="m", name="split text"),
], notes="This slide answers the first question an examiner should ask: how do you know you did not leak the future? The answer is a rule, enforced in code and checked by tests."))

# 7 ---------------------------------------------------------------- features
SLIDES.append(dict(els=frame(7, "Features encode the physics of wave generation",
    "Wind stress, duration, pressure falls, stability and direction, all from data at or before t",
    "Drag coefficient after Large and Pond (1981), held constant above 25 m/s (Powell et al. 2003)") + [
    table(0.6, 1.75, [2.6, 4.2, 5.3], [
        ["Feature", "Definition", "Why it matters"],
        ["Wind stress τ", "ρa Cd(U) |U| (u, v)", "Momentum input to the sea; grows ≈ U²"],
        ["Stress history", "mean τ, max U over 6, 12, 24 h", "Waves are duration-limited"],
        ["Pressure tendency", "p(t) − p(t − h), h = 3, 12, 24 h", "Falling pressure warns of a depression"],
        ["Pressure anomaly", "p − 30-day trailing mean", "Synoptic signal without the monsoon"],
        ["Air–sea ΔT", "T2m − SST", "Unstable air transfers more momentum"],
        ["Directions", "sin θ, cos θ (waves and wind)", "Removes the 359° → 0° jump"],
        ["Wind–wave alignment", "cos(θwind − θwave)", "Wind sea (+1) vs crossing swell (≤ 0)"],
        ["Steepness", "Hs / L0, L0 = gT²/2π", "Young wind sea vs long swell"]], size=13, row_h=0.52),
], notes="Pick two to explain: the stress history (a 20 m/s wind that has blown for 24 hours makes bigger waves than one that started an hour ago) and the pressure tendency (the earliest local sign of a cyclone)."))

# 8 ---------------------------------------------------------------- strategy
SLIDES.append(dict(els=frame(8, "Direct multi-step: one model per lead time",
    "Recursive one-step models feed their own errors back; direct models never do",
    "Ben Taieb et al., Expert Systems with Applications 39 (2012)") + [
    *card(0.6, 1.75, 5.9, 2.2, 1, "Samples", "Inputs: the last 72 h of Hs and forcing. Targets: Hs(t + 6), (t + 12), (t + 24), (t + 48), (t + 72)"),
    *card(6.8, 1.75, 5.9, 2.2, 2, "Trees: one model per horizon", "78 lagged and current features; LightGBM and CatBoost; boosting rounds set by early stopping per horizon"),
    *card(0.6, 4.15, 5.9, 2.2, 3, "Bayesian tuning", "Optuna TPE with median pruning; the objective is the RMSE pooled over the five horizons on 2021–22"),
    *card(6.8, 4.15, 5.9, 2.2, 4, "Baselines", "Persistence, day-of-year climatology, and ARIMA(2,1,1) with frozen training parameters, forecast from every origin"),
], notes="The trees and the LSTM see exactly the same forecast origins, so every comparison later is like for like."))

# 9 ---------------------------------------------------------------- architecture
SLIDES.append(dict(els=frame(9, "Seq2Seq Bi-LSTM with attention",
    "The decoder sees only what is known at forecast time: the lead time and the season") + [
    image(FIG / "fig_architecture.png", 0.6, 1.7, w=12.1, h=4.2),
    text(0.6, 6.0, 12.1, 0.8, ["Non-autoregressive decoder: all five horizons in one pass, no prediction fed back, no exposure bias",
                               "Residual head: forecast = Hs(t) + Δ, so the untrained network starts from persistence"], size=14, bullets=True, name="arch notes"),
], notes="Explain attention in one sentence: each horizon can look back at the part of the 72-hour history that matters most for it."))

# 10 ---------------------------------------------------------------- loss
SLIDES.append(dict(els=frame(10, "A loss that refuses to miss storms",
    "Ordinary seas: plain squared error. Extreme seas: errors ×2, under-predictions ×6") + [
    rect(0.6, 1.75, 12.1, 1.5, TINT, name="equation card"),
    text(0.9, 1.9, 11.5, 1.2, [[("L = mean over horizons of [1 + s(y) · (β + γ · 1{y > ŷ})] · (y − ŷ)²", {"bold": True, "size": 22})],
                               "where s(y) = sigmoid((y − q95) / 0.25 m), β = 1 and γ = 4"], size=16, color=NAVY, align="c", spacing=6, name="equation"),
    *card(0.6, 3.5, 3.9, 2.9, 1, "Why", "Missing a cyclone peak is far costlier than a false alarm for ports and shipping"),
    *card(4.7, 3.5, 3.9, 2.9, 2, "How it is smooth", "The sigmoid makes the weight continuous in y, and the loss stays continuous at y = ŷ"),
    *card(8.8, 3.5, 3.9, 2.9, 3, "How it is tested", "An identical network trained with MSE isolates the effect of the loss"),
], notes="q95 is the 95th percentile of training Hs: 3.0 m at BD08, 3.1 m at BD10 and 2.6 m at BD11."))

# 11 ---------------------------------------------------------------- results overview
gain24, gain72 = gain_pers[24], gain_pers[72]
SLIDES.append(dict(els=frame(11, f"Trees beat persistence: −{gain24:.0f}% at 24 h, −{gain72:.0f}% at 72 h",
    "RMSE grows with lead time for every model; the gap to the baselines widens with it", TEST_SRC) + [
    image(FIG / "fig_rmse_horizon.png", 0.6, 1.7, w=8.6, h=3.6),
    *stat(9.55, 1.8, 3.15, f"{f2(mean_rmse.loc[BEST[24], 24])} m", f"Best RMSE at 24 h ({LABEL[BEST[24]]}), mean of 3 sites", h=1.7),
    *stat(9.55, 3.7, 3.15, f"{f2(mean_rmse.loc[BEST[72], 72])} m", f"Best RMSE at 72 h ({LABEL[BEST[72]]})", color=AMBER, h=1.7),
    text(0.6, 5.5, 12.1, 1.3, [f"Persistence: {f2(mean_rmse.loc['persistence', 24])} m at 24 h, {f2(mean_rmse.loc['persistence', 72])} m at 72 h"
                               + (f"; ARIMA improves on it only at short lead times" if gain_arima else ""),
                               f"All models scored on the same {min(n_test.values()):,}+ forecast origins per site"], size=14, bullets=True, name="results notes"),
], notes="Read the chart left to right: at 6 hours everyone is close to persistence because the sea state barely changes; by 72 hours the learned models are clearly ahead."))

# 12 ---------------------------------------------------------------- bars
SLIDES.append(dict(els=frame(12, "ARIMA, trees and LSTM at 24 h and 72 h",
    "Gradient boosting and the Seq2Seq LSTM are close; both are well ahead of ARIMA", TEST_SRC) + [
    image(FIG / "fig_rmse_bars.png", 0.6, 1.7, w=12.1, h=4.6),
], notes="Same colours throughout the talk: blue LightGBM, orange CatBoost, green LSTM with the peak loss, yellow LSTM with MSE, pink ARIMA."))

# 13 ---------------------------------------------------------------- skill
SLIDES.append(dict(els=frame(13, "Skill rises with lead time",
    "SS = 1 − RMSE / RMSE(persistence): the further ahead, the more the forcing adds", TEST_SRC) + [
    image(FIG / "fig_skill_horizon.png", 0.6, 1.7, w=12.1, h=4.7),
], notes="Positive skill means better than persistence. Skill near zero at 6 hours is expected; it is the long horizons where a forecaster earns its keep."))

# 14 ---------------------------------------------------------------- timeline
SLIDES.append(dict(els=frame(14, "A month of forecasts against ERA5",
    "At 24 h the models track each event; at 72 h they keep the level but damp short events", TEST_SRC) + [
    image(FIG / "fig_timeline_BD08.png", 0.9, 1.65, w=11.5, h=5.2),
], notes="Window chosen automatically as the most variable 30 days of the test period at BD08."))

# 15 ---------------------------------------------------------------- cyclone
ev, day, peak = zoom["BD08"]
SLIDES.append(dict(els=frame(15, f"Cyclone {ev}: peak {peak:.1f} m at BD08",
    "24 h ahead the rise and timing are captured; 72 h ahead no model sees the storm coming", TEST_SRC) + [
    image(FIG / "fig_cyclone_BD08.png", 0.9, 1.65, w=11.5, h=5.2),
], notes="Three days before the peak the cyclone is far from the buoy and leaves almost no trace in the local wind, pressure and wave history. This is the clearest argument for adding forecast forcing."))

# 16 ---------------------------------------------------------------- peak loss
if has_lstm:
    peak_msg = (f"Under-predicted extremes at 24 h: {100 * under[24][0]:.0f}% → {100 * under[24][1]:.0f}%; "
                f"overall RMSE {cost[24]:+.0f}%")
else:
    peak_msg = "LSTM results pending"
SLIDES.append(dict(els=frame(16, "The peak-weighted loss misses fewer storms", peak_msg,
    "Extreme subset: observed Hs ≥ training 95th percentile, all sites pooled") + [
    image(FIG / "fig_peak_bias.png", 0.6, 1.7, w=8.4, h=3.6),
    *stat(9.4, 1.8, 3.3, f"{100 * under[24][1]:.0f}%" if has_lstm else "–", "of extremes under-predicted at 24 h with the peak loss", h=1.7),
    *stat(9.4, 3.7, 3.3, f"{100 * under[24][0]:.0f}%" if has_lstm else "–", "with the MSE loss", color=AMBER, h=1.7),
    text(0.6, 5.5, 12.1, 1.3, ["Lowest RMSE of all models on extreme seas, and mean error within ±0.07 m inside cyclone windows",
                               "Trade-off by design: higher overall RMSE, fewer missed peaks; choose the loss by the application"],
         size=14, bullets=True, name="peak notes"),
], notes="This is the key result for safety applications. The overall RMSE gets a little worse because the loss tolerates over-prediction near the threshold."))

# 17 ---------------------------------------------------------------- importance
SLIDES.append(dict(els=frame(17, "What the trees learned is physically sensible",
    "6 h: the last three hours of Hs. 72 h: season, SST and pressure, i.e. a state-aware climatology") + [
    image(FIG / "fig_importance.png", 0.6, 1.7, w=7.4, h=5.1),
    rect(8.4, 1.75, 4.3, 5.05, TINT, name="physics card"),
    text(8.65, 1.9, 3.9, 0.5, "Reading the chart", size=18, bold=True, color=NAVY, face=HEAD, name="physics heading"),
    text(8.65, 2.45, 3.9, 4.3, ["Short range = persistence plus a growth or decay correction",
                                "Long range = the slow large-scale state: monsoon phase, SST, pressure level",
                                "The 24 h pressure tendency is the one fast signal that still counts at 72 h"], size=14, bullets=True, spacing=10, name="physics text"),
], notes="CatBoost gain importance averaged over the three sites."))

# 18 ---------------------------------------------------------------- trees vs lstm
SLIDES.append(dict(els=frame(18, "Trees or LSTM?",
    "Similar accuracy here; each has a role") + [
    *card(0.6, 1.75, 5.9, 2.4, 1, "Gradient boosting", "78 engineered inputs, tuned, fast to train on a CPU, easy to explain; one model per lead time"),
    *card(6.8, 1.75, 5.9, 2.4, 2, "Seq2Seq LSTM", "Learns from raw sequences, one network for all horizons, consistent trajectories; hosts custom losses such as the peak loss"),
    rect(0.6, 4.4, 12.1, 2.35, NAVY, name="verdict"),
    text(0.9, 4.55, 11.5, 2.1, ["The trees benefit from feature engineering and tuning; the LSTM from its loss and architecture.",
                                "Check at BD08: giving the LSTM the 17 physics features cut its RMSE by 3–6%, narrowing the gap; at BD11 the LSTM is already best at 48–72 h.",
                                "For a fast, transparent operational baseline the trees are hard to beat; with buoy targets, forecast forcing and a GPU the LSTM has more headroom."],
         size=16, color=WHITE, spacing=10, name="verdict text"),
], notes="Be balanced: the comparison was done under a limited computing budget, which favours the cheaper trees."))

# 19 ---------------------------------------------------------------- limitations
SLIDES.append(dict(els=frame(19, "Limitations and future work",
    "The pipeline is ready for the next step: buoy targets and forecast forcing") + [
    rect(0.6, 1.75, 5.9, 5.05, TINT, name="limits card"),
    text(0.85, 1.9, 5.4, 0.5, "Limitations", size=19, bold=True, color=NAVY, face=HEAD, name="limits heading"),
    text(0.85, 2.45, 5.4, 4.25, ["Target is ERA5, which smooths cyclone peaks; buoy verification pending",
                                 "Only information up to t; no NWP forecast forcing",
                                 "Point inputs; remote swell sources unseen",
                                 "15-minute tuning budgets on a laptop CPU; LSTM untuned"], size=15, bullets=True, spacing=9, name="limits"),
    rect(6.8, 1.75, 5.9, 5.05, TINT, name="future card"),
    text(7.05, 1.9, 5.4, 0.5, "Future work", size=19, bold=True, color=NAVY, face=HEAD, name="future heading"),
    text(7.05, 2.45, 5.4, 4.25, ["Fine-tune on OMNI buoy observations",
                                 "NWP winds for t + h as decoder inputs",
                                 "Spatial inputs: ConvLSTM or graph networks",
                                 "Probabilistic (quantile) forecasts for go / no-go decisions",
                                 "Tree + LSTM ensemble on a GPU"], size=15, bullets=True, spacing=9, name="future"),
], notes="Stress that the buoy reader, QC and alignment are already built and tested, so swapping the target is a configuration change."))

# 20 ---------------------------------------------------------------- conclusions
cards = [(f"−{gain_pers[24]:.0f}%", "RMSE vs persistence at 24 h"),
         (f"−{gain_pers[72]:.0f}%", "RMSE vs persistence at 72 h")]
if gain_arima:
    cards.append((f"−{gain_arima[72]:.0f}%", "RMSE vs ARIMA at 72 h"))
if has_lstm:
    cards.append((f"{100 * under[24][0]:.0f}→{100 * under[24][1]:.0f}%", "Extremes under-predicted at 24 h, MSE → peak loss"))
SLIDES.append(dict(bg=NAVY, els=[
    *logo(0.5, 0.25, 0.85, card=True),
    text(1.5, 0.4, 11.2, 0.8, "Conclusions", size=32, color=WHITE, face=HEAD, bold=True, name="title"),
    *[e for i, (big, lab) in enumerate(cards[:4]) for e in [
        rect(0.6 + i * 3.1, 1.45, 2.85, 2.05, "13355C", name="result card"),
        text(0.8 + i * 3.1, 1.55, 2.5, 0.85, big, size=32, color=AMBER, face=BODY, bold=True, name="result value"),
        text(0.8 + i * 3.1, 2.4, 2.5, 1.05, glue(lab), size=13, color=WHITE, name="result label")]],
    text(0.6, 3.8, 6.0, 0.5, "Take-aways", size=19, bold=True, color="9FC9CE", face=HEAD, name="takeaway heading"),
    text(0.6, 4.3, 6.0, 2.6, ["Leakage-free by construction and by test", "Direct multi-step: no error feedback",
                              "Physics features make trees competitive", "Cyclones 72 h ahead need forecast forcing"],
         size=15, color=WHITE, bullets=True, spacing=8, name="takeaways"),
    rect(7.0, 3.85, 5.7, 2.45, "13355C", name="closing card"),
    text(7.25, 4.0, 5.2, 2.2, ["Data-driven point forecasts of wave height, trained on open reanalysis data, give useful skill to three days at a tiny fraction of the cost of a numerical wave model.",
                               [("Thank you. Questions?", {"bold": True, "color": AMBER, "size": 20})]], size=16, color=WHITE, spacing=16, name="closing"),
], notes="Summarise the numbers, then the one-sentence take-away. Invite questions."))

# 21 ---------------------------------------------------------------- references
from refs import REFS  # noqa: E402

DECK_REFS = ["era5", "era5ts", "incois", "imd", "wamdi", "ww3", "holthuijsen", "deo2001", "fan", "taieb", "largepond", "powell",
             "hochreiter", "schuster", "sutskever", "bahdanau", "friedman", "lightgbm", "catboost", "optuna", "tpe", "box",
             "dm", "hln", "mentaschi", "kingma"]


def _ref(i, key):
    return f"[{i}] " + REFS[key].replace("*", "")


_half = (len(DECK_REFS) + 1) // 2
SLIDES.append(dict(els=frame(21, "References", "Full list of 34 references in the thesis report (IEEE style)") + [
    text(0.6, 1.6, 5.95, 5.3, [_ref(i + 1, k) for i, k in enumerate(DECK_REFS[:_half])], size=8.5, color=INK, spacing=3, name="references left"),
    text(6.75, 1.6, 5.95, 5.3, [_ref(i + 1 + _half, k) for i, k in enumerate(DECK_REFS[_half:])], size=8.5, color=INK, spacing=3, name="references right"),
], notes="Backup slide with the sources behind the data, the models and the verification methods."))
