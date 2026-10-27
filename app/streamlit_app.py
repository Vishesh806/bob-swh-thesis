"""Interactive dashboard: multi-step significant wave height forecasting in the Bay of Bengal.

    streamlit run app/streamlit_app.py

Reads only app/data/ (built by app/prepare_data.py), so it deploys on its own (e.g. Streamlit Community Cloud).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
REPO = "https://github.com/Vishesh806/bob-swh-thesis"
H = [6, 12, 24, 48, 72]
BUOYS = ["BD08", "BD10", "BD11"]
SITES = {"BD08": (18.15, 89.67, "Northern bay, on the track of most cyclones"),
         "BD10": (16.32, 87.98, "Central bay"),
         "BD11": (13.58, 84.11, "South-western bay, off Andhra Pradesh and Tamil Nadu")}
MODELS = ["persistence", "climatology", "sarimax", "lightgbm", "catboost", "lstm_mse", "lstm_peak"]
LEARNED = ["sarimax", "lightgbm", "catboost", "lstm_mse", "lstm_peak"]
LABEL = {"persistence": "Persistence", "climatology": "Climatology", "sarimax": "ARIMA", "lightgbm": "LightGBM",
         "catboost": "CatBoost", "lstm_mse": "Seq2Seq LSTM (MSE)", "lstm_peak": "Seq2Seq LSTM (peak loss)"}
COLOR = {"lightgbm": "#2a78d6", "catboost": "#eb6834", "lstm_peak": "#1baf7a", "lstm_mse": "#eda100",
         "sarimax": "#e87ba4", "persistence": "#8f8e88", "climatology": "#bdbcb6", "obs": "#0b0b0b"}
DASH = {"persistence": "dash", "climatology": "dot"}
BLOCKS = [("Fit (training)", "2015-01-01", "2019-12-31", "#cfe0f6"), ("Early stopping", "2020-01-01", "2020-12-31", "#f8d9cb"),
          ("Validation (tuning)", "2021-01-01", "2022-12-31", "#c9ecdf"), ("Test (reported)", "2023-01-01", "2024-12-31", "#fbe6b8")]
METRICS = {"rmse": ("RMSE (m)", "lower is better"), "mae": ("MAE (m)", "lower is better"),
           "r2": ("R² (variance explained)", "higher is better"), "si": ("Scatter index", "lower is better"),
           "bias": ("Bias (m)", "closer to 0 is better"), "cc": ("Correlation", "higher is better"),
           "skill_vs_persistence": ("Skill vs persistence", "higher is better; 0 = persistence"),
           "within_25cm": ("Accuracy: within ±0.25 m (%)", "higher is better"),
           "within_50cm": ("Accuracy: within ±0.5 m (%)", "higher is better")}
NAMES = {"swh": "Hs", "wspd10": "U10", "msl": "MSLP", "sst": "SST", "t2m": "T2m", "tau": "wind stress τ",
         "tau_x": "τx", "tau_y": "τy", "mwp": "wave period Tm", "u10": "u10", "v10": "v10", "doy_sin": "season (sin)",
         "doy_cos": "season (cos)", "mwd_sin": "wave dir. (sin)", "mwd_cos": "wave dir. (cos)",
         "wdir10_sin": "wind dir. (sin)", "wdir10_cos": "wind dir. (cos)", "air_sea_dt": "air–sea ΔT",
         "msl_anom": "MSLP anomaly", "steepness": "wave steepness", "wind_wave_cos": "wind–wave alignment",
         "rho_air": "air density"}

st.set_page_config(page_title="Bay of Bengal wave forecasting", page_icon="🌊", layout="wide")


def lab(k):
    """Display name of a model key (passes unknown values through unchanged)."""
    return LABEL.get(k, k)


# ---------------------------------------------------------------- helpers
def show(fig, key=None):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13), legend_title_text="",
                      hoverlabel=dict(font_size=12), plot_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor="#ecebe7", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#ecebe7", zeroline=False)
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def pretty_feature(name: str) -> str:
    m = re.match(r"(.+)_lag(\d+)h$", name)
    if not m:
        return name
    base, lag = m.groups()
    base = base.replace("era5_", "")
    extra = ""
    t = re.match(r"(.+)_tend(\d+)h$", base)
    if t:
        base, extra = t.group(1), f" {t.group(2)} h tendency"
    t = re.match(r"(.+)_(mean|max)(\d+)h$", base)
    if t:
        base, extra = t.group(1), f" {t.group(3)} h {t.group(2)}"
    if base.startswith("doy"):
        return NAMES.get(base, base)
    return f"{NAMES.get(base, base)}{extra} ({'t' if lag == '0' else f't − {lag} h'})"


@st.cache_data
def load():
    d = {"series": pd.read_parquet(DATA / "series.parquet"),
         "pred": pd.read_parquet(DATA / "predictions_test.parquet"),
         "metrics": pd.read_csv(DATA / "metrics_test.csv"),
         "pooled": pd.read_csv(DATA / "metrics_test_pooled.csv"),
         "storm": pd.read_csv(DATA / "storm_metrics_test.csv"),
         "dm": pd.read_csv(DATA / "diebold_mariano_test.csv"),
         "dm_arima": pd.read_csv(DATA / "diebold_mariano_vs_arima.csv"),
         "imp": pd.read_csv(DATA / "importance.csv"),
         "optuna": pd.read_csv(DATA / "optuna_trials.csv"),
         "hist": pd.read_csv(DATA / "lstm_history.csv"),
         "splits": pd.read_csv(DATA / "split_summary.csv"),
         "nodes": pd.read_csv(DATA / "node_alignment.csv"),
         "summary": json.loads((DATA / "summary.json").read_text()),
         "params": json.loads((DATA / "params.json").read_text()),
         "cyclones": json.loads((DATA / "cyclones.json").read_text())}
    sens = DATA / "lstm_sensitivity.csv"
    d["sens"] = pd.read_csv(sens) if sens.exists() else None
    d["series"]["time"] = pd.to_datetime(d["series"]["time"])
    d["pred"]["origin"] = pd.to_datetime(d["pred"]["origin"])
    # accuracy rates: share of forecasts within a tolerance of the observed Hs
    rows = []
    for b, g in d["pred"].groupby("buoy"):
        for m in MODELS:
            for h in H:
                e = (g[f"{m}__t+{h}h"] - g[f"obs__t+{h}h"]).abs()
                rows.append({"buoy": b, "model": m, "horizon_h": h,
                             "within_25cm": 100 * float((e <= 0.25).mean()), "within_50cm": 100 * float((e <= 0.5).mean())})
    acc = pd.DataFrame(rows)
    d["metrics"] = d["metrics"].merge(acc, on=["buoy", "model", "horizon_h"], how="left")
    return d


D = load()
M = D["metrics"]
mean_m = M.groupby(["model", "horizon_h"]).mean(numeric_only=True).reset_index()


def mval(model, h, col="rmse", buoy=None):
    src = mean_m if buoy in (None, "Mean of 3 sites") else M[M.buoy == buoy]
    return float(src[(src.model == model) & (src.horizon_h == h)][col].iloc[0])


def best(h, col="rmse", buoy=None, among=None):
    cands = among or [m for m in MODELS if m != "climatology"]
    vals = {m: mval(m, h, col, buoy) for m in cands}
    return (min if col not in ("r2", "cc", "skill_vs_persistence", "within_25cm", "within_50cm") else max)(vals, key=vals.get)


def gain(model, ref, h):
    return 100 * (1 - mval(model, h) / mval(ref, h))


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🌊 Bay of Bengal Hs forecasting")
st.sidebar.caption("M.Tech Dual Degree project · IIT Kharagpur\n\nVishesh Bhagwandas Agarwal (22NA30045)\n\nSupervisor: Prof. Swapnadip De Chowdhury")
PAGES = ["Overview", "Data explorer", "Methodology", "Model leaderboard", "Forecast explorer", "Cyclone case studies",
         "Extreme seas and peak loss", "Statistical significance", "What the models learned", "About and reproduce"]
page = st.sidebar.radio("Page", PAGES)
st.sidebar.divider()
st.sidebar.markdown(f"[Code on GitHub]({REPO}) · [Thesis report]({REPO}/blob/main/report/Thesis_Report_BoB_SWH_Forecasting.docx)")
st.sidebar.caption("Test period 2023–2024, never seen in training, early stopping or tuning. Target: ERA5 Hs at the buoy positions.")


# ---------------------------------------------------------------- pages
def page_overview():
    st.title("Multi-step significant wave height forecasting in the Bay of Bengal")
    st.markdown("Forecasting wave height **6, 12, 24, 48 and 72 hours ahead** at three INCOIS OMNI buoy sites with "
                "gradient-boosted trees and a sequence-to-sequence LSTM, trained on ten years of ERA5 and tested on "
                "2023–2024 without any look-ahead.")
    b24, b72 = best(24), best(72)
    pk = D["pooled"].set_index(["model", "horizon_h"])
    c = st.columns(5)
    c[0].metric("Best RMSE, 24 h", f"{mval(b24, 24):.3f} m", f"{-gain(b24, 'persistence', 24):.0f}% vs pers.", delta_color="inverse", help=f"{LABEL[b24]}, mean of 3 sites")
    c[1].metric("Best RMSE, 72 h", f"{mval(b72, 72):.3f} m", f"{-gain(b72, 'persistence', 72):.0f}% vs pers.", delta_color="inverse", help=f"{LABEL[b72]}, mean of 3 sites")
    c[2].metric("R², 24 h", f"{100 * mval(b24, 24, 'r2'):.0f}%",
                f"{100 * (mval(b24, 24, 'r2') - mval('persistence', 24, 'r2')):+.0f} pts vs pers.",
                help=f"Variance explained by {LABEL[b24]}, mean of 3 sites")
    c[3].metric("Within ±0.5 m, 24 h", f"{mval(b24, 24, 'within_50cm'):.0f}%",
                f"{mval(b24, 24, 'within_50cm') - mval('persistence', 24, 'within_50cm'):+.0f} pts vs pers.",
                help=f"Share of {LABEL[b24]} forecasts within 0.5 m of the observed Hs")
    c[4].metric("Missed extremes, 24 h", f"{100 * pk.loc[('lstm_peak', 24), 'under_rate_peak']:.0f}%",
                f"{100 * (pk.loc[('lstm_peak', 24), 'under_rate_peak'] - pk.loc[('lstm_mse', 24), 'under_rate_peak']):.0f} pts vs MSE",
                delta_color="inverse", help="Share of extreme seas (≥ training q95) under-predicted by the peak-loss LSTM")
    st.divider()
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Study sites")
        nodes = D["nodes"].drop_duplicates(["site", "grid_resolution_deg"])
        fig = go.Figure()
        fig.add_trace(go.Scattergeo(lon=[SITES[b][1] for b in BUOYS], lat=[SITES[b][0] for b in BUOYS], text=BUOYS,
                                    mode="markers+text", textposition="top center", name="OMNI buoy",
                                    marker=dict(size=13, color="#1b6fd1", line=dict(color="white", width=1.5)),
                                    hovertext=[f"{b}: {SITES[b][2]}" for b in BUOYS], hoverinfo="text"))
        fig.add_trace(go.Scattergeo(lon=nodes.node_lon, lat=nodes.node_lat, mode="markers", name="Matched ERA5 node",
                                    marker=dict(size=7, symbol="x", color="#eb6834"),
                                    hovertext=[f"{r.site}: {r.grid_resolution_deg}° node, {r.distance_km:.0f} km" for r in nodes.itertuples()],
                                    hoverinfo="text"))
        fig.update_geos(lonaxis_range=[77, 101], lataxis_range=[4, 24], showland=True, landcolor="#efeee9",
                        showocean=True, oceancolor="#e6f0fb", showcountries=True, countrycolor="#bdbcb6", resolution=50)
        fig.update_layout(height=430, legend=dict(orientation="h", y=-0.05))
        show(fig, "map")
    with right:
        st.subheader("What was compared")
        st.markdown("""
| Model | Type | Inputs |
|---|---|---|
| Persistence | Baseline | Hs now |
| Climatology | Baseline | Day of year |
| ARIMA(2,1,1) | Statistical | Hs history |
| LightGBM | Gradient boosting, one model per lead time | 78 physics features |
| CatBoost | Gradient boosting, one model per lead time | 78 physics features |
| Seq2Seq LSTM (MSE) | Bi-LSTM encoder, attention, decoder | 72 h of Hs, wind, pressure, period |
| Seq2Seq LSTM (peak loss) | Same network, peak-weighted loss | Same |
""")
        st.info("**Key findings.** The tree models are the most accurate overall; the LSTM ties with them at 72 h. "
                "The peak-weighted loss misses far fewer extreme seas. Three days ahead, no model foresees a cyclone "
                "peak from local data alone, which motivates adding forecast winds.")


def page_data():
    st.title("Data explorer")
    st.markdown("Hourly ERA5 reanalysis at the three buoy positions, **2015–2024 (87,672 hours per site, no gaps)**. "
                "Shaded bands show how the record is split in time.")
    var_names = {"era5_swh": "Significant wave height Hs (m)", "era5_wspd10": "Wind speed at 10 m (m/s)",
                 "era5_msl": "Mean sea-level pressure (hPa)", "era5_mwp": "Mean wave period (s)",
                 "era5_sst": "Sea surface temperature (°C)", "era5_t2m": "Air temperature at 2 m (°C)"}
    c1, c2, c3 = st.columns([1, 1.4, 1])
    var = c1.selectbox("Variable", list(var_names), format_func=lambda k: var_names.get(k, k))
    sites = c2.multiselect("Sites", BUOYS, default=["BD08"])
    res = c3.selectbox("Resolution", ["Daily mean", "Daily max", "Hourly"], index=1)
    S = D["series"]
    fig = go.Figure()
    for name, s, e, col in BLOCKS:
        fig.add_vrect(x0=s, x1=e + " 23:00", fillcolor=col, opacity=0.45, line_width=0, annotation_text=name,
                      annotation_position="top left", annotation_font_size=11)
    palette = {"BD08": "#1b6fd1", "BD10": "#eb6834", "BD11": "#1baf7a"}
    for b in sites:
        s = S[S.buoy == b].set_index("time")[var]
        if res == "Daily mean":
            s = s.resample("1D").mean()
        elif res == "Daily max":
            s = s.resample("1D").max()
        fig.add_trace(go.Scattergl(x=s.index, y=s.values, name=b, mode="lines", line=dict(width=1, color=palette[b])))
    fig.update_layout(height=430, yaxis_title=var_names[var], xaxis=dict(rangeslider=dict(visible=True)))
    show(fig, "record")

    left, right = st.columns(2)
    with left:
        st.subheader("Seasonal cycle")
        s = S[S.buoy.isin(sites or BUOYS)].copy()
        s["month"] = s.time.dt.month
        fig = px.box(s, x="month", y=var, color="buoy", color_discrete_map=palette, points=False,
                     labels={"month": "Month", var: var_names[var], "buoy": "Site"})
        fig.update_layout(height=380, xaxis=dict(tickmode="array", tickvals=list(range(1, 13)),
                                                  ticktext=list("JFMAMJJASOND")))
        show(fig, "season")
    with right:
        st.subheader("Largest wave events (distinct, ≥ 5 days apart)")
        rows = []
        for b in BUOYS:
            g = S[S.buoy == b].set_index("time")
            hs = g["era5_swh"].sort_values(ascending=False)
            picked = []
            for t, v in hs.items():
                if all(abs((t - p).total_seconds()) > 5 * 86400 for p in picked):
                    picked.append(t)
                    rows.append({"Site": b, "Peak time (UTC)": t.strftime("%d %b %Y %H:00"), "Peak Hs (m)": round(float(v), 1),
                                 "Min MSLP ±2 days (hPa)": round(float(g["era5_msl"][t - pd.Timedelta("2D"):t + pd.Timedelta("2D")].min()))})
                if len(picked) == 4:
                    break
        table(pd.DataFrame(rows), hide_index=True, height=380)
    st.subheader("Summary statistics")
    stats = S.groupby("buoy").agg(**{"Mean Hs (m)": ("era5_swh", "mean"), "Max Hs (m)": ("era5_swh", "max"),
                                     "q95 Hs (m)": ("era5_swh", lambda x: x.quantile(0.95)),
                                     "Mean period (s)": ("era5_mwp", "mean"), "Mean wind (m/s)": ("era5_wspd10", "mean"),
                                     "Min MSLP (hPa)": ("era5_msl", "min"), "Mean SST (°C)": ("era5_sst", "mean")}).round(2)
    table(stats)


def page_method():
    st.title("Methodology")
    steps = [("1 · Ingestion and QC", "ERA5 time series 2015–2024 for a 1.5° box per buoy; INCOIS reader; physical range checks, "
              "isolated-spike test that keeps storm peaks, forward-fill of gaps ≤ 3 h; KD-tree match to the nearest valid ERA5 node."),
             ("2 · Physics features", "Wind stress τ = ρa·Cd(U)·|U|·(u, v) with a Large–Pond drag coefficient capped at 25 m/s; "
              "6–24 h stress history; pressure tendency and 30-day anomaly; air–sea ΔT; sin/cos directions; wave steepness."),
             ("3 · Baselines and trees", "Persistence, climatology, ARIMA(2,1,1). Direct multi-step LightGBM and CatBoost, one "
              "model per lead time, 78 inputs, tuned by Optuna (TPE) on the RMSE pooled over the five lead times."),
             ("4 · Seq2Seq LSTM", "Bi-LSTM encoder over 72 h (25 steps), non-autoregressive decoder with additive attention, "
              "residual head (Hs(t) + Δ); trained with MSE and with a peak-weighted loss."),
             ("5 · Evaluation", "Same 17,400 test origins for every model; RMSE, MAE, R², scatter index, skill, accuracy rates; "
              "Diebold–Mariano tests; scores in cyclone windows and on extreme seas.")]
    cols = st.columns(5)
    for col, (head, body) in zip(cols, steps):
        col.markdown(f"**{head}**")
        col.caption(body)
    st.divider()
    st.subheader("Zero look-ahead: four chronological blocks")
    st.markdown("A sample is used only if its **whole 72 h input window and all its targets** lie inside one block. "
                "This removes the classic multi-step leak where training targets spill into the validation period.")
    blk = pd.DataFrame([{"Block": n, "Start": s, "End": e + " 23:00"} for n, s, e, _ in BLOCKS])
    fig = px.timeline(blk, x_start="Start", x_end="End", y="Block", color="Block",
                      color_discrete_sequence=[c for *_, c in BLOCKS])
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_layout(height=230, showlegend=False)
    show(fig, "blocks")
    sp = D["splits"][D["splits"].buoy == "BD08"][["split", "n_samples", "first_origin", "last_origin"]]
    sp.columns = ["Block", "Samples (BD08)", "First forecast origin", "Last forecast origin"]
    table(sp, hide_index=True)

    st.divider()
    st.subheader("The peak-weighted loss, interactively")
    st.markdown("Loss weight = 1 + s(y)·(β + γ·[under-prediction]), with s(y) = sigmoid((y − q)/τ). "
                "In calm seas it is plain squared error; above the 95th percentile, errors count more and missed peaks most.")
    c1, c2, c3, c4 = st.columns(4)
    q = c1.slider("Threshold q (m)", 1.5, 4.5, 3.0, 0.1)
    tau = c2.slider("Width τ (m)", 0.05, 1.0, 0.25, 0.05)
    beta = c3.slider("β (extra weight on all errors)", 0.0, 4.0, 1.0, 0.5)
    gamma = c4.slider("γ (extra weight on under-prediction)", 0.0, 10.0, 4.0, 0.5)
    y = np.linspace(0, 6, 300)
    s = 1 / (1 + np.exp(-(y - q) / tau))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y, y=1 + s * beta, name="Over-prediction weight", line=dict(color="#2a78d6", width=3)))
    fig.add_trace(go.Scatter(x=y, y=1 + s * (beta + gamma), name="Under-prediction weight", line=dict(color="#d03b3b", width=3)))
    fig.add_vline(x=q, line_dash="dash", line_color="#8f8e88", annotation_text="q")
    fig.update_layout(height=340, xaxis_title="Observed Hs (m)", yaxis_title="Weight on the squared error")
    show(fig, "loss")


def page_leaderboard():
    st.title("Model leaderboard")
    c1, c2, c3 = st.columns([1.3, 1, 1])
    metric = c1.selectbox("Metric", list(METRICS), format_func=lambda k: METRICS[k][0] if k in METRICS else k)
    site = c2.selectbox("Site", ["Mean of 3 sites"] + BUOYS)
    models = c3.multiselect("Models", MODELS, default=MODELS, format_func=lab)
    st.caption(f"{METRICS[metric][0]}: {METRICS[metric][1]}. Test period 2023–2024.")
    src = mean_m if site == "Mean of 3 sites" else M[M.buoy == site]
    src = src[src.model.isin(models)]
    scale = 100 if metric == "r2" else 1
    fig = go.Figure()
    for m in models:
        r = src[src.model == m].sort_values("horizon_h")
        if metric == "skill_vs_persistence" and m == "persistence":
            continue
        fig.add_trace(go.Scatter(x=r.horizon_h, y=r[metric] * scale, name=LABEL[m], mode="lines+markers",
                                 line=dict(color=COLOR[m], width=3, dash=DASH.get(m, "solid")), marker=dict(size=8)))
    fig.update_layout(height=440, xaxis=dict(title="Lead time (h)", tickvals=H),
                      yaxis_title=METRICS[metric][0] + (" (%)" if metric == "r2" else ""))
    show(fig, "lead")

    st.subheader("Full table")
    piv = src.pivot(index="model", columns="horizon_h", values=metric).reindex([m for m in MODELS if m in models])
    piv.index = [LABEL[m] for m in piv.index]
    piv.columns = [f"{h} h" for h in piv.columns]
    better_high = metric in ("r2", "cc", "skill_vs_persistence", "within_25cm", "within_50cm")
    fmt = "{:.1f}" if metric.startswith("within") else "{:.3f}"
    styled = piv.style.format(fmt)
    try:
        styled = (styled.highlight_max if better_high else styled.highlight_min)(color="#d4f0e3", axis=0)
    except Exception:
        pass
    table(styled)

    st.subheader("All models at one lead time")
    h = st.select_slider("Lead time", H, value=24, format_func=lambda x: f"{x} h")
    bars = src[src.horizon_h == h].copy()
    bars["Model"] = bars.model.map(LABEL)
    fig = px.bar(bars.sort_values(metric, ascending=not better_high), x=metric, y="Model", orientation="h",
                 color="model", color_discrete_map=COLOR, text_auto=".3f" if not metric.startswith("within") else ".1f")
    fig.update_layout(height=360, showlegend=False, xaxis_title=METRICS[metric][0], yaxis_title="")
    show(fig, "bars")


def page_forecast():
    st.title("Forecast explorer")
    c1, c2, c3 = st.columns([0.8, 0.8, 2])
    site = c1.selectbox("Site", BUOYS)
    h = c2.selectbox("Lead time", H, index=2, format_func=lambda x: f"{x} h")
    models = c3.multiselect("Models", MODELS, default=["lightgbm", "catboost", "lstm_peak"], format_func=lab)
    P = D["pred"][D["pred"].buoy == site].copy()
    P["valid"] = P.origin + pd.Timedelta(hours=h)
    lo, hi = P.valid.min().date(), P.valid.max().date()
    start, end = st.slider("Window (forecast valid time)", min_value=lo, max_value=hi,
                           value=(pd.Timestamp("2023-05-01").date(), pd.Timestamp("2023-06-15").date()), format="DD MMM YYYY")
    W = P[(P.valid >= pd.Timestamp(start)) & (P.valid <= pd.Timestamp(end) + pd.Timedelta(hours=23))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=W.valid, y=W[f"obs__t+{h}h"], name="ERA5 (observed)", line=dict(color=COLOR["obs"], width=2.5)))
    for m in models:
        fig.add_trace(go.Scatter(x=W.valid, y=W[f"{m}__t+{h}h"], name=LABEL[m],
                                 line=dict(color=COLOR[m], width=1.6, dash=DASH.get(m, "solid"))))
    fig.update_layout(height=440, yaxis_title="Hs (m)", xaxis_title="Valid time (UTC)", hovermode="x unified",
                      title=f"{site}: forecasts issued {h} h before each valid time")
    show(fig, "fc")
    st.caption("Each forecast is placed at the time it is valid for: a 24 h forecast plotted at noon on 10 May was issued at noon on 9 May.")
    if models:
        left, right = st.columns(2)
        with left:
            m = st.selectbox("Scatter and error for", models, format_func=lab)
            fig = px.density_heatmap(P, x=f"obs__t+{h}h", y=f"{m}__t+{h}h", nbinsx=60, nbinsy=60,
                                     color_continuous_scale="Blues", labels={f"obs__t+{h}h": "Observed Hs (m)", f"{m}__t+{h}h": "Forecast Hs (m)"})
            top = float(P[f"obs__t+{h}h"].max())
            fig.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(color="#52514e", dash="dash"))
            fig.update_layout(height=400, title=f"{LABEL[m]}, whole test period", coloraxis_showscale=False)
            show(fig, "sc")
        with right:
            err = pd.DataFrame({LABEL[x]: P[f"{x}__t+{h}h"] - P[f"obs__t+{h}h"] for x in models}).melt(var_name="Model", value_name="Error (m)")
            fig = px.histogram(err, x="Error (m)", color="Model", barmode="overlay", nbins=120, histnorm="percent",
                               color_discrete_map={LABEL[x]: COLOR[x] for x in models}, range_x=[-2, 2])
            fig.update_layout(height=400, title="Error distribution (forecast − observed)", yaxis_title="% of forecasts")
            show(fig, "hist")
        row = M[(M.buoy == site) & (M.horizon_h == h) & (M.model.isin(models))].set_index("model").reindex(models)
        k = pd.DataFrame({"RMSE (m)": row.rmse, "MAE (m)": row.mae, "Bias (m)": row.bias, "R²": row.r2,
                          "Within ±0.25 m (%)": row.within_25cm, "Within ±0.5 m (%)": row.within_50cm}).round(3)
        k.index = [LABEL[x] for x in k.index]
        table(k)


def page_cyclones():
    st.title("Cyclone case studies (test period)")
    cyc = D["cyclones"]
    c1, c2, c3 = st.columns([1, 0.8, 2])
    name = c1.selectbox("Cyclone", list(cyc), index=0)
    site = c2.selectbox("Site", BUOYS)
    models = c3.multiselect("Models", LEARNED + ["persistence"], default=["sarimax", "lightgbm", "catboost", "lstm_peak"],
                            format_func=lab)
    s0, e0 = pd.Timestamp(cyc[name][0]) - pd.Timedelta(days=2), pd.Timestamp(cyc[name][1]) + pd.Timedelta(days=3)
    P = D["pred"][D["pred"].buoy == site]
    peak_hs, peak_t = None, None
    cols = st.columns(2)
    for col, h in zip(cols, (24, 72)):
        v = P.origin + pd.Timedelta(hours=h)
        W = P[(v >= s0) & (v <= e0)]
        vt = W.origin + pd.Timedelta(hours=h)
        if h == 24 and len(W):
            i = W[f"obs__t+{h}h"].idxmax()
            peak_hs, peak_t = float(W.loc[i, f"obs__t+{h}h"]), (W.loc[i, "origin"] + pd.Timedelta(hours=h))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vt, y=W[f"obs__t+{h}h"], name="ERA5 (observed)", line=dict(color=COLOR["obs"], width=3)))
        for m in models:
            fig.add_trace(go.Scatter(x=vt, y=W[f"{m}__t+{h}h"], name=LABEL[m], line=dict(color=COLOR[m], width=2, dash=DASH.get(m, "solid"))))
        fig.update_layout(height=420, title=f"{h} h ahead", yaxis_title="Hs (m)", hovermode="x unified",
                          legend=dict(orientation="h", y=-0.2))
        with col:
            show(fig, f"cy{h}")
    if peak_hs is not None:
        st.markdown(f"**{name} at {site}:** peak ERA5 Hs **{peak_hs:.1f} m** on {peak_t:%d %b %Y %H:00} UTC "
                    f"(cyclone window {cyc[name][0]} to {cyc[name][1]}). Small peaks mean the storm passed far from this site; "
                    f"try another site.")
    st.subheader("Errors inside all named cyclone windows of 2023–2024")
    sm = D["storm"].groupby(["model", "horizon_h"])[["rmse_storm", "bias_storm"]].mean().reset_index()
    ref = mean_m.set_index(["model", "horizon_h"])["rmse"]
    sm["all"] = [ref.loc[(r.model, r.horizon_h)] for r in sm.itertuples()]
    sm = sm[sm.model != "climatology"]
    fig = go.Figure()
    for m in [x for x in MODELS if x != "climatology"]:
        r = sm[sm.model == m].sort_values("horizon_h")
        fig.add_trace(go.Scatter(x=r.horizon_h, y=r.rmse_storm, name=LABEL[m], mode="lines+markers",
                                 line=dict(color=COLOR[m], width=3, dash=DASH.get(m, "solid"))))
    fig.update_layout(height=380, xaxis=dict(title="Lead time (h)", tickvals=H), yaxis_title="RMSE inside cyclone windows (m)")
    show(fig, "storm")
    st.caption("Mean over the three sites. Inside cyclone windows the errors are about twice the all-sample errors; "
               "the peak-loss LSTM has the smallest bias.")


def page_peak():
    st.title("Extreme seas and the peak-weighted loss")
    st.markdown("Extreme sea states = observed Hs at or above the 95th percentile of the training years "
                f"({', '.join(f'{b}: {v['peak_threshold_m']:.2f} m' for b, v in D['summary']['buoys'].items())}), "
                "all three sites pooled.")
    pk = D["pooled"]
    show_models = ["persistence", "sarimax", "lightgbm", "catboost", "lstm_mse", "lstm_peak"]
    c = st.columns(3)
    a = pk.set_index(["model", "horizon_h"])
    c[0].metric("Missed extremes at 24 h, LSTM (MSE)", f"{100 * a.loc[('lstm_mse', 24), 'under_rate_peak']:.0f}%")
    c[1].metric("Missed extremes at 24 h, LSTM (peak loss)", f"{100 * a.loc[('lstm_peak', 24), 'under_rate_peak']:.0f}%",
                f"{100 * (a.loc[('lstm_peak', 24), 'under_rate_peak'] - a.loc[('lstm_mse', 24), 'under_rate_peak']):.0f} pts", delta_color="inverse")
    c[2].metric("Cost: overall RMSE at 24 h", f"+{100 * (a.loc[('lstm_peak', 24), 'rmse'] / a.loc[('lstm_mse', 24), 'rmse'] - 1):.0f}%",
                "peak loss vs MSE", delta_color="off")
    opts = {"under_rate_peak": "Share of extremes under-predicted (%)", "bias_peak": "Mean error on extremes (m)",
            "rmse_peak": "RMSE on extremes (m)", "rmse": "RMSE on all samples (m)"}
    which = st.radio("Measure", list(opts), format_func=lambda k: opts.get(k, k), horizontal=True)
    d = pk[pk.model.isin(show_models)].copy()
    d["Model"] = d.model.map(LABEL)
    d["value"] = d[which] * (100 if which == "under_rate_peak" else 1)
    fig = px.bar(d, x="horizon_h", y="value", color="model", barmode="group", color_discrete_map=COLOR,
                 labels={"horizon_h": "Lead time (h)", "value": opts[which]},
                 category_orders={"model": show_models})
    fig.for_each_trace(lambda t: t.update(name=LABEL[t.name]))
    fig.update_layout(height=430, xaxis=dict(tickvals=H, type="category"))
    show(fig, "peak")
    st.caption("The peak-weighted loss trades a little accuracy in ordinary seas for fewer missed storms, the right "
               "trade for a warning system. The loss weights can be explored on the Methodology page.")


def page_significance():
    st.title("Statistical significance")
    st.markdown("Diebold–Mariano tests of equal squared error (Harvey–Leybourne–Newbold correction, Newey–West variance "
                "for overlapping hourly forecasts). **Small p** = the model is significantly worse than the reference.")
    mode = st.radio("Reference", ["Best model at each site and lead time", "ARIMA (classical baseline)"], horizontal=True)
    dm = D["dm"] if mode.startswith("Best") else D["dm_arima"]
    site = st.selectbox("Site", BUOYS)
    sub = dm[(dm.buoy == site)]
    models = [m for m in LEARNED + ["persistence", "climatology"] if m in set(sub.model) | set(sub.reference)]
    z, txt, hov = [], [], []
    for m in models:
        zr, tr, hr = [], [], []
        for h in H:
            r = sub[(sub.horizon_h == h) & (sub.model == m)]
            ref = sub[sub.horizon_h == h]["reference"].iloc[0]
            if m == ref:
                zr.append(0); tr.append("<b>ref.</b>"); hr.append(f"{LABEL[m]} is the reference (lowest RMSE) at {h} h")
            elif len(r):
                p = float(r.p_value.iloc[0]); stat = float(r.dm_stat.iloc[0])
                cat = (3 if p < 0.01 else 2 if p < 0.05 else 0) * (1 if stat > 0 else -1)
                ptxt = "< 0.001" if p < 1e-3 else f"{p:.3f}"
                zr.append(cat); tr.append(ptxt)
                verdict = ("not significantly different from" if p >= 0.05 else
                           "significantly worse than" if stat > 0 else "significantly better than")
                hr.append(f"{LABEL[m]} at {h} h: {verdict} {LABEL[ref]} (p = {ptxt})")
            else:
                zr.append(None); tr.append(""); hr.append("")
        z.append(zr); txt.append(tr); hov.append(hr)
    # z: -3 / -2 significantly better (p < 0.01 / 0.05), 0 no significant difference or reference, 2 / 3 significantly worse
    colorscale = [[0, "#1b6fd1"], [1 / 12, "#1b6fd1"], [1 / 12, "#9cc3f0"], [1 / 4, "#9cc3f0"], [1 / 4, "#ecebe6"],
                  [3 / 4, "#ecebe6"], [3 / 4, "#f4b9a0"], [11 / 12, "#f4b9a0"], [11 / 12, "#d03b3b"], [1, "#d03b3b"]]
    fig = go.Figure(go.Heatmap(z=z, x=[f"{h} h" for h in H], y=[LABEL[m] for m in models], text=txt, texttemplate="%{text}",
                               textfont=dict(size=13), colorscale=colorscale, zmin=-3, zmax=3, showscale=False,
                               customdata=hov, hovertemplate="%{customdata}<extra></extra>", xgap=3, ygap=3))
    fig.update_layout(height=80 + 55 * len(models), yaxis=dict(autorange="reversed"))
    show(fig, "dm")
    st.caption("Cells show the p-value; hover for the verdict. Red: significantly worse than the reference (dark p < 0.01, light p < 0.05). Grey: the reference itself, "
               "or no significant difference. Blue: significantly better than the reference (only possible against ARIMA).")


def page_inside():
    st.title("What the models learned")
    t1, t2, t3 = st.tabs(["Feature importance", "LSTM training", "Hyperparameter search"])
    with t1:
        c1, c2, c3 = st.columns(3)
        model = c1.selectbox("Tree model", ["catboost", "lightgbm"], format_func=lab)
        h = c2.selectbox("Lead time", H, index=4, format_func=lambda x: f"{x} h")
        site = c3.selectbox("Site", ["Mean of 3 sites"] + BUOYS, key="imp_site")
        I = D["imp"][D["imp"].model == model]
        if site != "Mean of 3 sites":
            I = I[I.buoy == site]
        s = I.groupby("feature")[f"t+{h}h"].mean().sort_values(ascending=False).head(15)[::-1]
        fig = go.Figure(go.Bar(x=100 * s.values, y=[pretty_feature(f) for f in s.index], orientation="h",
                               marker_color=COLOR[model]))
        fig.update_layout(height=520, xaxis_title="Share of total importance (%)", title=f"{LABEL[model]}, {h} h ahead, top 15")
        show(fig, "imp")
        st.caption("At 6 h the forecast rests on the last few hours of Hs (persistence plus a correction). At 72 h it "
                   "rests on season, SST and pressure: a climatology conditioned on the current large-scale state.")
    with t2:
        site = st.selectbox("Site", BUOYS, key="hist_site")
        Hh = D["hist"][D["hist"].buoy == site]
        fig = go.Figure()
        for m in ("lstm_mse", "lstm_peak"):
            r = Hh[Hh.model == m]
            fig.add_trace(go.Scatter(x=r.epoch + 1, y=r["loss"], name=f"{LABEL[m]}: training", line=dict(color=COLOR[m], width=3)))
            fig.add_trace(go.Scatter(x=r.epoch + 1, y=r["val_loss"], name=f"{LABEL[m]}: early-stopping block",
                                     line=dict(color=COLOR[m], width=3, dash="dash")))
        fig.update_layout(height=420, xaxis_title="Epoch", yaxis_title="Loss")
        show(fig, "hist_lstm")
        if D["sens"] is not None:
            st.markdown("**Sensitivity check at BD08** (LSTM, MSE): lower learning rate and more dropout, with the 5 core "
                        "inputs and with 17 physics inputs.")
            s = D["sens"].rename(columns={"config": "Configuration", "best_epoch": "Best epoch", **{f"rmse_{h}h": f"RMSE {h} h" for h in H}})
            table(s.drop(columns=["variant", "epochs"], errors="ignore"), hide_index=True)
    with t3:
        c1, c2 = st.columns(2)
        model = c1.selectbox("Model", ["lightgbm", "catboost"], format_func=lab, key="opt_m")
        site = c2.selectbox("Site", BUOYS, key="opt_s")
        O = D["optuna"][(D["optuna"].model == model) & (D["optuna"].buoy == site)].copy()
        O["best so far"] = O.loc[O.state == "COMPLETE", "value"].cummin()
        O["best so far"] = O["best so far"].ffill()
        fig = go.Figure()
        for state, col in (("COMPLETE", COLOR[model]), ("PRUNED", "#bdbcb6")):
            r = O[O.state == state]
            fig.add_trace(go.Scatter(x=r.number, y=r.value, mode="markers", name=state.title(), marker=dict(color=col, size=9)))
        fig.add_trace(go.Scatter(x=O.number, y=O["best so far"], mode="lines", name="Best so far", line=dict(color="#0b0b0b", width=2)))
        fig.update_layout(height=400, xaxis_title="Trial", yaxis_title="Pooled multi-horizon RMSE on validation (m)",
                          title=f"Optuna TPE search, {LABEL[model]} at {site}")
        show(fig, "opt")
        p = D["params"][f"{site}|{model}"]
        st.markdown("**Selected hyperparameters**")
        table(pd.DataFrame([{"Parameter": k, "Value": f"{v:.4g}" if isinstance(v, float) else str(v)}
                            for k, v in p["params"].items()]), hide_index=True)
        st.markdown("**Boosting rounds per lead time (early stopping):** " +
                    ", ".join(f"{h} h: {n}" for h, n in zip(H, p["best_iterations"])))


def page_about():
    st.title("About and reproduce")
    st.markdown(f"""
**Project.** Multi-step significant wave height forecasting in the Bay of Bengal using hybrid machine-learning and
deep-learning frameworks. M.Tech Dual Degree, Department of Ocean Engineering and Naval Architecture, IIT Kharagpur.

**Author.** Vishesh Bhagwandas Agarwal (22NA30045) · **Supervisor.** Professor Swapnadip De Chowdhury

**Code, report and slides.** [{REPO}]({REPO})

**Data.** ECMWF ERA5 reanalysis (Hersbach et al. 2020), distributed by the Copernicus Climate Change Service under
CC-BY-4.0, retrieved as hourly time series for a 1.5° box around each INCOIS OMNI buoy. All results here use ERA5 Hs
at the buoy positions as the target; the buoy reader and quality control are implemented for when the OMNI records
arrive.

**Guardrails.** Every input is at or before the forecast origin; each chronological block holds all inputs and targets
of its samples; scalers and thresholds come from the training block; 44 automated tests check these rules.

**Limitations.** ERA5 smooths cyclone peaks; no numerical-weather-prediction forcing for t + h; point inputs only;
tuning limited by a laptop CPU.
""")
    st.code("python3 -m venv .venv && .venv/bin/pip install -r requirements.txt\n"
            ".venv/bin/python scripts/download_era5_timeseries.py\n"
            ".venv/bin/python scripts/build_features.py --era5-only\n"
            "PHASE=3 bash scripts/run_era5_only.sh && PHASE=4 bash scripts/run_era5_only.sh\n"
            ".venv/bin/python scripts/run_phase5.py && .venv/bin/python app/prepare_data.py\n"
            "streamlit run app/streamlit_app.py", language="bash")


{"Overview": page_overview, "Data explorer": page_data, "Methodology": page_method, "Model leaderboard": page_leaderboard,
 "Forecast explorer": page_forecast, "Cyclone case studies": page_cyclones, "Extreme seas and peak loss": page_peak,
 "Statistical significance": page_significance, "What the models learned": page_inside,
 "About and reproduce": page_about}[page]()
