"""Report-only figures: data record, seasonal cycle, methodology flowchart, LSTM architecture, feature importance.

    python make_figs.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bobswh import config  # noqa: E402
from bobswh.plots import COLORS, GRID, INK, INK2, SURFACE, plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

FIG = config.PROJECT_DIR / "figures"
PROC = config.PROC_DIR
BUOYS = list(config.BUOYS)
BLOCK_FILL = {"fit": "#eef4fc", "es": "#fdf0ea", "val": "#e9f7f1", "test": "#fdf5e3"}
BLOCKS = [("fit", "2015-01-01", "2019-12-31 23:00", "Fit"), ("es", "2020-01-01", "2020-12-31 23:00", "Early stop"),
          ("val", "2021-01-01", "2022-12-31 23:00", "Validation"), ("test", "2023-01-01", "2024-12-31 23:00", "Test")]


def record():
    fig, axes = plt.subplots(len(BUOYS), 1, figsize=(7.2, 5.6), sharex=True, sharey=True)
    for ax, b in zip(axes, BUOYS):
        hs = pd.read_parquet(PROC / f"{b}_features.parquet", columns=["era5_swh"])["era5_swh"]
        for key, s, e, lab in BLOCKS:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=BLOCK_FILL[key], lw=0)
            if b == BUOYS[0]:
                ax.text(pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2, 11.3, lab, ha="center", va="bottom",
                        fontsize=9, color=INK2)
        ax.plot(hs.index, hs.to_numpy(), color=COLORS["lightgbm"], lw=0.4)
        ax.set_ylabel("$H_s$ (m)")
        ax.set_title(b, loc="left", color=INK)
        ax.set_ylim(0, 11.2)
    fig.savefig(FIG / "fig_record.png")
    plt.close(fig)


def season():
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    colors = [COLORS["lightgbm"], COLORS["catboost"], COLORS["lstm_peak"]]
    for i, (b, c) in enumerate(zip(BUOYS, colors)):
        hs = pd.read_parquet(PROC / f"{b}_features.parquet", columns=["era5_swh"])["era5_swh"]
        g = hs.groupby(hs.index.month)
        x = np.arange(1, 13) + (i - 1) * 0.22
        ax.vlines(x, g.quantile(0.05), g.quantile(0.95), color=c, lw=2, alpha=0.55)
        ax.plot(x, g.median(), "o", color=c, ms=5, label=b)
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("$H_s$ (m)")
    ax.set_xlabel("Month")
    ax.legend(ncol=3, loc="upper left")
    fig.savefig(FIG / "fig_season.png")
    plt.close(fig)


def _box(ax, x, y, w, h, head, body, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fill, ec="none"))
    ax.text(x + w / 2, y + h - 0.2, head, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=8.6, color=INK, linespacing=1.35)


def flowchart():
    stages = [
        ("1  Ingestion and QC", "ERA5 time series 2015–2024\nINCOIS buoy reader\nrange, spike and gap rules", "#eef4fc"),
        ("2  Features and windows", "wind stress, pressure tendency,\nsin/cos directions, ΔT, steepness\n72 h look-back, 5 horizons", "#eef4fc"),
        ("3  Baselines and trees", "persistence, climatology, ARIMA\nDMS LightGBM and CatBoost\nOptuna TPE on pooled RMSE", "#fdf0ea"),
        ("4  Seq2Seq LSTM", "Bi-LSTM encoder, attention,\nnon-autoregressive decoder\npeak-weighted loss vs MSE", "#e9f7f1"),
        ("5  Evaluation", "common test origins 2023–24\nRMSE, MAE, R², SI, skill\ncyclone windows, DM tests", "#fdf5e3"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, 10)
    ax.axis("off")
    for i, (head, body, fill) in enumerate(stages):
        y = 8.1 - i * 2.0
        ax.add_patch(FancyBboxPatch((0.1, y), 9.8, 1.6, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fill, ec="none"))
        ax.text(0.35, y + 0.8, head, ha="left", va="center", fontsize=11, fontweight="bold", color=INK)
        ax.text(4.1, y + 0.8, body, ha="left", va="center", fontsize=9, color=INK, linespacing=1.35)
        if i < len(stages) - 1:
            ax.annotate("", xy=(5.0, y - 0.38), xytext=(5.0, y - 0.02), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    ax.text(5.0, -0.45, "Guardrail throughout: every input at or before the forecast origin t;\n"
            "each chronological block contains all inputs and all targets of its samples",
            ha="center", fontsize=8.6, color=INK2, style="italic")
    fig.savefig(FIG / "fig_flowchart.png")
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis("off")
    _box(ax, 0.2, 0.6, 3.2, 5.8, "Encoder input", "25 steps (72 h, 3-hourly)\n\n$H_s$, $u_{10}$, $v_{10}$,\nMSLP, $T_m$\n\nstandardised on\nthe fit block", "#eef4fc")
    _box(ax, 4.0, 0.6, 3.0, 5.8, "Bi-LSTM encoder", "64 units per direction\n\nforward and backward\nstates concatenated\n→ $h_e$, $c_e$ (128)\n\nall step outputs $e_k$\nkept for attention", "#eef4fc")
    _box(ax, 7.6, 3.6, 3.2, 2.8, "LSTM decoder", "5 steps = 5 horizons\ninput: $h/72$, sin/cos doy\ninitial state $h_e$, $c_e$", "#e9f7f1")
    _box(ax, 7.6, 0.6, 3.2, 2.6, "Additive attention", "context $c_j = \\Sigma_k\\, \\alpha_{jk} e_k$\nper decoder step", "#fdf5e3")
    _box(ax, 11.4, 0.6, 3.4, 5.8, "Head and output", "dense (ReLU) → $\\Delta_j$\n\n$\\hat{H}_s(t+h_j) = H_s(t) + \\Delta_j$\n\nlosses: MSE, or the\npeak-weighted loss\n\n6, 12, 24, 48, 72 h", "#fdf0ea")
    for (x0, y0, x1, y1) in [(3.45, 3.5, 3.95, 3.5), (7.05, 5.0, 7.55, 5.0), (7.05, 1.9, 7.55, 1.9),
                             (10.85, 5.0, 11.35, 5.0), (10.85, 1.9, 11.35, 1.9), (9.2, 3.25, 9.2, 3.55)]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    fig.savefig(FIG / "fig_architecture.png")
    plt.close(fig)


NAMES = {"swh": "$H_s$", "wspd10": "$U_{10}$", "msl": "MSLP", "sst": "SST", "t2m": "$T_{2m}$", "tau": "τ",
         "tau_x": "$τ_x$", "tau_y": "$τ_y$", "mwp": "$T_m$", "u10": "$u_{10}$", "v10": "$v_{10}$", "doy_sin": "season (sin)",
         "doy_cos": "season (cos)", "mwd_sin": "wave dir. (sin)", "mwd_cos": "wave dir. (cos)", "wdir10_sin": "wind dir. (sin)",
         "wdir10_cos": "wind dir. (cos)", "air_sea_dt": "air–sea ΔT", "msl_anom": "MSLP anomaly", "steepness": "steepness",
         "wind_wave_cos": "wind–wave alignment", "rho_air": "air density"}


def pretty_feature(name):
    """'era5_swh_lag3h' -> 'Hs (t − 3 h)', 'era5_msl_tend24h_lag0h' -> 'MSLP 24 h tendency (t)'."""
    import re
    base, lag = re.match(r"(.+)_lag(\d+)h$", name).groups()
    base = base.replace("era5_", "")
    extra = ""
    m = re.match(r"(.+)_tend(\d+)h$", base)
    if m:
        base, extra = m.group(1), f" {m.group(2)} h tendency"
    m = re.match(r"(.+)_(mean|max)(\d+)h$", base)
    if m:
        base, extra = m.group(1), f" {m.group(3)} h {m.group(2)}"
    if base.startswith("doy"):
        return NAMES.get(base, base)
    when = "t" if lag == "0" else f"t − {lag} h"
    return f"{NAMES.get(base, base)}{extra} ({when})"


def importance(top=15):
    frames = []
    for b in BUOYS:
        p = config.RESULTS_DIR / "phase3" / b / "catboost_importance.csv"
        if p.exists():
            frames.append(pd.read_csv(p, index_col=0)[["t+6h", "t+72h"]].assign(buoy=b))
    if not frames:
        return
    imp = pd.concat(frames).groupby(level=0)[["t+6h", "t+72h"]].mean()
    order = imp.mean(axis=1).sort_values(ascending=False).index[:top][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = np.arange(len(order))
    ax.barh(y + 0.2, imp.loc[order, "t+6h"], 0.38, color=COLORS["lightgbm"], label="t + 6 h", edgecolor=SURFACE)
    ax.barh(y - 0.2, imp.loc[order, "t+72h"], 0.38, color=COLORS["catboost"], label="t + 72 h", edgecolor=SURFACE)
    ax.set_yticks(y, [pretty_feature(o) for o in order], fontsize=8.5)
    ax.set_xlabel("Share of total importance (mean over buoys)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.savefig(FIG / "fig_importance.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    record(); season(); flowchart(); architecture(); importance()
    print("report figures written to", FIG)
