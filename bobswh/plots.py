"""Thesis figures (static, print-ready PNG at 300 dpi).

Colour follows the model, never its rank: each model keeps one hue in every
figure (reference categorical palette, slots 1-5 in their validated order).
Observations are black; the naive baselines are neutral grey and dashed so
they recede behind the learned models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .evaluation import MODEL_LABELS  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
COLORS = {
    "lightgbm": "#2a78d6",    # slot 1 blue
    "catboost": "#eb6834",    # slot 2 orange
    "lstm_peak": "#1baf7a",   # slot 3 aqua
    "lstm_mse": "#eda100",    # slot 4 yellow
    "sarimax": "#e87ba4",     # slot 5 magenta
    "persistence": "#9a9993",
    "climatology": "#bdbcb6",
}
DASHED = {"persistence", "climatology"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
})


def _header(fig, title: str, ax, ncol: int, top: float | None = None) -> None:
    """Title and a shared legend stacked above the axes (sizes in inches, so they never overlap)."""
    import math
    h = fig.get_figheight()
    handles, labels = ax.get_legend_handles_labels()
    rows = math.ceil(len(labels) / ncol)
    title_in, row_in, gap_in = 0.34, 0.24, 0.32
    fig.suptitle(title, x=0.01, y=1 - 0.04 / h, ha="left", va="top", fontsize=12, color=INK)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 1 - title_in / h), ncol=ncol, frameon=False)
    fig.subplots_adjust(top=1 - (title_in + rows * row_in + gap_in) / h)


def _series(pred: pd.DataFrame, col: str, h: int) -> pd.Series:
    """Forecast or observation re-indexed by its valid time (origin + h)."""
    return pd.Series(pred[col].to_numpy(), index=pred.index + pd.Timedelta(hours=h))


def _line(ax, s: pd.Series, model: str, lw: float = 1.4) -> None:
    ax.plot(s.index, s.to_numpy(), color=COLORS.get(model, INK2), lw=lw,
            ls="--" if model in DASHED else "-", label=MODEL_LABELS.get(model, model))


def timeline(pred: pd.DataFrame, start: pd.Timestamp, days: int, models: Sequence[str], horizons: Sequence[int],
             title: str, out: Path) -> None:
    """Forecast vs observed Hs over a continuous window, one panel per horizon."""
    end = start + pd.Timedelta(days=days)
    fig, axes = plt.subplots(len(horizons), 1, figsize=(7.2, 2.3 * len(horizons)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, horizons):
        obs = _series(pred, f"obs__t+{h}h", h)[start:end]
        ax.plot(obs.index, obs.to_numpy(), color=INK, lw=1.8, label="ERA5 (observed)")
        for m in models:
            _line(ax, _series(pred, f"{m}__t+{h}h", h)[start:end], m)
        ax.set_ylabel("$H_s$ (m)")
        ax.set_title(f"Lead time t + {h} h", loc="left", color=INK)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=4 if days > 12 else 1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    _header(fig, title, axes[0], min(len(models) + 1, 3), top=0.82)
    fig.savefig(out)
    plt.close(fig)


def bars(rmse: pd.DataFrame, models: Sequence[str], horizons: Sequence[int], title: str, out: Path) -> None:
    """Grouped bars of RMSE: one panel per buoy (columns of a MultiIndex frame (buoy, model) x horizon)."""
    buoys = list(dict.fromkeys(rmse.index.get_level_values(0)))
    fig, axes = plt.subplots(1, len(buoys), figsize=(2.6 * len(buoys) + 0.8, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.8 / len(models)
    for ax, b in zip(axes, buoys):
        x = np.arange(len(horizons))
        for i, m in enumerate(models):
            vals = [rmse.loc[(b, m), h] for h in horizons]
            pos = x - 0.4 + width * (i + 0.5)
            ax.bar(pos, vals, width * 0.92, color=COLORS.get(m, INK2), label=MODEL_LABELS.get(m, m),
                   edgecolor=SURFACE, linewidth=0.8)
        ax.set_xticks(x, [f"t + {h} h" for h in horizons])
        ax.set_title(b, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test RMSE (m)")
    _header(fig, title, axes[0], 3, top=0.74)
    fig.savefig(out)
    plt.close(fig)


def rmse_vs_horizon(table: pd.DataFrame, models: Sequence[str], title: str, out: Path, value: str = "rmse",
                    ylabel: str = "Test RMSE (m)") -> None:
    """Metric vs lead time, one panel per buoy; ``table`` is indexed (buoy, model, horizon)."""
    buoys = list(dict.fromkeys(table.index.get_level_values(0)))
    fig, axes = plt.subplots(1, len(buoys), figsize=(2.6 * len(buoys) + 0.8, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, buoys):
        for m in models:
            s = table.loc[(b, m), value]
            ax.plot(s.index, s.to_numpy(), color=COLORS.get(m, INK2), lw=2, ls="--" if m in DASHED else "-",
                    marker="o", ms=4, label=MODEL_LABELS.get(m, m))
        ax.set_xticks(list(s.index))
        ax.set_xlabel("Lead time (h)")
        ax.set_title(b, color=INK)
    axes[0].set_ylabel(ylabel)
    _header(fig, title, axes[0], 3, top=0.72)
    fig.savefig(out)
    plt.close(fig)


def peak_bias(table: pd.DataFrame, models: Sequence[str], horizons: Sequence[int], title: str, out: Path) -> None:
    """Bias and under-prediction rate on the extreme subset (obs >= q95), pooled over buoys."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    width = 0.8 / len(models)
    x = np.arange(len(horizons))
    for k, (col, lab) in enumerate((("bias_peak", "Mean error at peaks (m)"),
                                    ("under_rate_peak", "Share of peaks under-predicted"))):
        ax = axes[k]
        for i, m in enumerate(models):
            vals = [table.loc[(m, h), col] for h in horizons]
            ax.bar(x - 0.4 + width * (i + 0.5), vals, width * 0.92, color=COLORS.get(m, INK2),
                   label=MODEL_LABELS.get(m, m), edgecolor=SURFACE, linewidth=0.8)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, [f"{h} h" for h in horizons])
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
    fig.subplots_adjust(wspace=0.32)
    _header(fig, title, axes[0], len(models), top=0.8)
    fig.savefig(out)
    plt.close(fig)


def scatter(pred: pd.DataFrame, models: Sequence[str], h: int, title: str, out: Path) -> None:
    """Predicted vs observed Hs at one horizon (hexbin density), one panel per model."""
    fig, axes = plt.subplots(1, len(models), figsize=(2.5 * len(models), 2.7), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    obs = pred[f"obs__t+{h}h"].to_numpy()
    lim = float(np.nanmax(obs)) * 1.05
    for ax, m in zip(axes, models):
        p = pred[f"{m}__t+{h}h"].to_numpy()
        ax.hexbin(obs, p, gridsize=45, bins="log", cmap="Blues", mincnt=1, extent=(0, lim, 0, lim))
        ax.plot([0, lim], [0, lim], color=INK2, lw=0.8)
        ax.set_title(MODEL_LABELS.get(m, m), color=INK, fontsize=10)
        ax.set_xlabel("Observed $H_s$ (m)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Forecast $H_s$ (m)")
    fig.suptitle(title, x=0.01, ha="left", y=1.04, fontsize=12, color=INK)
    fig.savefig(out)
    plt.close(fig)


def history(histories: dict[str, pd.DataFrame], title: str, out: Path) -> None:
    """LSTM training curves (training and early-stopping loss per epoch)."""
    fig, axes = plt.subplots(1, len(histories), figsize=(3.4 * len(histories), 2.6))
    axes = np.atleast_1d(axes)
    for ax, (name, h) in zip(axes, histories.items()):
        ax.plot(h.index + 1, h["loss"], color=COLORS.get(name, INK), lw=2, label="training")
        ax.plot(h.index + 1, h["val_loss"], color=COLORS.get(name, INK), lw=2, ls="--", label="early-stopping block")
        ax.set_title(MODEL_LABELS.get(name, name), color=INK, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    fig.suptitle(title, x=0.01, ha="left", y=1.04, fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
