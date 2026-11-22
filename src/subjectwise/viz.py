"""Static matplotlib figures for the demo and the README.

Figures are built on :class:`matplotlib.figure.Figure` directly (no pyplot), so they work in
headless environments and leave no global state behind. Styling: solid hairline grid, thin marks,
text in ink colours (never in a series colour), categorical hues assigned in a fixed order, and a
legend whenever there are two or more series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .cv import EvalResult
from .generator import fraction_calibrated
from .models import MODEL_LABELS
from .permutation import PermutationResult

SURFACE = "#fcfcfb"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua: fixed order

SCENARIO_LABELS = {
    ("kfold", "none"): "Random split (leaky)",
    ("loso", "none"): "Leave-one-subject-out",
    ("loso", "within_group"): "LOSO + within-subject z-score",
}


def _figure(width: float, height: float) -> Figure:
    return Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")


def _style_axes(ax, grid_axis: str | None = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=8, length=3, width=0.75)
    ax.grid(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.75)
        ax.set_axisbelow(True)
    ax.xaxis.label.set_color(SECONDARY)
    ax.yaxis.label.set_color(SECONDARY)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=10.5, fontweight="bold", color=PRIMARY)


def _percent_axis(ax, top: float = 108) -> None:
    ax.set_ylim(0, top)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])


def save_figure(fig: Figure, path: str | Path, dpi: int = 160) -> Path:
    """Save as PNG (or any format matplotlib supports) on the figure surface colour."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    return path


def plot_scheme_comparison(
    table: pd.DataFrame,
    *,
    chance: float | None = None,
    title: str = "Accuracy depends on how you validate",
) -> Figure:
    """Grouped bars: models on x, one bar per ``(scheme, normalize)`` scenario."""
    combos = list(dict.fromkeys(zip(table["scheme"], table["normalize"], strict=True)))
    if len(combos) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} scenarios can be drawn")
    models = list(dict.fromkeys(table["model"]))
    fig = _figure(8.6, 4.4)
    ax = fig.subplots()
    n, group_w = len(combos), 0.66
    bar_w = group_w / n
    values = np.zeros((n, len(models)))
    for i, (scheme, norm) in enumerate(combos):
        sub = table[(table["scheme"] == scheme) & (table["normalize"] == norm)]
        acc = sub.set_index("model").loc[models, "mean_accuracy"].to_numpy() * 100
        values[i] = acc
        x = np.arange(len(models)) + (i - (n - 1) / 2) * bar_w
        ax.bar(
            x, acc, width=bar_w * 0.88, color=SERIES[i],
            label=SCENARIO_LABELS.get((scheme, norm), f"{scheme} / {norm}"),
        )  # fmt: skip
    best = int(values[-1].argmax())  # label only the model that the story is about
    for i in range(n):
        x_label = best + (i - (n - 1) / 2) * bar_w
        ax.text(
            x_label, values[i, best] + 1.2, f"{values[i, best]:.0f}%",
            ha="center", va="bottom", fontsize=8, color=SECONDARY,
        )  # fmt: skip
    right = len(models) - 0.5
    if chance is not None:
        right += 0.4
        ax.axhline(chance * 100, color=SECONDARY, lw=0.75)
        ax.text(
            right, chance * 100 + 1.2, f"chance ({chance * 100:.0f}%)",
            ha="right", va="bottom", fontsize=8, color=SECONDARY,
        )  # fmt: skip
    ax.set_xlim(-0.5, right)
    ax.set_xticks(range(len(models)), [MODEL_LABELS.get(m, m) for m in models])
    _percent_axis(ax)
    ax.set_ylabel("Mean per-subject accuracy")
    _style_axes(ax)
    ax.tick_params(axis="x", length=0)
    _title(ax, title)
    fig.legend(loc="outside lower center", ncol=n, frameon=False, fontsize=8)
    return fig


def plot_subject_accuracy(
    result: EvalResult, *, chance: float | None = None, perm: PermutationResult | None = None
) -> Figure:
    """Accuracy of every held-out subject, with the mean and (optionally) the permutation null."""
    per = result.per_group()
    fig = _figure(max(6.8, 0.6 * len(per) + 4.2), 4.2)
    ax = fig.subplots()
    x = np.arange(len(per))
    acc = per["accuracy"].to_numpy() * 100
    ax.bar(x, acc, width=0.62, color=SERIES[0])
    right = len(per) - 0.5 + 2.4  # empty strip on the right, only for the reference labels
    ax.set_xlim(-0.6, right)
    mean = result.mean_accuracy * 100
    ax.hlines(mean, -0.6, len(per) - 0.4, color=PRIMARY, lw=0.9)
    ax.text(
        right, mean + 1.2, f"mean {mean:.0f}%", ha="right", va="bottom", fontsize=8, color=PRIMARY
    )
    if perm is not None:
        p95 = perm.null_p95 * 100
        ax.hlines(p95, -0.6, len(per) - 0.4, color=SECONDARY, lw=0.9, linestyles="--")
        ax.text(
            right, p95 - 1.2, f"null 95th pct {p95:.0f}%",
            ha="right", va="top", fontsize=8, color=SECONDARY,
        )  # fmt: skip
    elif chance is not None:
        ax.hlines(chance * 100, -0.6, len(per) - 0.4, color=SECONDARY, lw=0.9, linestyles="--")
        ax.text(
            right, chance * 100 - 1.2, f"chance {chance * 100:.0f}%",
            ha="right", va="top", fontsize=8, color=SECONDARY,
        )  # fmt: skip
    ax.set_xticks(x, [str(g) for g in per["group"]])
    _percent_axis(ax)
    ax.set_ylabel("Accuracy on the held-out subject")
    _style_axes(ax)
    ax.tick_params(axis="x", length=0)
    _title(ax, "Leave-one-subject-out accuracy, subject by subject")
    return fig


def plot_permutation(perm: PermutationResult) -> Figure:
    """Histogram of the permutation null with the observed score."""
    fig = _figure(6.6, 3.8)
    ax = fig.subplots()
    ax.hist(perm.null * 100, bins=24, color=SERIES[0], alpha=0.35, lw=0)
    ax.axvline(perm.observed * 100, color=SERIES[1], lw=1.6)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.12)
    ax.text(
        perm.observed * 100, top * 1.1,
        f" observed {perm.observed * 100:.0f}%  (p = {perm.p_value:.3f})",
        ha="right" if perm.observed * 100 > np.median(perm.null * 100) + 15 else "left",
        va="top", fontsize=8, color=PRIMARY,
    )  # fmt: skip
    ax.set_xlabel(
        "Score under shuffled labels (%)" if perm.metric == "mean_accuracy"
        else "Balanced accuracy under shuffled labels (%)"
    )  # fmt: skip
    ax.set_ylabel("Permutations")
    _style_axes(ax)
    _title(ax, f"Permutation null ({perm.n_perm} shuffles inside each subject)")
    return fig


_POWER_COLUMNS = {
    "calibrated": ("power_calibrated", "Power (above the simulated-null 95th pct)"),
    "ttest": ("power_ttest", "Power (one-sided t-test, p < 0.05)"),
    "binom": ("power_binom", "Power (binomial test, p < 0.05)"),
}


def plot_power(
    summary: pd.DataFrame,
    *,
    test: str = "calibrated",
    target: float = 0.8,
    title: str | None = None,
) -> Figure:
    """Power versus number of subjects; one line per trials-per-class, one panel per model.

    ``test`` picks the decision rule: ``"calibrated"`` (default), ``"ttest"`` or ``"binom"``.
    """
    if test not in _POWER_COLUMNS:
        raise ValueError(f"test must be one of {sorted(_POWER_COLUMNS)}")
    column, ylabel = _POWER_COLUMNS[test]
    if column not in summary.columns:
        raise ValueError(f"the summary has no {column!r} column (was calibrate=False used?)")
    models = list(dict.fromkeys(summary["model"]))
    trials = sorted(summary["trials_per_level"].unique())
    if len(trials) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} lines can be drawn")
    fig = _figure(4.6 * len(models) + 0.6, 4.2)
    axes = np.atleast_1d(fig.subplots(1, len(models), sharey=True))
    for ax, model in zip(axes, models, strict=True):
        sub = summary[summary["model"] == model]
        for i, t in enumerate(trials):
            line = sub[sub["trials_per_level"] == t].sort_values("n_subjects")
            ax.plot(
                line["n_subjects"], line[column] * 100, color=SERIES[i], lw=1.8, marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
            )  # fmt: skip
        ax.axhline(target * 100, color=SECONDARY, lw=0.75, linestyle="--")
        ax.text(
            sub["n_subjects"].max(), target * 100 - 2, f"{target:.0%} power",
            ha="right", va="top", fontsize=8, color=SECONDARY,
        )  # fmt: skip
        ax.set_xticks(sorted(sub["n_subjects"].unique()))
        ax.set_xlabel("Subjects in the study")
        _percent_axis(ax)
        _style_axes(ax)
        _title(ax, title if (title and len(models) == 1) else MODEL_LABELS.get(model, model))
    if title and len(models) > 1:
        fig.suptitle(title, x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=PRIMARY)
    axes[0].set_ylabel(ylabel, fontsize=8)
    handles = [Line2D([0], [0], color=SERIES[i], lw=1.8, marker="o") for i in range(len(trials))]
    fig.legend(
        handles, [f"{t} trials per class" for t in trials],
        loc="outside lower center", ncol=len(trials), frameon=False, fontsize=8,
    )  # fmt: skip
    return fig


def plot_calibration(calibration: pd.DataFrame) -> Figure:
    """Percentile of each real subject among simulated ones, per feature (leave-one-out)."""
    long = calibration.melt(id_vars=["held_out", "k_train"], var_name="feature", value_name="pct")
    features = list(dict.fromkeys(long["feature"]))
    fig = _figure(max(6.4, 0.9 * len(features) + 2.5), 4.2)
    ax = fig.subplots()
    ax.axhspan(5, 95, color=SERIES[0], alpha=0.10, lw=0)
    rng = np.random.default_rng(0)
    for i, name in enumerate(features):
        vals = long.loc[long["feature"] == name, "pct"].to_numpy(dtype=float)
        ax.plot(
            i + rng.uniform(-0.12, 0.12, size=len(vals)), vals, linestyle="none", marker="o",
            markersize=6, color=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=1.0,
        )  # fmt: skip
    ax.set_xticks(range(len(features)), features)
    ax.set_ylim(-4, 104)
    ax.set_yticks(range(0, 101, 25))
    ax.set_xlim(-0.6, len(features) - 0.4)
    frac = fraction_calibrated(calibration)
    ax.text(
        len(features) - 0.45, 50, f"{frac:.0%} of the points\ninside the 5-95 band",
        ha="right", va="center", fontsize=8, color=SECONDARY,
    )  # fmt: skip
    ax.set_ylabel("Percentile of the real subject")
    _style_axes(ax)
    ax.tick_params(axis="x", length=0)
    _title(ax, "Generator calibration (about 90% should fall inside the band)")
    return fig
