"""Evaluation metrics and report figures.

Metrics are computed on the ORIGINAL price scale (currency units), because
that is what a user cares about. A log-scale R2 and RMSLE are reported as well
since the models are trained on log prices and price errors are relative.

matplotlib is imported lazily inside the plotting functions, so importing this
module for metrics only (e.g. from the app) does not require it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ACCENT = "#7c6cf5"
MUTED = "#9aa0b4"
GOOD = "#2fa37a"


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Metrics on the price scale, plus relative and log-scale views."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    abs_pct = np.abs(err) / y_true

    def r2(a, b):
        ss_res = np.sum((a - b) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        return float(1 - ss_res / ss_tot)

    log_true, log_pred = np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape_pct": float(np.mean(abs_pct) * 100),
        "median_ape_pct": float(np.median(abs_pct) * 100),
        "r2": r2(y_true, y_pred),
        "r2_log": r2(log_true, log_pred),
        "rmsle": float(np.sqrt(np.mean((log_true - log_pred) ** 2))),
    }


def interval_coverage(y_true, lower, upper) -> float:
    """Share of true values that fall inside [lower, upper]."""
    y_true, lower, upper = map(np.asarray, (y_true, lower, upper))
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def error_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Median absolute % error and MAE per group. df needs actual/predicted + group."""
    d = df.copy()
    d["ape"] = (d["predicted"] - d["actual"]).abs() / d["actual"] * 100
    d["ae"] = (d["predicted"] - d["actual"]).abs()
    out = d.groupby(group_col).agg(
        n=("actual", "size"), median_ape_pct=("ape", "median"), mae=("ae", "mean")
    )
    return out.sort_values("median_ape_pct").round(2)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )
    return plt


def plot_model_comparison(table: pd.DataFrame, path: Path) -> None:
    """Bar chart of cross-validated error per model (lower is better)."""
    plt = _plt()
    t = table.sort_values("cv_rmsle", ascending=False)
    colors = [GOOD if n == table.iloc[0]["model"] else ACCENT for n in t["model"]]
    colors = [MUTED if "Baseline" in n else c for n, c in zip(t["model"], colors)]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(t["model"], t["cv_rmsle"], color=colors)
    ax.set_xlabel("Cross-validated RMSLE on training data (lower is better)")
    ax.set_title("Model comparison (5-fold CV, tuned)")
    for y, v in enumerate(t["cv_rmsle"]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, t["cv_rmsle"].max() * 1.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_actual_vs_predicted(y_true, y_pred, path: Path, title: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(y_true, y_pred, s=10, alpha=0.5, color=ACCENT, edgecolors="none")
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], color=MUTED, linestyle="--", linewidth=1, label="perfect prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual price (log scale)")
    ax.set_ylabel("Predicted price (log scale)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_residuals(y_true, y_pred, path: Path) -> None:
    """Relative residuals (log scale): should be centred on 0 and roughly symmetric."""
    plt = _plt()
    resid = np.log1p(np.asarray(y_true, float)) - np.log1p(np.clip(np.asarray(y_pred, float), 0, None))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].hist(resid, bins=40, color=ACCENT, alpha=0.85)
    axes[0].axvline(0, color=MUTED, linestyle="--", linewidth=1)
    axes[0].set_xlabel("log(actual) - log(predicted)")
    axes[0].set_ylabel("Cars")
    axes[0].set_title("Residual distribution (test set)")
    axes[1].scatter(np.log1p(y_pred), resid, s=8, alpha=0.45, color=ACCENT, edgecolors="none")
    axes[1].axhline(0, color=MUTED, linestyle="--", linewidth=1)
    axes[1].set_xlabel("log(predicted price)")
    axes[1].set_ylabel("residual")
    axes[1].set_title("Residuals vs prediction")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_feature_importance(imp: pd.DataFrame, path: Path) -> None:
    """Permutation importance: how much the error grows when a feature is shuffled."""
    plt = _plt()
    imp = imp.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"], color=ACCENT, alpha=0.9)
    ax.set_xlabel("Increase in RMSLE when the feature is shuffled")
    ax.set_title("Permutation importance (test set)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_error_by_group(table: pd.DataFrame, path: Path, group_label: str) -> None:
    plt = _plt()
    t = table.sort_values("median_ape_pct")
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.barh(t.index.astype(str), t["median_ape_pct"], color=ACCENT, alpha=0.9)
    ax.set_xlabel("Median absolute % error (test set)")
    ax.set_title(f"Error by {group_label}")
    for y, (v, n) in enumerate(zip(t["median_ape_pct"], t["n"])):
        ax.text(v + 0.2, y, f"{v:.1f}%  (n={n})", va="center", fontsize=8.5)
    ax.set_xlim(0, t["median_ape_pct"].max() * 1.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
