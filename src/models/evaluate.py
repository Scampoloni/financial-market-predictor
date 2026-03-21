"""
evaluate.py — Shared evaluation utilities for the ablation study.

Provides confusion matrix plotting, per-class F1 bar charts, feature
importance visualization, and rolling accuracy analysis. Used by both
the training script and the integrated model notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)


from src.config import TARGET_CLASSES


def plot_confusion_matrices(
    results: dict[str, dict],
    y_test: pd.Series,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Plot row-normalised confusion matrices side-by-side for multiple configs.

    Args:
        results: Dict mapping config label → dict with 'y_pred' key.
        y_test: True labels (Series).
        figsize: Optional figure size override.
    """
    n = len(results)
    figsize = figsize or (6 * n, 5)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (label, r) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, r["y_pred"], labels=TARGET_CLASSES, normalize="true")
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES, ax=ax,
        )
        f1 = f1_score(y_test, r["y_pred"], average="macro")
        ax.set_title(f"{label}\nTest F1={f1:.4f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.show()


def plot_ablation_bar(results: dict[str, float], title: str = "Ablation Study — Test F1 Macro") -> None:
    """Bar chart comparing macro F1 across ablation configs.

    Args:
        results: Dict mapping config label → test F1 value.
        title: Plot title.
    """
    labels = list(results.keys())
    values = list(results.values())
    baseline = values[0]

    colors = ["steelblue"] + [
        "#2ca02c" if v > baseline else "#d62728"
        for v in values[1:]
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="black")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    # Annotate deltas vs baseline
    for i, (bar, val) in enumerate(zip(bars[1:], values[1:]), 1):
        delta = val - baseline
        sign = "+" if delta >= 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{sign}{delta:.4f}",
            ha="center", va="center", fontsize=10, color="white", fontweight="bold",
        )

    ax.axhline(baseline, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylim(max(0, min(values) - 0.05), max(values) + 0.04)
    ax.set_ylabel("Macro F1 (test set)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_per_class_f1(results: dict[str, dict], y_test: pd.Series) -> None:
    """Grouped bar chart of per-class F1 across configs.

    Args:
        results: Dict mapping config label → dict with 'y_pred' key.
        y_test: True labels.
    """
    per_class = {}
    for label, r in results.items():
        report = classification_report(
            y_test, r["y_pred"], labels=TARGET_CLASSES, output_dict=True
        )
        per_class[label] = {cls: report[cls]["f1-score"] for cls in TARGET_CLASSES}

    df = pd.DataFrame(per_class).T  # rows = configs, cols = classes

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.25
    colors = ["#d62728", "#aec7e8", "#2ca02c"]

    for i, (cls, color) in enumerate(zip(TARGET_CLASSES, colors)):
        ax.bar(x + i * width, df[cls], width, label=cls, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df.index)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 by Ablation Config")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_cols: list[str], top_n: int = 20) -> None:
    """Bar chart of XGBoost/RF feature importances.

    Args:
        model: Fitted sklearn model with feature_importances_ attribute.
        feature_cols: Column names corresponding to importances.
        top_n: Number of top features to display.
    """
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(12, 7))
    importances.plot(kind="barh", ax=ax, color="steelblue", alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.show()


def rolling_accuracy(
    y_true: pd.Series,
    y_pred: np.ndarray,
    window: int = 20,
    ax=None,
) -> pd.Series:
    """Compute and optionally plot rolling prediction accuracy.

    Args:
        y_true: True labels with DatetimeIndex.
        y_pred: Predicted labels array.
        window: Rolling window size in days.
        ax: Optional matplotlib axis.

    Returns:
        Rolling accuracy Series.
    """
    correct = pd.Series(
        (y_true.values == y_pred).astype(float),
        index=y_true.index,
    )
    daily = correct.groupby(correct.index).mean()
    rolled = daily.rolling(window, min_periods=1).mean()

    if ax is not None:
        rolled.plot(ax=ax, color="steelblue", label=f"{window}d rolling")
        ax.axhline(correct.mean(), color="red", linestyle="--",
                   label=f"Overall: {correct.mean():.3f}")
        ax.set_ylabel("Accuracy")
        ax.legend()

    return rolled


def ablation_summary_table(results: dict) -> pd.DataFrame:
    """Return a formatted DataFrame summarising ablation results.

    Args:
        results: Output dict from run_ablation().

    Returns:
        DataFrame with one row per config.
    """
    rows = []
    for config, r in results.items():
        rows.append({
            "Config": f"Config {config}",
            "Features": r["n_features"],
            "CV F1 (mean)": f"{r['cv_f1_mean']:.4f}",
            "CV F1 (std)":  f"{r['cv_f1_std']:.4f}",
            "Test F1":      f"{r['test_f1_macro']:.4f}",
            "Test Accuracy": f"{r['test_accuracy']:.4f}",
        })
    return pd.DataFrame(rows).set_index("Config")
