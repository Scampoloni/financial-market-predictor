"""model_analysis.py — Ablation results and feature importance page."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

from src.app.utils import load_ablation_results
from src.config import TARGET_CLASSES, CV_FOLDS


def _block_color(feat: str) -> str:
    if any(k in feat for k in ("finbert", "vader", "news", "headline", "sentiment")):
        return "#ff7f0e"
    if "chart" in feat:
        return "#2ca02c"
    return "#4e79a7"


def render() -> None:
    st.header("Model Analysis — Ablation Study")
    st.markdown(
        "The ablation study measures the marginal contribution of each feature block "
        "by training the same RandomForest model on progressively richer feature sets."
    )

    results = load_ablation_results()
    if not results:
        st.error("Ablation results not found. Run `python -m src.models.train_ml` first.")
        return

    # --- Summary table ---
    st.subheader("Official Ablation Results")
    config_names = {"A": "Market only", "B": "Market + NLP", "C": "Market + NLP + CV"}
    baseline_f1 = results["A"]["test_f1_macro"]
    rows = []
    for cfg, r in results.items():
        delta = r["test_f1_macro"] - baseline_f1
        rows.append({
            "Config": cfg,
            "Description": config_names.get(cfg, cfg),
            "# Features": r["n_features"],
            "CV F1 (mean ± std)": f"{r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}",
            "Test F1": round(r["test_f1_macro"], 4),
            "Test Acc": round(r["test_accuracy"], 4),
            "Δ vs A": f"{delta:+.4f}" if cfg != "A" else "baseline",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # --- Bar chart ---
    st.subheader("Test F1 by Config")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    cfgs = list(results.keys())
    f1s  = [results[c]["test_f1_macro"] for c in cfgs]
    bar_colors = ["#4e79a7", "#ff7f0e", "#2ca02c"]
    bars = ax.bar([f"Config {c}" for c in cfgs], f1s,
                  color=bar_colors[:len(cfgs)], width=0.5)
    ax.axhline(baseline_f1, color="grey", linestyle="--", linewidth=1, label="Baseline (A)")
    ax.set_ylim(max(0, min(f1s) - 0.02), max(f1s) + 0.02)
    ax.set_ylabel("Macro F1")
    ax.set_title("Ablation — Test Macro F1 (2025 held-out set)")
    for bar, val, cfg in zip(bars, f1s, cfgs):
        delta = val - baseline_f1
        label = f"{val:.4f}" if cfg == "A" else f"{val:.4f}\n({delta:+.4f})"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                label, ha="center", va="bottom", fontsize=9)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # --- Per-class F1 ---
    st.subheader("Per-Class F1 Breakdown")
    per_class_data = []
    for cfg, r in results.items():
        for cls in TARGET_CLASSES:
            per_class_data.append({
                "Config": f"Config {cfg}",
                "Class": cls,
                "F1": r["per_class"][cls]["f1"],
                "Precision": r["per_class"][cls]["precision"],
                "Recall": r["per_class"][cls]["recall"],
            })
    pc_df = pd.DataFrame(per_class_data)
    pivot = pc_df.pivot(index="Class", columns="Config", values="F1").round(4)
    st.dataframe(pivot, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(TARGET_CLASSES))
    width = 0.25
    cfg_list = [f"Config {c}" for c in cfgs]
    cfg_colors = bar_colors[:len(cfgs)]
    for i, (cfg, color) in enumerate(zip(cfg_list, cfg_colors)):
        vals = [pc_df[(pc_df["Config"] == cfg) & (pc_df["Class"] == cls)]["F1"].values[0]
                for cls in TARGET_CLASSES]
        ax2.bar(x + i * width, vals, width, label=cfg, color=color, alpha=0.85)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(TARGET_CLASSES)
    ax2.set_ylabel("F1")
    ax2.set_title("Per-Class F1 — All Configs")
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # --- CV fold F1 ---
    st.subheader("Cross-Validation Fold F1")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    folds = list(range(1, CV_FOLDS + 1))
    styles = {"A": ("o-", "#4e79a7"), "B": ("s-", "#ff7f0e"), "C": ("^-", "#2ca02c")}
    for cfg, r in results.items():
        style, color = styles.get(cfg, ("o-", "grey"))
        ax3.plot(folds, r["fold_f1"], style, color=color,
                 label=f"Config {cfg} (mean={r['cv_f1_mean']:.4f})", linewidth=2)
    ax3.set_xlabel("Fold")
    ax3.set_ylabel("F1 Macro")
    ax3.set_title("CV F1 per Fold (TimeSeriesSplit)")
    ax3.set_xticks(folds)
    ax3.legend()
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # --- Feature importance (Config C) ---
    if "C" in results:
        st.subheader("Feature Importance — Config C (Top 25)")
        feat_cols = results["C"]["feature_cols"]
        st.markdown(
            "Feature importances are estimated from the saved Config C RandomForest model "
            "(mean decrease in impurity). "
            "**Blue** = market, **orange** = NLP, **green** = CV."
        )
        try:
            import pickle
            from src.config import STACKING_MODEL_PATH
            with open(STACKING_MODEL_PATH, "rb") as f:
                saved = pickle.load(f)
            model = saved["model"]
            importances = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
            top25 = importances.head(25)
            colors = [_block_color(f) for f in top25.index]

            fig4, ax4 = plt.subplots(figsize=(10, 7))
            top25.plot(kind="barh", ax=ax4, color=colors, alpha=0.85)
            ax4.invert_yaxis()
            ax4.set_title("Config C — Top 25 Feature Importances")
            ax4.set_xlabel("Mean Decrease in Impurity")
            legend_handles = [
                mpatches.Patch(color="#4e79a7", label="Market"),
                mpatches.Patch(color="#ff7f0e", label="NLP"),
                mpatches.Patch(color="#2ca02c", label="CV"),
            ]
            ax4.legend(handles=legend_handles)
            fig4.tight_layout()
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

            st.dataframe(
                top25.reset_index().rename(columns={"index": "Feature", 0: "Importance"}).head(15),
                hide_index=True,
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"Could not load model for feature importance: {exc}")
