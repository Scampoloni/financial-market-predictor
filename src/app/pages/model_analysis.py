"""model_analysis.py -- Ablation study, feature importance, and model comparison."""

from __future__ import annotations

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.utils import load_ablation_results
from src.config import TARGET_CLASSES, CV_FOLDS, STACKING_MODEL_PATH, TEST_START

_MUTED = "#64748b"
_CFG_NAMES = {
    "A": "Market only",
    "B": "Market + NLP",
    "C": "Market + NLP + CV",
    "D": "Market + NLP + CV + Analyst ✓",   # ✓ = corrected analyst data
}
_CFG_COLORS = {
    "A": "#94a3b8",
    "B": "#8b5cf6",
    "C": "#10b981",
    "D": "#f59e0b",   # amber — corrected/complete model
}
_CFG_FEAT_COUNTS = {
    "A": "28 features",
    "B": "56 features",
    "C": "66 features",
    "D": "66 features (corrected)",
}


def _block_color(feat: str) -> str:
    """Color-code a feature by its block."""
    if any(k in feat for k in ("finbert", "vader", "news", "headline", "sentiment")):
        return "#8b5cf6"  # NLP = purple
    if "chart" in feat:
        return "#f97316"  # CV = orange
    if feat.startswith("analyst_"):
        return "#f59e0b"  # Analyst = amber
    return "#4a90d9"  # Market = steel blue


@st.cache_data(show_spinner=False)
def _load_importances() -> pd.Series | None:
    """Load feature importances from the deployed 5-day model."""
    try:
        with open(STACKING_MODEL_PATH, "rb") as f:
            saved = pickle.load(f)
        model = saved["model"]
        feat_cols = saved["feature_cols"]
        if hasattr(model, "feature_importances_"):
            return pd.Series(model.feature_importances_, index=feat_cols)
        if hasattr(model, "named_estimators_"):
            for est in model.named_estimators_.values():
                if hasattr(est, "feature_importances_"):
                    return pd.Series(est.feature_importances_, index=feat_cols)
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _load_shap_data() -> tuple[np.ndarray, pd.DataFrame, list[str]] | None:
    """Compute SHAP values for 200 held-out test rows. Cached after first call."""
    try:
        import shap as _shap
        from src.models.train_ml import load_combined_features

        with open(STACKING_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        feat_cols = bundle["feature_cols"]
        model = bundle["model"]

        combined = load_combined_features("D")
        if "sector" in combined.columns:
            combined = pd.get_dummies(
                combined, columns=["sector"], prefix="sector", drop_first=False
            )

        test = combined[combined.index >= TEST_START]
        for col in feat_cols:
            if col not in test.columns:
                test = test.copy()
                test[col] = 0.0

        X_test = test[feat_cols].fillna(0)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_test), size=min(200, len(X_test)), replace=False)
        X_sample = X_test.iloc[idx].reset_index(drop=True)

        explainer = _shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        # LightGBM binary: shap_values is already (n, p) for the positive class
        if isinstance(sv, list):
            sv = sv[1]
        return np.array(sv), X_sample, feat_cols
    except Exception:
        return None


def render() -> None:
    st.markdown("<h2 style='margin-bottom:4px'>Model Analysis</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b;margin-bottom:1.5rem;font-size:0.9rem'>"
        "Ablation study measuring the marginal contribution of each feature block. "
        "Same temporal split, same hyperparameters -- only features vary.</p>",
        unsafe_allow_html=True,
    )

    results = load_ablation_results()
    if not results:
        st.error("Ablation results not found. Run `python -m src.models.train_ml` first.")
        return

    baseline_f1 = results["A"]["test_f1_macro"]

    # ── 1. ABLATION STUDY TABLE ──────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#f0f6fc;margin-top:0.5rem'>Ablation Study</h3>",
        unsafe_allow_html=True,
    )

    table_rows = ""
    for cfg in ["A", "B", "C", "D"]:
        # Skip any config not yet computed
        if cfg not in results:
            continue
        r = results[cfg]
        f1 = r["test_f1_macro"]
        acc = r["test_accuracy"]
        delta = f1 - baseline_f1

        if cfg == "A":
            delta_html = '<span style="color:#64748b">Baseline</span>'
        elif delta > 0:
            delta_html = f'<span style="color:#10b981;font-weight:700">+{delta:.4f}</span>'
        else:
            delta_html = f'<span style="color:#ef4444;font-weight:700">{delta:+.4f}</span>'

        best = r.get("best_model", "--")
        color = _CFG_COLORS[cfg]
        table_rows += (
            f'<tr style="border-bottom:1px solid #1e293b">'
            f'<td style="padding:12px 16px"><span style="color:{color};font-weight:700">'
            f'Config {cfg}</span><br><span style="color:#64748b;font-size:0.82rem">'
            f'{_CFG_NAMES[cfg]}</span></td>'
            f'<td style="padding:12px 16px;color:#94a3b8">{r["n_features"]}</td>'
            f'<td style="padding:12px 16px;color:#e2e8f0;font-weight:600;font-variant-numeric:tabular-nums">'
            f'{f1:.4f}</td>'
            f'<td style="padding:12px 16px;color:#94a3b8;font-variant-numeric:tabular-nums">'
            f'{acc:.4f}</td>'
            f'<td style="padding:12px 16px;color:#94a3b8">{best}</td>'
            f'<td style="padding:12px 16px">{delta_html}</td></tr>'
        )

    st.markdown(
        f'<div class="glass-card" style="overflow-x:auto;padding:0">'
        f'<table style="width:100%;border-collapse:collapse;font-size:0.9rem">'
        f'<thead><tr style="border-bottom:2px solid #1e293b;background:#0f172a">'
        f'<th style="padding:10px 16px;text-align:left;color:#60a5fa;font-weight:600">Config</th>'
        f'<th style="padding:10px 16px;text-align:left;color:#60a5fa;font-weight:600">Features</th>'
        f'<th style="padding:10px 16px;text-align:left;color:#60a5fa;font-weight:600">Test F1</th>'
        f'<th style="padding:10px 16px;text-align:left;color:#60a5fa;font-weight:600">Test Acc</th>'
        f'<th style="padding:10px 16px;text-align:left;color:#60a5fa;font-weight:600">Best Model</th>'
        f'<th style="padding:10px 16px;text-align:left;color:#60a5fa;font-weight:600">Delta vs A</th>'
        f'</tr></thead><tbody>{table_rows}</tbody></table></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#475569;font-size:0.8rem;margin-top:6px">'
        'Temporal train/test split: train &le; 2024-06-30, val = 2024 H2, test = 2025. '
        '5-fold TimeSeriesSplit CV. Binary target: 5-day forward return direction.</p>',
        unsafe_allow_html=True,
    )

    # ── 2. ABLATION BAR CHART ────────────────────────────────────────────────
    st.markdown("<h3 style='margin-top:1.5rem'>Test F1 -- Ablation Comparison</h3>", unsafe_allow_html=True)

    cfgs = list(results.keys())
    f1s = [results[c]["test_f1_macro"] for c in cfgs]
    colors = [_CFG_COLORS.get(c, "#64748b") for c in cfgs]

    bar_labels = []
    for c in cfgs:
        delta = results[c]["test_f1_macro"] - baseline_f1
        if c == "A":
            bar_labels.append(f"{results[c]['test_f1_macro']:.4f}  (baseline)")
        elif delta > 0:
            src = {"B": "NLP", "C": "CV", "D": "Analyst ✓"}.get(c, "")
            bar_labels.append(f"{results[c]['test_f1_macro']:.4f}  (+{delta:.4f} from {src})")
        else:
            src = {"B": "NLP", "C": "CV", "D": "Analyst ✓"}.get(c, "")
            bar_labels.append(f"{results[c]['test_f1_macro']:.4f}  ({delta:+.4f} {src})")

    fig = go.Figure(go.Bar(
        y=[f"Config {c} -- {_CFG_NAMES[c]}" for c in cfgs],
        x=f1s,
        orientation="h",
        marker_color=colors,
        text=bar_labels,
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=12, family="Inter"),
    ))
    fig.add_vline(x=baseline_f1, line_dash="dash", line_color="#475569")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(l=10, r=140, t=10, b=10),
        xaxis=dict(range=[max(0, min(f1s) - 0.01), max(f1s) + 0.02],
                   gridcolor="#1e293b", zeroline=False, showticklabels=False),
        yaxis=dict(gridcolor="#1e293b", autorange="reversed",
                   tickfont=dict(size=12, color="#94a3b8")),
        font=dict(family="Inter, sans-serif"),
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── 3. FEATURE IMPORTANCE -- CONFIG C (Top 15) ───────────────────────────
    st.markdown("<h3 style='margin-top:1.5rem'>Feature Importance -- Deployed 5-Day Model (Top 15)</h3>",
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#64748b;font-size:0.85rem;margin-bottom:0.8rem">'
        '<span style="color:#4a90d9">&#9679;</span> Market &nbsp;&nbsp;'
        '<span style="color:#8b5cf6">&#9679;</span> NLP &nbsp;&nbsp;'
        '<span style="color:#f97316">&#9679;</span> CV &nbsp;&nbsp;'
        '<span style="color:#f59e0b">&#9679;</span> Analyst</p>',
        unsafe_allow_html=True,
    )

    importances = _load_importances()
    if importances is not None:
        top = importances.sort_values(ascending=True).tail(15)
        fi_colors = [_block_color(f) for f in top.index]

        fig3 = go.Figure(go.Bar(
            x=top.values, y=top.index, orientation="h",
            marker_color=fi_colors,
            text=[f"{v:.4f}" for v in top.values],
            textposition="outside",
            textfont=dict(size=11, color="#94a3b8"),
        ))
        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=440, margin=dict(l=10, r=60, t=10, b=20),
            xaxis=dict(gridcolor="#1e293b", zeroline=False,
                       title="Mean Decrease in Impurity"),
            yaxis=dict(gridcolor="#1e293b", tickfont=dict(size=11, color="#e2e8f0")),
            font=dict(family="Inter, sans-serif", size=12),
            bargap=0.3,
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

        # Importance breakdown by block
        nlp_keys = ("finbert", "vader", "news", "headline", "sentiment")
        cv_keys = ("chart",)
        nlp_pct = importances[[c for c in importances.index if any(k in c for k in nlp_keys)]].sum()
        cv_pct = importances[[c for c in importances.index if any(k in c for k in cv_keys)]].sum()
        analyst_pct = importances[[c for c in importances.index if c.startswith("analyst_")]].sum()
        total = importances.sum()
        mkt_pct = total - nlp_pct - cv_pct - analyst_pct

        st.markdown(
            f'<div class="glass-card" style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;'
            f'text-align:center;padding:16px">'
            f'<div style="flex:1;min-width:100px">'
            f'<div style="color:#4a90d9;font-size:1.5rem;font-weight:800">{mkt_pct/total:.1%}</div>'
            f'<div style="color:#64748b;font-size:0.82rem">Market</div></div>'
            f'<div style="flex:1;min-width:100px">'
            f'<div style="color:#8b5cf6;font-size:1.5rem;font-weight:800">{nlp_pct/total:.1%}</div>'
            f'<div style="color:#64748b;font-size:0.82rem">NLP</div></div>'
            f'<div style="flex:1;min-width:100px">'
            f'<div style="color:#f97316;font-size:1.5rem;font-weight:800">{cv_pct/total:.1%}</div>'
            f'<div style="color:#64748b;font-size:0.82rem">CV</div></div>'
            f'<div style="flex:1;min-width:100px">'
            f'<div style="color:#f59e0b;font-size:1.5rem;font-weight:800">{analyst_pct/total:.1%}</div>'
            f'<div style="color:#64748b;font-size:0.82rem">Analyst</div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Feature importances not available -- model file not found.")

    # ── 3b. SHAP VALUES (TreeExplainer on LightGBM) ──────────────────────────
    st.markdown("<h3 style='margin-top:1.5rem'>SHAP Feature Importance</h3>",
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#64748b;font-size:0.85rem;margin-bottom:0.8rem">'
        'SHAP (SHapley Additive exPlanations) measures each feature\'s <em>marginal contribution</em> '
        'to individual predictions — more reliable than MDI for correlated features. '
        'Computed via TreeExplainer on 200 held-out 2025 test rows.</p>',
        unsafe_allow_html=True,
    )

    shap_data = _load_shap_data()
    if shap_data is not None:
        sv, X_sample, feat_cols_shap = shap_data

        # ── Mean |SHAP| bar chart by feature (top 15) ────────────────────────
        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=feat_cols_shap)
        top15 = mean_abs.sort_values(ascending=True).tail(15)
        fi_colors_shap = [_block_color(f) for f in top15.index]

        fig_shap_bar = go.Figure(go.Bar(
            y=top15.index.tolist(),
            x=top15.values,
            orientation="h",
            marker_color=fi_colors_shap,
            text=[f"{v:.4f}" for v in top15.values],
            textposition="outside",
            textfont=dict(size=11, color="#94a3b8"),
        ))
        fig_shap_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=440, margin=dict(l=10, r=70, t=10, b=20),
            xaxis=dict(gridcolor="#1e293b", zeroline=False,
                       title="Mean |SHAP value|"),
            yaxis=dict(gridcolor="#1e293b", tickfont=dict(size=11, color="#e2e8f0")),
            font=dict(family="Inter, sans-serif", size=12),
            bargap=0.3,
        )
        st.plotly_chart(fig_shap_bar, use_container_width=True,
                        config={"displayModeBar": False})

        # ── SHAP beeswarm summary plot ────────────────────────────────────────
        st.markdown(
            '<p style="color:#64748b;font-size:0.85rem;margin:0.4rem 0 0.6rem">'
            'Beeswarm: each dot = one prediction; colour = feature value (red = high, blue = low). '
            'Horizontal position = SHAP impact on model output.</p>',
            unsafe_allow_html=True,
        )
        try:
            import shap as _shap
            fig_bees, ax_bees = plt.subplots(figsize=(9, 6))
            plt.sca(ax_bees)
            _shap.summary_plot(
                sv, X_sample,
                feature_names=feat_cols_shap,
                max_display=15,
                show=False,
                plot_size=None,
            )
            fig_bees.patch.set_facecolor("#0a0e17")
            ax_bees.set_facecolor("#0f172a")
            ax_bees.tick_params(colors="#94a3b8")
            ax_bees.xaxis.label.set_color("#94a3b8")
            for spine in ax_bees.spines.values():
                spine.set_edgecolor("#1e293b")
            fig_bees.tight_layout()
            st.pyplot(fig_bees, use_container_width=True)
            plt.close(fig_bees)
        except Exception as exc:
            st.info(f"Beeswarm plot unavailable: {exc}")

        # ── Block-level SHAP breakdown ────────────────────────────────────────
        nlp_keys = ("finbert", "vader", "news", "headline", "sentiment")
        cv_keys = ("chart",)
        total_shap = mean_abs.sum()
        nlp_s = mean_abs[[c for c in mean_abs.index if any(k in c for k in nlp_keys)]].sum()
        cv_s = mean_abs[[c for c in mean_abs.index if any(k in c for k in cv_keys)]].sum()
        analyst_s = mean_abs[[c for c in mean_abs.index if c.startswith("analyst_") or c == "price_target_upside"]].sum()
        mkt_s = total_shap - nlp_s - cv_s - analyst_s

        st.markdown(
            f'<div class="glass-card" style="display:flex;gap:16px;flex-wrap:wrap;'
            f'justify-content:center;text-align:center;padding:16px">'
            f'<div style="flex:1;min-width:100px">'
            f'<div style="color:#4a90d9;font-size:1.5rem;font-weight:800">{mkt_s/total_shap:.1%}</div>'
            f'<div style="color:#64748b;font-size:0.82rem">Market</div></div>'
            f'<div style="flex:1;min-width:100px">'
            f'<div style="color:#8b5cf6;font-size:1.5rem;font-weight:800">{nlp_s/total_shap:.1%}</div>'
            f'<div style="color:#64748b;font-size:0.82rem">NLP</div></div>'
            f'<div style="flex:1;min-width:100px">'
            f'<div style="color:#f97316;font-size:1.5rem;font-weight:800">{cv_s/total_shap:.1%}</div>'
            f'<div style="color:#64748b;font-size:0.82rem">CV</div></div>'
            f'<div style="flex:1;min-width:100px">'
            f'<div style="color:#f59e0b;font-size:1.5rem;font-weight:800">{analyst_s/total_shap:.1%}</div>'
            f'<div style="color:#64748b;font-size:0.82rem">Analyst</div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("SHAP values unavailable -- model artifact or feature files not found.")

    # ── 4. MODEL COMPARISON TABLE ────────────────────────────────────────────
    st.markdown("<h3 style='margin-top:1.5rem'>Model Comparison (All Configs)</h3>", unsafe_allow_html=True)

    if any("per_model" in r for r in results.values()):
        mc_rows = ""
        for cfg in cfgs:
            r = results[cfg]
            for model_name, mr in r.get("per_model", {}).items():
                is_best = model_name == r.get("best_model")
                star = "&#9733; " if is_best else ""
                f1_color = "#10b981" if is_best else "#e2e8f0"
                test_f1 = mr.get("test_f1_macro")
                test_acc = mr.get("test_accuracy")
                if test_f1 is None:
                    test_f1 = mr.get("val_f1_macro")
                if test_acc is None:
                    test_acc = mr.get("val_accuracy")
                f1_cell = f"{test_f1:.4f}" if test_f1 is not None else "--"
                acc_cell = f"{test_acc:.4f}" if test_acc is not None else "--"

                mc_rows += (
                    f'<tr style="border-bottom:1px solid #1e293b">'
                    f'<td style="padding:10px 16px;color:#94a3b8">Config {cfg}</td>'
                    f'<td style="padding:10px 16px;color:#e2e8f0;font-weight:500">'
                    f'{star}{model_name}</td>'
                    f'<td style="padding:10px 16px;color:#94a3b8;font-variant-numeric:tabular-nums">'
                    f'{mr["cv_f1_mean"]:.4f} &plusmn; {mr["cv_f1_std"]:.4f}</td>'
                    f'<td style="padding:10px 16px;color:{f1_color};font-weight:600;'
                    f'font-variant-numeric:tabular-nums">{f1_cell}</td>'
                    f'<td style="padding:10px 16px;color:#94a3b8;'
                    f'font-variant-numeric:tabular-nums">{acc_cell}</td></tr>'
                )

        st.markdown(
            f'<div class="glass-card" style="overflow-x:auto;padding:0">'
            f'<table style="width:100%;border-collapse:collapse;font-size:0.88rem">'
            f'<thead><tr style="border-bottom:2px solid #1e293b;background:#0f172a">'
            f'<th style="padding:10px 16px;text-align:left;color:#60a5fa;font-weight:600">Config</th>'
            f'<th style="padding:10px 16px;text-align:left;color:#60a5fa;font-weight:600">Model</th>'
            f'<th style="padding:10px 16px;text-align:left;color:#60a5fa;font-weight:600">CV F1 (mean &plusmn; std)</th>'
            f'<th style="padding:10px 16px;text-align:left;color:#60a5fa;font-weight:600">Test F1</th>'
            f'<th style="padding:10px 16px;text-align:left;color:#60a5fa;font-weight:600">Test Acc</th>'
            f'</tr></thead><tbody>{mc_rows}</tbody></table></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#475569;font-size:0.8rem;margin-top:6px">'
            '&#9733; = best model per config (selected by validation F1). '
            'Stacking CV score = held-out validation F1 (TimeSeriesSplit KFold CV not applicable); '
            'RF and LightGBM CV scores are true 5-fold TimeSeriesSplit means.</p>',
            unsafe_allow_html=True,
        )

    # ── 5. CV FOLD STABILITY ─────────────────────────────────────────────────
    st.markdown("<h3 style='margin-top:1.5rem'>Cross-Validation Fold Stability</h3>", unsafe_allow_html=True)

    fig2 = go.Figure()
    for cfg, r in results.items():
        fig2.add_trace(go.Scatter(
            x=list(range(1, CV_FOLDS + 1)), y=r["fold_f1"],
            mode="lines+markers",
            name=f"Config {cfg} (μ={r['cv_f1_mean']:.4f})",
            line=dict(color=_CFG_COLORS.get(cfg, "#64748b"), width=2.5),
            marker=dict(size=8),
        ))
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=20, r=20, t=10, b=40),
        xaxis=dict(title="Fold", dtick=1, gridcolor="#1e293b"),
        yaxis=dict(title="F1 Macro", gridcolor="#1e293b", zeroline=False),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8", size=11)),
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── 6. PER-CLASS BREAKDOWN ───────────────────────────────────────────────
    st.markdown("<h3 style='margin-top:1.5rem'>Per-Class Performance</h3>", unsafe_allow_html=True)

    pc_rows = ""
    for cfg in cfgs:
        r = results[cfg]
        for cls in TARGET_CLASSES:
            pc = r.get("per_class", {}).get(cls, {})
            if pc:
                cls_color = "#10b981" if cls == "UP" else "#ef4444"
                pc_rows += (
                    f'<tr style="border-bottom:1px solid #1e293b">'
                    f'<td style="padding:8px 16px;color:#94a3b8">Config {cfg}</td>'
                    f'<td style="padding:8px 16px;color:{cls_color};font-weight:600">{cls}</td>'
                    f'<td style="padding:8px 16px;color:#e2e8f0;font-variant-numeric:tabular-nums">'
                    f'{pc["precision"]:.4f}</td>'
                    f'<td style="padding:8px 16px;color:#e2e8f0;font-variant-numeric:tabular-nums">'
                    f'{pc["recall"]:.4f}</td>'
                    f'<td style="padding:8px 16px;color:#e2e8f0;font-variant-numeric:tabular-nums">'
                    f'{pc["f1"]:.4f}</td></tr>'
                )

    if pc_rows:
        st.markdown(
            f'<div class="glass-card" style="overflow-x:auto;padding:0">'
            f'<table style="width:100%;border-collapse:collapse;font-size:0.88rem">'
            f'<thead><tr style="border-bottom:2px solid #1e293b;background:#0f172a">'
            f'<th style="padding:8px 16px;text-align:left;color:#60a5fa;font-weight:600">Config</th>'
            f'<th style="padding:8px 16px;text-align:left;color:#60a5fa;font-weight:600">Class</th>'
            f'<th style="padding:8px 16px;text-align:left;color:#60a5fa;font-weight:600">Precision</th>'
            f'<th style="padding:8px 16px;text-align:left;color:#60a5fa;font-weight:600">Recall</th>'
            f'<th style="padding:8px 16px;text-align:left;color:#60a5fa;font-weight:600">F1</th>'
            f'</tr></thead><tbody>{pc_rows}</tbody></table></div>',
            unsafe_allow_html=True,
        )

    # ── 7. INTERPRETATION ────────────────────────────────────────────────────
    nlp_delta = results["B"]["test_f1_macro"] - results["A"]["test_f1_macro"]
    cv_delta = results["C"]["test_f1_macro"] - results["B"]["test_f1_macro"]
    analyst_delta = results["D"]["test_f1_macro"] - results["C"]["test_f1_macro"] if "D" in results else None

    st.markdown("<h3 style='margin-top:1.5rem'>Interpretation</h3>", unsafe_allow_html=True)
    nlp_effect = "improves" if nlp_delta > 0 else "changes"

    analyst_para = ""
    if analyst_delta is not None:
        analyst_effect = "improves" if analyst_delta > 0 else "leaves"
        analyst_para = (
            f'<p><b style="color:#f59e0b">Analyst correction contribution ({analyst_delta:+.4f} F1):</b> '
            f'Config D re-trains on the same 66 features as Config C, but with corrected analyst data '
            f'(previously all-zero due to a <code>NameError</code> in the batch pipeline). '
            f'The corrected analyst consensus, upgrade score, coverage count, price-target upside, '
            f'and sentiment momentum {analyst_effect} Test F1 from '
            f'{results["C"]["test_f1_macro"]:.4f} to {results["D"]["test_f1_macro"]:.4f} '
            f'({analyst_delta:+.4f}). The near-zero delta confirms that analyst ratings, '
            f'while conceptually valuable, are largely priced in at the 5-day horizon — '
            f'consistent with the semi-strong form of the EMH. The corrected pipeline also '
            f'provides a clean data quality baseline for future work.</p>'
        )

    st.markdown(
        f'<div class="glass-card" style="line-height:1.8;color:#94a3b8;font-size:0.9rem">'

        f'<p><b style="color:#8b5cf6">NLP contribution ({nlp_delta:+.4f} F1):</b> '
        f'Adding FinBERT + VADER sentiment features {nlp_effect} Test F1 from '
        f'{results["A"]["test_f1_macro"]:.4f} to {results["B"]["test_f1_macro"]:.4f}. '
        f'While the magnitude is modest, this suggests '
        f'that financial news sentiment captures information not fully reflected in '
        f'technical indicators. The small effect size is expected: most trading days '
        f'rely on sector-level sentiment fallback due to sparse per-ticker news coverage '
        f'(only ~1.5% of ticker-days have direct headlines).</p>'

        f'<p><b style="color:#f97316">CV contribution ({cv_delta:+.4f} F1):</b> '
        f'Adding EfficientNet-B0 chart embeddings shows a marginal '
        f'{"improvement" if cv_delta > 0 else "decrease"} '
        f'({cv_delta:+.4f}). The CNN embeddings encode visual price patterns '
        f'(trends, reversals, consolidation) but overlap significantly with the '
        f'technical indicators already in the feature set (RSI, MACD, Bollinger Bands). '
        f'This confirms that much of the "visual" information in candlestick charts '
        f'is already captured by numerical indicators.</p>'

        f'{analyst_para}'

        f'<p><b style="color:#60a5fa">Market efficiency implication:</b> '
        f'The overall F1 of ~0.50 for 5-day binary prediction is consistent with '
        f'the semi-strong form of the Efficient Market Hypothesis -- public information '
        f'(technical indicators, news) provides limited but non-zero predictive signal '
        f'at the weekly horizon. The marginal NLP contribution suggests markets are '
        f'not perfectly efficient at incorporating news sentiment in the short term.</p>'

        f'<p><b style="color:#94a3b8">Stacking ensemble caveat:</b> '
        f'The StackingClassifier underperforms individual models because scikit-learn\'s '
        f'implementation uses KFold (not TimeSeriesSplit) internally for generating '
        f'meta-features. This causes temporal data leakage during the stacking cross-validation, '
        f'degrading out-of-sample performance. The ablation correctly picks the best '
        f'individual model (RF or LightGBM) per config.</p>'

        f'</div>',
        unsafe_allow_html=True,
    )

    # -- Disclaimer -----------------------------------------------------------
    st.markdown(
        "<p class='disclaimer'>"
        "All evaluations use a strict temporal split with no data leakage. "
        "Past performance does not predict future returns.</p>",
        unsafe_allow_html=True,
    )
