"""Tests for Computer Vision pipeline components — chart generation, CNN, and feature contracts."""

import numpy as np
import pytest

from src import config


# ---------------------------------------------------------------------------
# Chart generation contract
# ---------------------------------------------------------------------------


def test_chart_configuration_is_valid() -> None:
    """Chart image dimensions and window must match EfficientNet-B0 input spec."""
    width, height = config.CHART_IMAGE_SIZE
    assert width == 224 and height == 224, (
        f"EfficientNet-B0 expects 224×224 input, got {config.CHART_IMAGE_SIZE}"
    )
    assert config.CHART_WINDOW_DAYS > 0, "Chart window must cover a positive number of days"


def test_chart_label_threshold_is_positive() -> None:
    """Chart label threshold (for UP/DOWN class assignment) must be > 0."""
    assert config.CHART_LABEL_THRESHOLD > 0, (
        "CHART_LABEL_THRESHOLD must be positive; a zero threshold makes all moves ambiguous"
    )


# ---------------------------------------------------------------------------
# ChartCNN contract (schema / import only — no GPU required)
# ---------------------------------------------------------------------------


def test_chart_cnn_is_importable() -> None:
    """ChartCNN class must be importable without loading weights."""
    from src.cv.chart_classifier import ChartCNN
    assert ChartCNN is not None


def test_chart_cnn_has_embed_batch_method() -> None:
    """ChartCNN must expose embed_batch() for batch image encoding."""
    from src.cv.chart_classifier import ChartCNN
    assert callable(getattr(ChartCNN, "embed_batch", None)), (
        "ChartCNN.embed_batch() method missing"
    )


# ---------------------------------------------------------------------------
# CV features parquet contract
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cv_df():
    if not config.FEATURES_CV_PATH.exists():
        pytest.skip("features_cv.parquet not present (run cv_features pipeline first)")
    import pandas as pd
    return pd.read_parquet(config.FEATURES_CV_PATH)


def test_cv_pca_columns_count(cv_df) -> None:
    """CV feature matrix must have exactly CV_PCA_COMPONENTS embedding dims."""
    pca_cols = [c for c in cv_df.columns if c.startswith("chart_embed_pca_")]
    assert len(pca_cols) == config.CV_PCA_COMPONENTS, (
        f"Expected {config.CV_PCA_COMPONENTS} PCA columns, found {len(pca_cols)}"
    )


def test_cv_pca_columns_are_indexed_correctly(cv_df) -> None:
    """PCA columns must be named chart_embed_pca_1 through chart_embed_pca_N."""
    expected = [f"chart_embed_pca_{i+1}" for i in range(config.CV_PCA_COMPONENTS)]
    for col in expected:
        assert col in cv_df.columns, f"Expected column {col!r} not found in CV parquet"


def test_cv_chart_available_is_binary(cv_df) -> None:
    """chart_available must contain only 0 (no chart) or 1 (chart present)."""
    unique = set(cv_df["chart_available"].unique())
    assert unique.issubset({0, 1}), f"chart_available has non-binary values: {unique}"


def test_cv_pca_values_are_finite(cv_df) -> None:
    """All PCA embedding values must be finite (no NaN or Inf after preprocessing)."""
    pca_cols = [c for c in cv_df.columns if c.startswith("chart_embed_pca_")]
    pca_data = cv_df[pca_cols]
    assert not pca_data.isna().any().any(), "NaN found in CV PCA columns"
    assert np.isfinite(pca_data.values).all(), "Non-finite values found in CV PCA columns"


def test_cv_has_chart_coverage(cv_df) -> None:
    """At least some ticker-days must have chart coverage (chart_available == 1)."""
    coverage = (cv_df["chart_available"] == 1).sum()
    assert coverage > 0, (
        "No charts available (chart_available == 1 nowhere). "
        "Run chart_generator before cv_features."
    )


def test_cv_no_raw_embed_columns_in_output(cv_df) -> None:
    """Raw EfficientNet embedding columns (embed_*) must NOT appear in the output parquet.
    Only the PCA-compressed columns should be stored."""
    raw_cols = [c for c in cv_df.columns if c.startswith("embed_")]
    assert raw_cols == [], (
        f"Raw embedding columns found in CV parquet (should have been dropped): {raw_cols}"
    )


# ---------------------------------------------------------------------------
# PCA artifact tests
# ---------------------------------------------------------------------------


def test_cv_pca_artifact_n_components() -> None:
    """Saved CV PCA must have exactly CV_PCA_COMPONENTS components."""
    import pickle
    if not config.PCA_CV_PATH.exists():
        pytest.skip("CV PCA artifact not present")
    with open(config.PCA_CV_PATH, "rb") as f:
        cache = pickle.load(f)
    pca = cache["pca"]
    assert pca.n_components_ == config.CV_PCA_COMPONENTS, (
        f"Saved PCA has {pca.n_components_} components, expected {config.CV_PCA_COMPONENTS}"
    )


def test_cv_pca_explains_meaningful_variance() -> None:
    """Saved CV PCA must explain at least 10% of variance (non-trivial compression)."""
    import pickle
    if not config.PCA_CV_PATH.exists():
        pytest.skip("CV PCA artifact not present")
    with open(config.PCA_CV_PATH, "rb") as f:
        cache = pickle.load(f)
    pca = cache["pca"]
    total_variance = pca.explained_variance_ratio_.sum()
    assert total_variance >= 0.1, (
        f"CV PCA explains only {total_variance*100:.1f}% variance — expected ≥10%"
    )


def test_cv_pca_scaler_has_correct_input_dim() -> None:
    """CV PCA scaler must have been fitted on 1280-dim EfficientNet-B0 embeddings."""
    import pickle
    if not config.PCA_CV_PATH.exists():
        pytest.skip("CV PCA artifact not present")
    with open(config.PCA_CV_PATH, "rb") as f:
        cache = pickle.load(f)
    scaler = cache["scaler"]
    assert scaler.n_features_in_ == 1280, (
        f"CV scaler was fitted on {scaler.n_features_in_} dims, expected 1280 (EfficientNet-B0)"
    )
