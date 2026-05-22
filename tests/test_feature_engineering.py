"""Tests for feature engineering configuration and processed parquet contracts."""

import json

import pandas as pd
import pytest

from src import config

# ---------------------------------------------------------------------------
# Configuration sanity tests
# ---------------------------------------------------------------------------


def test_feature_component_counts_are_positive() -> None:
    """Configured PCA dimensions and return windows must be valid."""
    assert config.NLP_PCA_COMPONENTS > 0
    assert config.CV_PCA_COMPONENTS > 0
    assert len(config.RETURN_PERIODS) >= 1


def test_temporal_boundaries_are_ordered() -> None:
    """Train, validation, and test ranges must be strictly ordered."""
    train_end = pd.Timestamp(config.TRAIN_END)
    val_start = pd.Timestamp(config.VAL_START)
    val_end = pd.Timestamp(config.VAL_END)
    test_start = pd.Timestamp(config.TEST_START)

    assert train_end < val_start, "Train end must be before val start"
    assert val_start <= val_end, "Val start must not be after val end"
    assert val_end < test_start, "Val end must be before test start"


def test_no_temporal_overlap_between_splits() -> None:
    """Train, val, and test date intervals must be mutually exclusive."""
    train_end = pd.Timestamp(config.TRAIN_END)
    val_start = pd.Timestamp(config.VAL_START)
    val_end = pd.Timestamp(config.VAL_END)
    test_start = pd.Timestamp(config.TEST_START)

    assert val_start > train_end, "Val overlaps with train"
    assert test_start > val_end, "Test overlaps with val"


def test_ticker_universe_is_non_empty() -> None:
    """The ticker list must contain at least the configured sectors."""
    assert len(config.TICKERS_ALL) >= 50
    for ticker in config.TICKERS_TECH[:3]:
        assert ticker in config.TICKERS_ALL
    for ticker in config.TICKERS_FINANCE[:3]:
        assert ticker in config.TICKERS_ALL


def test_sector_map_covers_all_tickers() -> None:
    """Every ticker in TICKERS_ALL must have a sector assignment."""
    missing = [t for t in config.TICKERS_ALL if t not in config.TICKER_SECTOR_MAP]
    assert missing == [], f"Tickers missing from TICKER_SECTOR_MAP: {missing}"


# ---------------------------------------------------------------------------
# Market features parquet contract
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def market_df() -> pd.DataFrame:
    if not config.FEATURES_MARKET_PATH.exists():
        pytest.skip("features_market.parquet not present (run pipeline first)")
    return pd.read_parquet(config.FEATURES_MARKET_PATH)


def test_market_parquet_has_target_column(market_df: pd.DataFrame) -> None:
    """Market features must include the binary UP/DOWN target."""
    assert "target" in market_df.columns, "target column missing from market features"


def test_market_parquet_has_core_features(market_df: pd.DataFrame) -> None:
    """Essential technical indicators must be present in the feature matrix."""
    required = ["rsi_14", "macd", "volatility_20d", "volume_ratio", "vix_level"]
    missing = [c for c in required if c not in market_df.columns]
    assert missing == [], f"Missing market feature columns: {missing}"


def test_market_parquet_has_cyclical_features(market_df: pd.DataFrame) -> None:
    """Day-of-week and month cyclical encodings must be present."""
    for col in ["dow_sin", "dow_cos", "month_sin", "month_cos"]:
        assert col in market_df.columns, f"Cyclical feature {col!r} missing"


def test_market_parquet_no_nan_in_features(market_df: pd.DataFrame) -> None:
    """Feature columns (not target) must have no NaN values after preprocessing."""
    feature_cols = [c for c in market_df.columns
                    if c not in ("ticker", "sector", "target", "close")]
    nan_counts = market_df[feature_cols].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    assert cols_with_nan.empty, f"NaN values in features: {cols_with_nan.to_dict()}"


def test_market_parquet_target_is_binary(market_df: pd.DataFrame) -> None:
    """Target values must be exactly UP/DOWN (or 0/1) — no other classes."""
    valid = market_df["target"].dropna()
    unique_labels = set(valid.unique())
    assert unique_labels.issubset({"UP", "DOWN", 0, 1}), (
        f"Unexpected target labels: {unique_labels}"
    )


def test_market_parquet_ticker_count(market_df: pd.DataFrame) -> None:
    """Feature matrix must cover a meaningful number of tickers."""
    n = market_df["ticker"].nunique()
    assert n >= 3, f"Expected ≥3 tickers, got {n}"


# ---------------------------------------------------------------------------
# NLP features parquet contract
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nlp_df() -> pd.DataFrame:
    if not config.FEATURES_NLP_PATH.exists():
        pytest.skip("features_nlp.parquet not present (run pipeline first)")
    return pd.read_parquet(config.FEATURES_NLP_PATH)


def test_nlp_parquet_has_pca_columns(nlp_df: pd.DataFrame) -> None:
    """NLP parquet must contain exactly NLP_PCA_COMPONENTS FinBERT embedding dims."""
    pca_cols = [c for c in nlp_df.columns if c.startswith("finbert_embed_pca_")]
    assert len(pca_cols) == config.NLP_PCA_COMPONENTS, (
        f"Expected {config.NLP_PCA_COMPONENTS} PCA cols, found {len(pca_cols)}"
    )


def test_nlp_parquet_has_sentiment_columns(nlp_df: pd.DataFrame) -> None:
    """Primary sentiment features must be present."""
    for col in ["finbert_sentiment", "finbert_confidence", "vader_sentiment", "is_sentiment_imputed"]:
        assert col in nlp_df.columns, f"NLP column {col!r} missing"


def test_nlp_sentiment_range(nlp_df: pd.DataFrame) -> None:
    """FinBERT and VADER compound scores must stay in [-1, 1]."""
    for col in ["finbert_sentiment", "vader_sentiment"]:
        if col in nlp_df.columns:
            assert nlp_df[col].between(-1, 1).all(), (
                f"{col} has values outside [-1, 1]: "
                f"min={nlp_df[col].min():.3f}, max={nlp_df[col].max():.3f}"
            )


def test_nlp_confidence_range(nlp_df: pd.DataFrame) -> None:
    """FinBERT confidence (max softmax probability) must be in [0, 1]."""
    if "finbert_confidence" in nlp_df.columns:
        assert nlp_df["finbert_confidence"].between(0, 1).all(), (
            f"finbert_confidence outside [0, 1]: "
            f"min={nlp_df['finbert_confidence'].min():.3f}"
        )


def test_nlp_has_dynamic_features(nlp_df: pd.DataFrame) -> None:
    """Dynamic NLP features (momentum, surprise, z-score) must be present."""
    for col in ["sentiment_momentum", "sentiment_shift_3d", "sentiment_surprise",
                "news_volume_zscore", "sentiment_x_volume"]:
        assert col in nlp_df.columns, f"Dynamic NLP feature {col!r} missing"


# ---------------------------------------------------------------------------
# CV features parquet contract
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cv_df() -> pd.DataFrame:
    if not config.FEATURES_CV_PATH.exists():
        pytest.skip("features_cv.parquet not present (run pipeline first)")
    return pd.read_parquet(config.FEATURES_CV_PATH)


def test_cv_parquet_has_pca_columns(cv_df: pd.DataFrame) -> None:
    """CV parquet must contain exactly CV_PCA_COMPONENTS chart embedding dims."""
    pca_cols = [c for c in cv_df.columns if c.startswith("chart_embed_pca_")]
    assert len(pca_cols) == config.CV_PCA_COMPONENTS, (
        f"Expected {config.CV_PCA_COMPONENTS} PCA cols, found {len(pca_cols)}"
    )


def test_cv_parquet_has_availability_flag(cv_df: pd.DataFrame) -> None:
    """chart_available must be a binary flag (0 or 1)."""
    assert "chart_available" in cv_df.columns, "chart_available column missing"
    unique_vals = set(cv_df["chart_available"].unique())
    assert unique_vals.issubset({0, 1}), f"chart_available has non-binary values: {unique_vals}"


def test_cv_parquet_column_count(cv_df: pd.DataFrame) -> None:
    """CV feature parquet must have exactly ticker + availability + PCA cols."""
    expected = 1 + 1 + config.CV_PCA_COMPONENTS  # ticker + chart_available + 10 PCA
    assert len(cv_df.columns) == expected, (
        f"Expected {expected} CV columns, got {len(cv_df.columns)}: {list(cv_df.columns)}"
    )


# ---------------------------------------------------------------------------
# Ablation results contract
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ablation() -> dict:
    ablation_path = config.PROCESSED_DIR / "ablation_results.json"
    if not ablation_path.exists():
        pytest.skip("ablation_results.json not present (run train_ml first)")
    with open(ablation_path) as f:
        return json.load(f)


def test_ablation_has_all_configs(ablation: dict) -> None:
    """Ablation results must include configs A, B, C, and D."""
    for cfg in ("A", "B", "C", "D"):
        assert cfg in ablation, f"Config {cfg!r} missing from ablation results"


def test_ablation_f1_values_in_range(ablation: dict) -> None:
    """All test F1-macro values must be in [0, 1]."""
    for cfg, result in ablation.items():
        f1 = result["test_f1_macro"]
        assert 0 <= f1 <= 1, f"Config {cfg}: test_f1_macro={f1} out of [0, 1]"


def test_ablation_feature_counts_increase(ablation: dict) -> None:
    """Config C must have more features than B, which must have more than A."""
    n_a = ablation["A"]["n_features"]
    n_b = ablation["B"]["n_features"]
    n_c = ablation["C"]["n_features"]
    assert n_a < n_b, f"Config B should have more features than A: {n_b} vs {n_a}"
    assert n_b < n_c, f"Config C should have more features than B: {n_c} vs {n_b}"


def test_ablation_best_model_is_known(ablation: dict) -> None:
    """Best model for each config must be one of the trained model types."""
    known_models = {"LightGBM", "RandomForest", "Stacking"}
    for cfg, result in ablation.items():
        bm = result["best_model"]
        assert bm in known_models, f"Config {cfg}: unknown best_model={bm!r}"


def test_ablation_selection_metric_is_val_based(ablation: dict) -> None:
    """The selection_metric field must indicate validation-only model selection."""
    for cfg, result in ablation.items():
        if "selection_metric" in result:
            assert "val" in result["selection_metric"].lower(), (
                f"Config {cfg}: selection_metric {result['selection_metric']!r} "
                "does not appear to be validation-based"
            )
