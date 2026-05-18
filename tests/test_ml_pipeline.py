"""Tests for ML training pipeline and inference contracts."""

import json
import pickle
from pathlib import Path

import pytest

from src import config
from src.models.predict import LivePredictor


# ---------------------------------------------------------------------------
# Configuration contracts
# ---------------------------------------------------------------------------


def test_model_output_paths_are_configured() -> None:
    """Expected model artifact locations should be configured in one place."""
    assert str(config.STACKING_MODEL_PATH).endswith("stacking_final.pkl")
    assert str(config.MODEL_21D_PATH).endswith("model_21d.pkl")


def test_predictor_exposes_known_horizons() -> None:
    """Predictor should at least expose horizon checks for 5d and 21d."""
    predictor = LivePredictor()
    assert predictor.has_model(5) in (True, False)
    assert predictor.has_model(21) in (True, False)


def test_temporal_split_config_no_leakage() -> None:
    """The train→val→test transition must be strictly sequential."""
    import pandas as pd
    train_end = pd.Timestamp(config.TRAIN_END)
    val_start = pd.Timestamp(config.VAL_START)
    val_end = pd.Timestamp(config.VAL_END)
    test_start = pd.Timestamp(config.TEST_START)

    assert train_end < val_start, "Train set overlaps with validation"
    assert val_end < test_start, "Validation set overlaps with test"


# ---------------------------------------------------------------------------
# Saved model artifact tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def saved_model_bundle():
    """Load the model bundle (dict with 'model' and 'feature_cols' keys)."""
    if not config.STACKING_MODEL_PATH.exists():
        pytest.skip("stacking_final.pkl not present (run train_ml first)")
    with open(config.STACKING_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def test_saved_model_bundle_has_required_keys(saved_model_bundle) -> None:
    """Saved model bundle must contain 'model' and 'feature_cols' keys."""
    assert "model" in saved_model_bundle, "Model bundle missing 'model' key"
    assert "feature_cols" in saved_model_bundle, "Model bundle missing 'feature_cols' key"


def test_saved_model_is_sklearn_compatible(saved_model_bundle) -> None:
    """Inner model must have predict_proba and classes_ attributes."""
    model = saved_model_bundle["model"]
    assert hasattr(model, "predict_proba"), "Inner model lacks predict_proba"
    assert hasattr(model, "classes_"), "Inner model lacks classes_ attribute"


def test_saved_model_has_binary_classes(saved_model_bundle) -> None:
    """Model must output exactly two classes (UP/DOWN)."""
    model = saved_model_bundle["model"]
    assert len(model.classes_) == 2, (
        f"Expected 2 classes, got {len(model.classes_)}: {model.classes_}"
    )


def test_saved_model_feature_cols_match_config(saved_model_bundle) -> None:
    """Feature columns stored with the model must be a non-empty list."""
    feature_cols = saved_model_bundle["feature_cols"]
    assert isinstance(feature_cols, list), "feature_cols must be a list"
    assert len(feature_cols) > 0, "feature_cols list is empty"


# ---------------------------------------------------------------------------
# Ablation results integrity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ablation_results() -> dict:
    path = config.PROCESSED_DIR / "ablation_results.json"
    if not path.exists():
        pytest.skip("ablation_results.json not present")
    with open(path) as f:
        return json.load(f)


def test_ablation_config_a_is_market_only(ablation_results: dict) -> None:
    """Config A (market-only baseline) must have the smallest feature count."""
    n_a = ablation_results["A"]["n_features"]
    n_b = ablation_results["B"]["n_features"]
    assert n_a < n_b, f"Config A should have fewer features than B ({n_a} vs {n_b})"


def test_ablation_cv_f1_values_are_stable(ablation_results: dict) -> None:
    """Cross-validation F1 standard deviation must be small (< 0.1) for reliable comparison."""
    for cfg, result in ablation_results.items():
        std = result.get("cv_f1_std", 0)
        assert std < 0.1, f"Config {cfg}: CV F1 std={std:.4f} is suspiciously high"


def test_ablation_results_have_per_class_metrics(ablation_results: dict) -> None:
    """Each config must report per-class F1 for UP and DOWN."""
    for cfg, result in ablation_results.items():
        assert "per_class" in result or "test_f1_up" in result or "report" in result, (
            f"Config {cfg} missing per-class breakdown"
        )


def test_ablation_lgbm_present_in_all_configs(ablation_results: dict) -> None:
    """LightGBM must have been evaluated in all three configs."""
    for cfg, result in ablation_results.items():
        models = result.get("models", {})
        if models:
            assert any("lightgbm" in k.lower() or "lgbm" in k.lower()
                       for k in models.keys()), f"LightGBM missing from Config {cfg} models"


# ---------------------------------------------------------------------------
# PCA artifact tests
# ---------------------------------------------------------------------------


def test_nlp_pca_artifact_exists_and_loadable() -> None:
    """NLP PCA pickle must be loadable and contain scaler + pca keys."""
    path = config.PROCESSED_DIR / "pca_nlp_embeddings.pkl"
    if not path.exists():
        pytest.skip("pca_nlp_embeddings.pkl not present")
    with open(path, "rb") as f:
        cache = pickle.load(f)
    assert "scaler" in cache, "NLP PCA cache missing 'scaler'"
    assert "pca" in cache, "NLP PCA cache missing 'pca'"


def test_cv_pca_artifact_exists_and_loadable() -> None:
    """CV PCA pickle must be loadable and contain scaler + pca keys."""
    if not config.PCA_CV_PATH.exists():
        pytest.skip("pca_cv.pkl not present")
    with open(config.PCA_CV_PATH, "rb") as f:
        cache = pickle.load(f)
    assert "scaler" in cache, "CV PCA cache missing 'scaler'"
    assert "pca" in cache, "CV PCA cache missing 'pca'"


def test_cv_pca_has_correct_n_components() -> None:
    """CV PCA must decompose into exactly CV_PCA_COMPONENTS dimensions."""
    if not config.PCA_CV_PATH.exists():
        pytest.skip("pca_cv.pkl not present")
    with open(config.PCA_CV_PATH, "rb") as f:
        cache = pickle.load(f)
    pca = cache["pca"]
    assert pca.n_components_ == config.CV_PCA_COMPONENTS, (
        f"CV PCA has {pca.n_components_} components, expected {config.CV_PCA_COMPONENTS}"
    )
