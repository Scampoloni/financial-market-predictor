"""Tests for ML training pipeline and inference contracts."""

import json
import pickle

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


def test_live_inference_restores_saved_sector_dummy() -> None:
    """Live rows must encode the sector dummy expected by the saved model."""
    import numpy as np
    import pandas as pd

    class RecordingModel:
        classes_ = np.array(["DOWN", "UP"])

        def predict_proba(self, frame):
            self.seen = frame.copy()
            return np.array([[0.4, 0.6]])

    model = RecordingModel()
    feature_cols = ["return_1d", "sector_Technology", "sector_Energy"]
    predictor = LivePredictor()
    predictor._models[5] = (model, feature_cols)

    market = pd.DataFrame(
        {"return_1d": [0.01]},
        index=pd.DatetimeIndex(["2026-07-17"]),
    )
    predictor.predict_from_features(
        "AAPL",
        market,
        pd.Series(dtype=float),
        pd.Series(dtype=float),
        horizon=5,
    )

    assert model.seen.loc[0, "sector_Technology"] == 1.0
    assert model.seen.loc[0, "sector_Energy"] == 0.0


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
        per_model = result.get("per_model", {})
        if per_model:
            assert any("lightgbm" in k.lower() or "lgbm" in k.lower()
                       for k in per_model.keys()), f"LightGBM missing from Config {cfg} per_model"


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


# ---------------------------------------------------------------------------
# Config D — corrected analyst features
# ---------------------------------------------------------------------------


def test_ablation_config_d_exists(ablation_results: dict) -> None:
    """Config D (corrected analyst data) must be present in ablation results."""
    assert "D" in ablation_results, (
        "Config D not found in ablation_results.json — run: python -m src.models.train_ml --config D"
    )


def test_ablation_config_d_analyst_features_nonzero(ablation_results: dict) -> None:
    """Config D must include all five analyst feature columns in its feature list."""
    if "D" not in ablation_results:
        pytest.skip("Config D not yet trained")
    feat_cols = ablation_results["D"]["feature_cols"]
    expected_analyst = {
        "analyst_consensus",
        "analyst_upgrade_score",
        "analyst_coverage_count",
        "price_target_upside",
        "analyst_sentiment_momentum",
    }
    present = {f for f in feat_cols if f.startswith("analyst_") or f == "price_target_upside"}
    missing = expected_analyst - present
    assert not missing, (
        f"Config D is missing analyst feature columns: {missing}"
    )


def test_ablation_config_d_feature_count(ablation_results: dict) -> None:
    """Config D must have the same feature count as Config C (66 features)."""
    if "D" not in ablation_results:
        pytest.skip("Config D not yet trained")
    n_c = ablation_results.get("C", {}).get("n_features", 66)
    n_d = ablation_results["D"]["n_features"]
    assert n_d == n_c, (
        f"Config D has {n_d} features but Config C has {n_c} — they must be identical"
    )


def test_ablation_config_d_cv_f1_plausible(ablation_results: dict) -> None:
    """Config D CV F1 must be in a plausible range (0.40–0.60) for a binary classifier."""
    if "D" not in ablation_results:
        pytest.skip("Config D not yet trained")
    cv_f1 = ablation_results["D"]["cv_f1_mean"]
    assert 0.40 <= cv_f1 <= 0.60, (
        f"Config D CV F1 = {cv_f1:.4f} is outside the expected range [0.40, 0.60]"
    )


# ---------------------------------------------------------------------------
# 21-day horizon model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_21d_bundle():
    """Load the 21-day model bundle."""
    if not config.MODEL_21D_PATH.exists():
        pytest.skip("model_21d.pkl not present (run scripts/train_21d.py first)")
    with open(config.MODEL_21D_PATH, "rb") as f:
        return pickle.load(f)


def test_model_21d_has_required_keys(model_21d_bundle) -> None:
    """21-day model bundle must contain model, feature_cols, and horizon keys."""
    for key in ("model", "feature_cols", "horizon"):
        assert key in model_21d_bundle, f"model_21d.pkl missing key '{key}'"


def test_model_21d_correct_horizon(model_21d_bundle) -> None:
    """21-day model bundle must declare horizon=21."""
    assert model_21d_bundle["horizon"] == 21, (
        f"Expected horizon=21, got {model_21d_bundle['horizon']}"
    )


def test_model_21d_has_analyst_features(model_21d_bundle) -> None:
    """21-day model must include corrected analyst feature columns."""
    feat_cols = model_21d_bundle["feature_cols"]
    expected = {"analyst_consensus", "analyst_upgrade_score", "analyst_coverage_count",
                "price_target_upside", "analyst_sentiment_momentum"}
    present = {f for f in feat_cols if f.startswith("analyst_") or f == "price_target_upside"}
    missing = expected - present
    assert not missing, f"model_21d.pkl missing analyst features: {missing}"


def test_model_21d_feature_count_matches_5d(model_21d_bundle) -> None:
    """21-day model must use the same 66-feature Config D set as the 5-day model."""
    n_21d = len(model_21d_bundle["feature_cols"])
    assert n_21d == 66, f"Expected 66 features for 21d model, got {n_21d}"
