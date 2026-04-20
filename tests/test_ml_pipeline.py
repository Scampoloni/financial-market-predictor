"""Smoke tests for ML training and inference contracts."""

from src import config
from src.models.predict import LivePredictor


def test_model_output_paths_are_configured() -> None:
	"""Expected model artifact locations should be configured in one place."""
	assert str(config.STACKING_MODEL_PATH).endswith("stacking_final.pkl")
	assert str(config.MODEL_21D_PATH).endswith("model_21d.pkl")


def test_predictor_exposes_known_horizons() -> None:
	"""Predictor should at least expose horizon checks for 5d and 21d."""
	predictor = LivePredictor()
	assert predictor.has_model(5) in (True, False)
	assert predictor.has_model(21) in (True, False)
