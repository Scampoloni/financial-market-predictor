"""Smoke tests for feature engineering configuration."""

import pandas as pd

from src import config


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

	assert train_end < val_start
	assert val_start <= val_end
	assert val_end < test_start
