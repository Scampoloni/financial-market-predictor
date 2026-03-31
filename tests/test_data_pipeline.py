"""Smoke tests for data collection pipeline contracts."""

from src import config


def test_raw_data_directories_exist() -> None:
	"""Raw data directories should be available in the repository layout."""
	assert config.RAW_DIR.exists()
	assert config.RAW_MARKET_DIR.exists()
	assert config.RAW_NEWS_DIR.exists()
	assert config.RAW_CHARTS_DIR.exists()


def test_ticker_metadata_exists() -> None:
	"""Ticker metadata file must exist for pipeline reproducibility."""
	assert config.TICKERS_CSV_PATH.exists()
