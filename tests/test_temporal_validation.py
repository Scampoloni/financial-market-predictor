"""High-value guards for forward-return and panel time validation."""

import pandas as pd

from src.models.splits import PurgedDateTimeSeriesSplit
from src.models.train_ml import _temporal_split


def test_purged_split_keeps_each_date_in_one_fold_and_applies_embargo() -> None:
    dates = pd.date_range("2024-01-02", periods=70, freq="B")
    index = dates.repeat(2)  # two tickers on every market session
    frame = pd.DataFrame({"x": range(len(index))}, index=index)
    splitter = PurgedDateTimeSeriesSplit(n_splits=3, embargo_days=5)

    for train_idx, test_idx in splitter.split(frame):
        train_dates = pd.DatetimeIndex(frame.index[train_idx]).normalize()
        test_dates = pd.DatetimeIndex(frame.index[test_idx]).normalize()
        assert set(train_dates).isdisjoint(set(test_dates))
        assert train_dates.max() < test_dates.min() - pd.offsets.BDay(5)


def test_temporal_split_excludes_rows_with_targets_crossing_boundaries(monkeypatch) -> None:
    import src.models.train_ml as train_ml

    monkeypatch.setattr(train_ml, "TRAIN_END", "2024-06-30")
    monkeypatch.setattr(train_ml, "VAL_START", "2024-07-01")
    monkeypatch.setattr(train_ml, "VAL_END", "2024-12-31")
    monkeypatch.setattr(train_ml, "TEST_START", "2025-01-01")
    monkeypatch.setattr(train_ml, "TEST_END", "2025-12-31")
    dates = pd.date_range("2024-06-14", "2026-01-08", freq="B")
    frame = pd.DataFrame({"feature": 1.0, "target": "UP"}, index=dates)

    x_train, _, x_val, _, x_test, _ = _temporal_split(frame, ["feature"])

    assert x_train.index.max() <= pd.Timestamp("2024-06-24")
    assert x_val.index.max() <= pd.Timestamp("2024-12-24")
    assert x_test.index.max() <= pd.Timestamp("2025-12-24")
