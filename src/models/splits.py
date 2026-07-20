"""Date-aware splitters for panels with multiple ticker observations per day."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class PurgedDateTimeSeriesSplit(BaseCrossValidator):
    """Expanding time-series splits grouped by date with a forward-label embargo.

    A row's label is based on a future return.  Keeping ``embargo_days`` business
    days between a training fold and its validation fold prevents labels in the
    training fold from using prices in the validation fold.  Grouping by date
    also prevents different tickers from the same market session being split
    across folds.
    """

    def __init__(self, n_splits: int = 5, embargo_days: int = 5) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if embargo_days < 0:
            raise ValueError("embargo_days must be non-negative")
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise TypeError("X must be a pandas object with a DatetimeIndex")
        dates = pd.DatetimeIndex(X.index).normalize()
        unique_dates = dates.unique().sort_values()
        test_size = len(unique_dates) // (self.n_splits + 1)
        if test_size == 0:
            raise ValueError("Not enough unique dates for the requested splits")

        first_test = len(unique_dates) - self.n_splits * test_size
        for fold in range(self.n_splits):
            test_dates = unique_dates[
                first_test + fold * test_size : first_test + (fold + 1) * test_size
            ]
            train_cutoff = test_dates[0] - pd.offsets.BDay(self.embargo_days)
            train_idx = np.flatnonzero(dates < train_cutoff)
            test_idx = np.flatnonzero(dates.isin(test_dates))
            if len(train_idx) == 0 or len(test_idx) == 0:
                raise ValueError("A fold is empty after applying the embargo")
            yield train_idx, test_idx
