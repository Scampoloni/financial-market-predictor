"""Integrity checks for the audited A/B rerun artefacts."""

import json

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.config import PROCESSED_DIR


def test_audited_rerun_predictions_reproduce_saved_metrics() -> None:
    result_path = PROCESSED_DIR / "rerun_purged_ab_results.json"
    prediction_path = PROCESSED_DIR / "rerun_purged_ab_predictions.parquet"
    results = json.loads(result_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(prediction_path)

    assert set(predictions["config"]) == {"A", "B"}
    for config, rows in predictions.groupby("config"):
        stored = results[config]["selected_model_reporting_metrics"]
        assert len(rows) == results[config]["sample_sizes"]["reporting"]
        assert round(f1_score(rows["actual"], rows["predicted"], average="macro"), 6) == stored["macro_f1"]
        assert round(accuracy_score(rows["actual"], rows["predicted"]), 6) == stored["accuracy"]
        assert round(balanced_accuracy_score(rows["actual"], rows["predicted"]), 6) == stored["balanced_accuracy"]
