"""Feature-block boundaries used by the audited A/B/C ablations."""

from src.models.train_ml import load_combined_features


def test_config_b_excludes_exploratory_analyst_features() -> None:
    df = load_combined_features("B")
    forbidden = [c for c in df.columns if c.startswith("analyst_") or c == "price_target_upside"]
    assert forbidden == []
