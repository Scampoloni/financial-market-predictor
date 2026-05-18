"""Tests for NLP pipeline components — VADER, FinBERT output contracts, and RAG setup."""

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# VADER pipeline contract
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vader():
    from src.nlp.vader_sentiment import VaderPipeline
    return VaderPipeline()


SAMPLE_HEADLINES = [
    "Apple beats earnings expectations with record revenue",
    "Company announces bankruptcy filing amid debt crisis",
    "Markets open flat ahead of Federal Reserve decision",
    "Strong jobs report pushes S&P 500 to new all-time high",
    "Pharmaceutical giant recalls drug after safety concerns",
]


def test_vader_returns_dataframe(vader) -> None:
    """VADER score() must return a DataFrame."""
    result = vader.score(SAMPLE_HEADLINES)
    assert isinstance(result, pd.DataFrame), f"Expected DataFrame, got {type(result)}"


def test_vader_has_required_columns(vader) -> None:
    """VADER output must include text, vader_compound, and vader_label."""
    result = vader.score(SAMPLE_HEADLINES)
    for col in ["text", "vader_compound", "vader_label"]:
        assert col in result.columns, f"VADER output missing column {col!r}"


def test_vader_compound_in_range(vader) -> None:
    """VADER compound score must be in [-1.0, 1.0]."""
    result = vader.score(SAMPLE_HEADLINES)
    assert result["vader_compound"].between(-1, 1).all(), (
        f"VADER compound score out of range: {result['vader_compound'].describe()}"
    )


def test_vader_label_valid_values(vader) -> None:
    """VADER labels must be one of positive/negative/neutral."""
    result = vader.score(SAMPLE_HEADLINES)
    valid_labels = {"positive", "negative", "neutral"}
    unexpected = set(result["vader_label"].unique()) - valid_labels
    assert not unexpected, f"Unexpected VADER labels: {unexpected}"


def test_vader_positive_headline_scores_positive(vader) -> None:
    """A clearly positive financial headline should yield positive sentiment."""
    result = vader.score(["Company reports record profits and raises dividend"])
    score = result["vader_compound"].iloc[0]
    assert score > 0, f"Expected positive sentiment, got {score:.3f}"


def test_vader_negative_headline_scores_negative(vader) -> None:
    """A clearly negative financial headline should yield negative sentiment."""
    result = vader.score(["Company announces massive layoffs and misses revenue forecast"])
    score = result["vader_compound"].iloc[0]
    assert score < 0, f"Expected negative sentiment, got {score:.3f}"


def test_vader_handles_empty_list(vader) -> None:
    """VADER must handle an empty input list without error."""
    result = vader.score([])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_vader_output_length_matches_input(vader) -> None:
    """Output row count must equal input length."""
    texts = SAMPLE_HEADLINES[:3]
    result = vader.score(texts)
    assert len(result) == len(texts), f"Output length {len(result)} != input length {len(texts)}"


def test_vader_handles_single_text(vader) -> None:
    """VADER must work with a single-element list."""
    result = vader.score(["Earnings surprise beats estimates"])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# FinBERT pipeline contract (offline / schema-only — no model download)
# ---------------------------------------------------------------------------


def test_finbert_pipeline_is_importable() -> None:
    """FinBERT pipeline class must be importable without loading the model."""
    from src.nlp.finbert_sentiment import FinBertPipeline
    assert FinBertPipeline is not None


def test_finbert_pipeline_has_score_method() -> None:
    """FinBERT pipeline must expose a score() method."""
    from src.nlp.finbert_sentiment import FinBertPipeline
    assert callable(getattr(FinBertPipeline, "score", None)), (
        "FinBertPipeline.score() method missing"
    )


# ---------------------------------------------------------------------------
# RAG chatbot contract (schema only — no external API calls)
# ---------------------------------------------------------------------------


def test_rag_chatbot_is_importable() -> None:
    """RAG chatbot module must be importable."""
    from src.nlp import rag_chatbot
    assert rag_chatbot is not None


def test_rag_index_artifact_exists() -> None:
    """Pre-built RAG index must be present for the deployed app to function."""
    from src import config
    rag_index_path = config.PROCESSED_DIR / "rag_index.pkl"
    assert rag_index_path.exists(), (
        f"RAG index not found at {rag_index_path}. "
        "Run the news scraper and NLP pipeline to build it."
    )
