"""
vader_sentiment.py — VADER sentiment baseline using NLTK.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based
sentiment model tuned for social media / news text. It requires no GPU and
runs ~100k headlines/sec — making it an ideal fast baseline against FinBERT.

Usage:
    from src.nlp.vader_sentiment import VaderPipeline
    pipe = VaderPipeline()
    results = pipe.score(["Apple beats earnings estimates", "Tesla recalls vehicles"])
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class VaderPipeline:
    """Wraps NLTK VADER for batch sentiment scoring of news headlines.

    Downloads the VADER lexicon on first use if not already present.
    """

    def __init__(self) -> None:
        """Initialize VADER, downloading lexicon if needed."""
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            self._sia = SentimentIntensityAnalyzer()
        except LookupError:
            logger.info("Downloading VADER lexicon ...")
            nltk.download("vader_lexicon", quiet=True)
            self._sia = SentimentIntensityAnalyzer()

        logger.info("VaderPipeline initialized.")

    def score(self, texts: list[str]) -> pd.DataFrame:
        """Score a list of texts with VADER.

        VADER returns four scores:
          - neg: proportion of text with negative valence
          - neu: proportion of text with neutral valence
          - pos: proportion of text with positive valence
          - compound: normalized composite score in [-1, +1]
            > +0.05  → positive
            < -0.05  → negative
            otherwise → neutral

        Args:
            texts: List of headline strings.

        Returns:
            DataFrame with columns: text, vader_compound, vader_pos,
            vader_neg, vader_neu, vader_label.
        """
        rows = []
        for text in texts:
            scores = self._sia.polarity_scores(text or "")
            compound = scores["compound"]
            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"
            rows.append({
                "text": text,
                "vader_compound": compound,
                "vader_pos": scores["pos"],
                "vader_neg": scores["neg"],
                "vader_neu": scores["neu"],
                "vader_label": label,
            })
        return pd.DataFrame(rows)

    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "title",
    ) -> pd.DataFrame:
        """Run VADER on a DataFrame column and join results back.

        Args:
            df: Input DataFrame.
            text_col: Column name with headline text.

        Returns:
            Original DataFrame with VADER columns added.
        """
        texts = df[text_col].fillna("").tolist()
        scores = self.score(texts).drop(columns=["text"])
        return pd.concat([df.reset_index(drop=True), scores], axis=1)


def score_news_file(ticker: str, news_dir: Path) -> pd.DataFrame:
    """Score all headlines for a ticker using VADER.

    Args:
        ticker: Ticker symbol.
        news_dir: Directory containing {ticker}.parquet news files.

    Returns:
        DataFrame with VADER scores. Empty if no news file found.
    """
    news_path = news_dir / f"{ticker}.parquet"
    if not news_path.exists():
        logger.warning("%s: no news file found", ticker)
        return pd.DataFrame()

    news_df = pd.read_parquet(news_path)
    if news_df.empty:
        return pd.DataFrame()

    pipe = VaderPipeline()
    scored = pipe.score_dataframe(news_df, text_col="title")
    scored["ticker"] = ticker
    logger.info("%s: VADER scored %d headlines", ticker, len(scored))
    return scored


def compare_finbert_vader(
    finbert_df: pd.DataFrame,
    vader_df: pd.DataFrame,
    n_examples: int = 10,
) -> pd.DataFrame:
    """Compare FinBERT and VADER labels on the same headlines.

    Args:
        finbert_df: DataFrame with columns [text/title, finbert_label, finbert_score].
        vader_df: DataFrame with columns [text/title, vader_label, vader_compound].
        n_examples: How many disagreement examples to return.

    Returns:
        DataFrame of cases where FinBERT and VADER disagree, sorted by
        absolute score difference (largest disagreements first).
    """
    text_col = "title" if "title" in finbert_df.columns else "text"
    merged = pd.merge(
        finbert_df[[text_col, "finbert_label", "finbert_score"]],
        vader_df[[text_col, "vader_label", "vader_compound"]],
        on=text_col,
        how="inner",
    )
    merged["agree"] = merged["finbert_label"] == merged["vader_label"]
    merged["score_diff"] = (merged["finbert_score"] - merged["vader_compound"]).abs()
    disagreements = merged[~merged["agree"]].sort_values("score_diff", ascending=False)
    return disagreements.head(n_examples)
