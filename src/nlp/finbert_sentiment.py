"""
finbert_sentiment.py — FinBERT sentiment pipeline (ProsusAI/finbert).

Runs finance-domain BERT on news headlines and returns:
  - sentiment label (positive / negative / neutral)
  - sentiment score  (-1 to +1, compound)
  - confidence       (max softmax probability)
  - CLS embedding    (768-dim, for PCA features)

Embeddings are cached to disk after first run (slow inference, ~32 headlines/s on CPU).

Usage:
    from src.nlp.finbert_sentiment import FinBertPipeline
    pipe = FinBertPipeline()
    results = pipe.score(["Apple beats earnings estimates", "Tesla recalls vehicles"])
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import (
    FINBERT_BATCH_SIZE,
    FINBERT_MAX_LENGTH,
    FINBERT_MODEL_NAME,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

# Cache path for pre-computed FinBERT scores
FINBERT_CACHE_PATH = PROCESSED_DIR / "finbert_scores_cache.parquet"

# FinBERT label order from ProsusAI/finbert
FINBERT_LABELS = ["positive", "negative", "neutral"]


class FinBertPipeline:
    """Wraps ProsusAI/finbert for batch sentiment inference and embedding extraction.

    Lazy-loads the model on first use. Handles CPU and CUDA automatically.
    """

    def __init__(self, model_name: str = FINBERT_MODEL_NAME, device: str | None = None):
        """Initialize the pipeline (model not loaded until first call).

        Args:
            model_name: HuggingFace model identifier.
            device: 'cpu', 'cuda', or None (auto-detect).
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None
        logger.info("FinBertPipeline initialized (device=%s, model=%s)", self.device, model_name)

    def _load(self) -> None:
        """Load tokenizer and model (called lazily on first inference)."""
        if self._model is not None:
            return
        logger.info("Loading FinBERT model '%s' ...", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        logger.info("FinBERT loaded on %s", self.device)

    def score(
        self,
        texts: list[str],
        batch_size: int = FINBERT_BATCH_SIZE,
        return_embeddings: bool = False,
    ) -> pd.DataFrame:
        """Run FinBERT sentiment inference on a list of texts.

        Args:
            texts: List of headline strings.
            batch_size: Number of texts per forward pass (default 32).
            return_embeddings: If True, also return 768-dim CLS embeddings.

        Returns:
            DataFrame with columns:
              text, label, score (-1 to 1), confidence, [embed_0..embed_767]
        """
        self._load()
        if not texts:
            return pd.DataFrame()

        all_labels, all_scores, all_confs, all_embeds = [], [], [], []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=FINBERT_MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**encoded, output_hidden_states=return_embeddings)

            logits = outputs.logits  # (B, 3)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # ProsusAI/finbert label order: positive=0, negative=1, neutral=2
            for prob_row in probs:
                pos, neg, neu = prob_row[0], prob_row[1], prob_row[2]
                label_idx = int(prob_row.argmax())
                label = FINBERT_LABELS[label_idx]
                # Compound score: positive contribution - negative contribution
                score = float(pos - neg)
                confidence = float(prob_row.max())
                all_labels.append(label)
                all_scores.append(score)
                all_confs.append(confidence)

            if return_embeddings:
                # CLS token from last hidden state
                cls_embeds = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                all_embeds.append(cls_embeds)

        result = pd.DataFrame({
            "text": texts,
            "finbert_label": all_labels,
            "finbert_score": all_scores,
            "finbert_confidence": all_confs,
        })

        if return_embeddings and all_embeds:
            embed_matrix = np.vstack(all_embeds)
            embed_df = pd.DataFrame(
                embed_matrix,
                columns=[f"embed_{i}" for i in range(embed_matrix.shape[1])],
            )
            result = pd.concat([result, embed_df], axis=1)

        return result

    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "title",
        batch_size: int = FINBERT_BATCH_SIZE,
        return_embeddings: bool = False,
    ) -> pd.DataFrame:
        """Run FinBERT on a DataFrame column and join results back.

        Args:
            df: Input DataFrame with a text column.
            text_col: Column name containing headline text.
            batch_size: Batch size for inference.
            return_embeddings: Whether to return CLS embeddings.

        Returns:
            Original DataFrame with FinBERT columns added.
        """
        texts = df[text_col].fillna("").tolist()
        scores = self.score(texts, batch_size=batch_size, return_embeddings=return_embeddings)
        scores = scores.drop(columns=["text"])
        return pd.concat([df.reset_index(drop=True), scores], axis=1)


def score_news_file(
    ticker: str,
    news_dir: Path,
    cache_path: Path = FINBERT_CACHE_PATH,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """Score all headlines for a ticker using FinBERT, with disk caching.

    Args:
        ticker: Ticker symbol.
        news_dir: Directory containing {ticker}.parquet news files.
        cache_path: Path to the combined cache Parquet file.
        force_rerun: If True, ignore cache and rerun inference.

    Returns:
        DataFrame with FinBERT scores for all headlines of this ticker.
        Returns empty DataFrame if no news file found.
    """
    news_path = news_dir / f"{ticker}.parquet"
    if not news_path.exists():
        logger.warning("%s: no news file found at %s", ticker, news_path)
        return pd.DataFrame()

    # Check cache
    if not force_rerun and cache_path.exists():
        cache = pd.read_parquet(cache_path)
        if ticker in cache.get("ticker", pd.Series()).values:
            cached = cache[cache["ticker"] == ticker]
            logger.info("%s: loaded %d rows from FinBERT cache", ticker, len(cached))
            return cached

    news_df = pd.read_parquet(news_path)
    if news_df.empty:
        return pd.DataFrame()

    logger.info("%s: running FinBERT on %d headlines ...", ticker, len(news_df))
    pipe = FinBertPipeline()
    scored = pipe.score_dataframe(news_df, text_col="title", return_embeddings=True)
    scored["ticker"] = ticker

    # Update cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
        existing = existing[existing["ticker"] != ticker]
        combined = pd.concat([existing, scored], ignore_index=True)
    else:
        combined = scored
    combined.to_parquet(cache_path, index=False)
    logger.info("%s: FinBERT cache updated (%d total rows)", ticker, len(combined))

    return scored
