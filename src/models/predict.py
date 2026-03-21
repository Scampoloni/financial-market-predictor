"""
predict.py — Inference pipeline for live predictions (Streamlit app).

Loads the saved Config C model and all feature pipelines, then produces
a UP/DOWN/SIDEWAYS prediction with confidence for a given ticker and date.

The prediction uses:
  1. Market features from yfinance (last 60 days OHLCV)
  2. NLP features from RSS headlines (last 24h, scored with FinBERT + VADER)
  3. CV features from a 30-day candlestick chart (EfficientNet-B0 embedding)

Usage:
    from src.models.predict import LivePredictor
    predictor = LivePredictor()
    result = predictor.predict("AAPL")
    # → {'ticker': 'AAPL', 'prediction': 'UP', 'confidence': 0.62, 'probabilities': {...}}
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FEATURES_MARKET_PATH,
    LIVE_DATA_LOOKBACK_DAYS,
    MODELS_DIR,
    PCA_CV_PATH,
    PCA_NLP_PATH,
    RAW_NEWS_DIR,
    STACKING_MODEL_PATH,
    TARGET_CLASSES,
)

logger = logging.getLogger(__name__)


class LivePredictor:
    """End-to-end live prediction pipeline.

    Lazy-loads all models (RF, FinBERT, CNN, PCA) on first use.
    Caches loaded models across calls to avoid repeated disk I/O.
    """

    def __init__(self) -> None:
        self._model = None
        self._feature_cols: list[str] | None = None
        self._nlp_pca: dict | None = None
        self._cv_pca: dict | None = None
        self._finbert = None
        self._vader = None
        self._cnn = None

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------
    def _load_main_model(self) -> None:
        if self._model is not None:
            return
        if not STACKING_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {STACKING_MODEL_PATH}. "
                "Run: python -m src.models.train_ml"
            )
        with open(STACKING_MODEL_PATH, "rb") as f:
            saved = pickle.load(f)
        self._model = saved["model"]
        self._feature_cols = saved["feature_cols"]
        logger.info("Loaded Config C model (%d features)", len(self._feature_cols))

    def _load_nlp_pca(self) -> None:
        if self._nlp_pca is not None:
            return
        if PCA_NLP_PATH.exists():
            with open(PCA_NLP_PATH, "rb") as f:
                self._nlp_pca = pickle.load(f)
        else:
            logger.warning("NLP PCA not found — embed PCA features will be 0")
            self._nlp_pca = None

    def _load_cv_pca(self) -> None:
        if self._cv_pca is not None:
            return
        if PCA_CV_PATH.exists():
            with open(PCA_CV_PATH, "rb") as f:
                self._cv_pca = pickle.load(f)
        else:
            logger.warning("CV PCA not found — chart embed features will be 0")
            self._cv_pca = None

    # ------------------------------------------------------------------
    # Feature building helpers
    # ------------------------------------------------------------------
    def _build_market_features(self, ticker: str) -> pd.DataFrame:
        """Download recent OHLCV and compute technical indicators."""
        import yfinance as yf
        from src.features.market_features import build_ticker_features

        logger.info("%s: fetching live OHLCV (%d days) ...", ticker, LIVE_DATA_LOOKBACK_DAYS)
        end = datetime.today()
        start = end - timedelta(days=LIVE_DATA_LOOKBACK_DAYS + 60)  # extra for warmup
        raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), progress=False)
        if raw.empty:
            raise ValueError(f"No market data returned for {ticker}")

        # Flatten MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.index.name = "Date"
        feat = build_ticker_features(ticker, raw_df=raw)
        return feat

    def _build_nlp_features(self, ticker: str, n_pca: int = 10) -> pd.Series:
        """Score recent headlines and return NLP feature Series."""
        from src.data_collection.news_scraper import load_ticker_news, fetch_all_rss, match_headlines_to_ticker
        from src.nlp.finbert_sentiment import FinBertPipeline
        from src.nlp.vader_sentiment import VaderPipeline

        if self._finbert is None:
            self._finbert = FinBertPipeline()
        if self._vader is None:
            self._vader = VaderPipeline()

        try:
            # Try saved parquet first, then live RSS as fallback
            news_df = load_ticker_news(ticker)
            headlines = news_df.head(50).to_dict("records")
        except FileNotFoundError:
            try:
                rss_df = fetch_all_rss()
                matched = match_headlines_to_ticker(rss_df, ticker)
                headlines = matched.head(50).to_dict("records")
            except Exception as exc:
                logger.warning("%s: news fetch failed (%s) — using neutral NLP", ticker, exc)
                headlines = []
        except Exception as exc:
            logger.warning("%s: news load failed (%s) — using neutral NLP", ticker, exc)
            headlines = []

        nlp_feat: dict = {
            "finbert_sentiment": 0.0, "finbert_confidence": 0.0,
            "vader_sentiment": 0.0, "news_volume_1d": 0.0,
            "news_volume_5d": 0.0, "headline_avg_length": 0.0,
            "sentiment_momentum": 0.0, "sentiment_dispersion": 0.0,
        }
        for i in range(n_pca):
            nlp_feat[f"finbert_embed_pca_{i+1}"] = 0.0

        if not headlines:
            return pd.Series(nlp_feat)

        texts = [h.get("title", "") for h in headlines if h.get("title")]
        if not texts:
            return pd.Series(nlp_feat)

        fb = self._finbert.score(texts, return_embeddings=True)
        vd = self._vader.score(texts)

        nlp_feat["finbert_sentiment"] = float(fb["finbert_score"].mean())
        nlp_feat["finbert_confidence"] = float(fb["finbert_confidence"].mean())
        nlp_feat["vader_sentiment"] = float(vd["vader_compound"].mean())
        nlp_feat["news_volume_1d"] = float(len(texts))
        nlp_feat["news_volume_5d"] = float(len(texts))
        nlp_feat["headline_avg_length"] = float(
            pd.Series(texts).str.split().str.len().mean()
        )
        nlp_feat["sentiment_dispersion"] = float(fb["finbert_score"].std())
        nlp_feat["sentiment_momentum"] = nlp_feat["finbert_sentiment"]

        # PCA embeddings
        embed_cols = [c for c in fb.columns if c.startswith("embed_")]
        if embed_cols and self._nlp_pca is not None:
            mean_embed = fb[embed_cols].fillna(0).values.mean(axis=0).reshape(1, -1)
            scaled = self._nlp_pca["scaler"].transform(mean_embed)
            pca_vals = self._nlp_pca["pca"].transform(scaled)[0]
            for i, v in enumerate(pca_vals):
                nlp_feat[f"finbert_embed_pca_{i+1}"] = float(v)

        return pd.Series(nlp_feat)

    def _build_cv_features(self, ticker: str, n_pca: int = 10) -> pd.Series:
        """Generate latest chart image and extract CNN embedding."""
        import tempfile
        from src.data_collection.chart_generator import generate_charts_for_ticker
        from src.cv.chart_classifier import ChartCNN

        if self._cnn is None:
            self._cnn = ChartCNN()

        cv_feat: dict = {}
        for i in range(n_pca):
            cv_feat[f"chart_embed_pca_{i+1}"] = 0.0

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                n = generate_charts_for_ticker(ticker, output_dir=tmp_path, step=1, force=True)
                chart_files = sorted((tmp_path / ticker).glob("*.png"))
                if not chart_files:
                    return pd.Series(cv_feat)

                latest_chart = chart_files[-1]
                embed = self._cnn.embed_image(latest_chart)

                if self._cv_pca is not None:
                    scaled = self._cv_pca["scaler"].transform(embed.reshape(1, -1))
                    pca_vals = self._cv_pca["pca"].transform(scaled)[0]
                    for i, v in enumerate(pca_vals):
                        cv_feat[f"chart_embed_pca_{i+1}"] = float(v)
        except Exception as exc:
            logger.warning("%s: CV feature extraction failed (%s) — using zeros", ticker, exc)

        return pd.Series(cv_feat)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, ticker: str) -> dict:
        """Generate a live UP/DOWN/SIDEWAYS prediction for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').

        Returns:
            Dict with keys: ticker, prediction, confidence, probabilities,
            n_headlines, market_date.
        """
        self._load_main_model()
        self._load_nlp_pca()
        self._load_cv_pca()

        # Build features
        market_feat = self._build_market_features(ticker)
        if market_feat.empty:
            raise ValueError(f"Could not build market features for {ticker}")

        latest_row = market_feat.iloc[-1].copy()
        latest_date = market_feat.index[-1]

        nlp_feat = self._build_nlp_features(ticker)
        cv_feat  = self._build_cv_features(ticker)

        # Assemble feature vector — reindex to training columns, missing = 0
        all_feat = pd.concat([latest_row, nlp_feat, cv_feat])
        feature_vec = (
            pd.DataFrame([all_feat])
            .reindex(columns=self._feature_cols, fill_value=0)
            .fillna(0)
        )

        # Predict
        proba = self._model.predict_proba(feature_vec)[0]
        classes = self._model.classes_
        pred_idx = int(proba.argmax())
        prediction = classes[pred_idx]
        confidence = float(proba[pred_idx])

        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

        logger.info(
            "%s: %s (conf=%.2f) | DOWN=%.2f SIDEWAYS=%.2f UP=%.2f",
            ticker, prediction, confidence,
            prob_dict.get("DOWN", 0), prob_dict.get("SIDEWAYS", 0), prob_dict.get("UP", 0),
        )

        return {
            "ticker": ticker,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "n_headlines": int(nlp_feat.get("news_volume_1d", 0)),
            "market_date": str(latest_date.date()),
        }
