"""
predict.py — Inference pipeline for live predictions.

Supports multiple prediction horizons (5-day and 21-day).
Exposes individual build steps so the Streamlit UI can show progress.

Usage:
    from src.models.predict import LivePredictor
    predictor = LivePredictor()
    result = predictor.predict("AAPL", horizon=5)
"""

import logging
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    LIVE_DATA_LOOKBACK_DAYS,
    MODEL_21D_PATH,
    MODELS_DIR,
    PCA_CV_PATH,
    PCA_NLP_PATH,
    STACKING_MODEL_PATH,
)

logger = logging.getLogger(__name__)

# Model paths per horizon
_MODEL_PATHS: dict[int, Path] = {
    5: STACKING_MODEL_PATH,
    21: MODEL_21D_PATH,
}


class LivePredictor:
    """End-to-end live prediction pipeline with multi-horizon support.

    Lazy-loads all models and feature transformers on first use.
    """

    def __init__(self) -> None:
        self._models: dict[int, tuple] = {}  # {horizon: (model, feature_cols)}
        self._nlp_pca: dict | None = None
        self._cv_pca: dict | None = None
        self._finbert = None
        self._vader = None
        self._cnn = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self, horizon: int = 5) -> None:
        """Load the ML model for a given prediction horizon."""
        if horizon in self._models:
            return
        path = _MODEL_PATHS.get(horizon)
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"No model for {horizon}-day horizon at {path}. "
                "Run training first."
            )
        with open(path, "rb") as f:
            saved = pickle.load(f)
        self._models[horizon] = (saved["model"], saved["feature_cols"])
        logger.info("Loaded %d-day model (%d features)", horizon, len(saved["feature_cols"]))

    def has_model(self, horizon: int) -> bool:
        """Check if a model file exists for the given horizon."""
        path = _MODEL_PATHS.get(horizon)
        return path is not None and path.exists()

    def load_nlp_pca(self) -> None:
        if self._nlp_pca is not None:
            return
        if PCA_NLP_PATH.exists():
            with open(PCA_NLP_PATH, "rb") as f:
                self._nlp_pca = pickle.load(f)

    def load_cv_pca(self) -> None:
        if self._cv_pca is not None:
            return
        if PCA_CV_PATH.exists():
            with open(PCA_CV_PATH, "rb") as f:
                self._cv_pca = pickle.load(f)

    @property
    def available_horizons(self) -> list[int]:
        """Return list of horizons that have trained models on disk."""
        return [h for h in sorted(_MODEL_PATHS) if self.has_model(h)]

    # ------------------------------------------------------------------
    # Feature building (public for step-by-step UI)
    # ------------------------------------------------------------------
    def fetch_ohlcv(self, ticker: str) -> pd.DataFrame:
        """Download OHLCV data from yfinance (shared between chart + features)."""
        import yfinance as yf

        end = datetime.today()
        start = end - timedelta(days=max(LIVE_DATA_LOOKBACK_DAYS + 60, 400))
        raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index.name = "Date"
        if raw.empty:
            raise ValueError(f"No market data returned for {ticker}")
        return raw

    def build_market_features(self, ticker: str, ohlcv_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Build technical indicator features from OHLCV data."""
        from src.features.market_features import build_ticker_features

        if ohlcv_df is None:
            ohlcv_df = self.fetch_ohlcv(ticker)
        feat = build_ticker_features(ticker, raw_df=ohlcv_df)
        if feat is None or feat.empty:
            raise ValueError(f"Could not build market features for {ticker}")
        return feat

    def build_nlp_features(self, ticker: str, n_pca: int = 10) -> pd.Series:
        """Score recent headlines and return NLP feature vector."""
        from src.data_collection.news_scraper import load_ticker_news
        from src.nlp.finbert_sentiment import FinBertPipeline
        from src.nlp.vader_sentiment import VaderPipeline

        self.load_nlp_pca()

        if self._finbert is None:
            self._finbert = FinBertPipeline()
        if self._vader is None:
            self._vader = VaderPipeline()

        # Try saved news first
        try:
            news_df = load_ticker_news(ticker)
            headlines = news_df.head(50).to_dict("records")
        except Exception:
            headlines = []

        nlp_feat: dict = {
            "finbert_sentiment": 0.0, "finbert_confidence": 0.0,
            "vader_sentiment": 0.0, "news_volume_1d": 0.0,
            "news_volume_5d": 0.0, "headline_avg_length": 0.0,
            "sentiment_momentum": 0.0, "sentiment_dispersion": 0.0,
            "sentiment_shift_3d": 0.0, "sentiment_surprise": 0.0,
            "sentiment_x_volume": 0.0, "news_volume_zscore": 0.0,
            "is_sentiment_imputed": 1.0,
        }
        for i in range(n_pca):
            nlp_feat[f"finbert_embed_pca_{i+1}"] = 0.0

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
        nlp_feat["is_sentiment_imputed"] = 0.0

        # PCA embeddings
        embed_cols = [c for c in fb.columns if c.startswith("embed_")]
        if embed_cols and self._nlp_pca is not None:
            mean_embed = fb[embed_cols].fillna(0).values.mean(axis=0).reshape(1, -1)
            scaled = self._nlp_pca["scaler"].transform(mean_embed)
            pca_vals = self._nlp_pca["pca"].transform(scaled)[0]
            for i, v in enumerate(pca_vals):
                nlp_feat[f"finbert_embed_pca_{i+1}"] = float(v)

        return pd.Series(nlp_feat)

    def build_cv_features(self, ticker: str, ohlcv_df: pd.DataFrame | None = None, n_pca: int = 10) -> pd.Series:
        """Generate latest chart image and extract CNN embedding.

        Generates a SINGLE chart from the latest 30 days of OHLCV data
        instead of all historical charts (massive speed improvement).
        """
        from src.cv.chart_classifier import ChartCNN

        self.load_cv_pca()

        if self._cnn is None:
            self._cnn = ChartCNN()

        cv_feat: dict = {}
        for i in range(n_pca):
            cv_feat[f"chart_embed_pca_{i+1}"] = 0.0

        try:
            if ohlcv_df is None:
                ohlcv_df = self.fetch_ohlcv(ticker)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            if self._generate_single_chart(ohlcv_df, tmp_path):
                embed = self._cnn.embed_image(tmp_path)
                if self._cv_pca is not None:
                    scaled = self._cv_pca["scaler"].transform(embed.reshape(1, -1))
                    pca_vals = self._cv_pca["pca"].transform(scaled)[0]
                    for i, v in enumerate(pca_vals):
                        cv_feat[f"chart_embed_pca_{i+1}"] = float(v)

            tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("%s: CV feature extraction failed (%s) — using zeros", ticker, exc)

        return pd.Series(cv_feat)

    @staticmethod
    def _generate_single_chart(ohlcv_df: pd.DataFrame, out_path: Path) -> bool:
        """Generate a single 30-day candlestick chart image for CNN input."""
        import matplotlib
        matplotlib.use("Agg")
        import mplfinance as mpf

        window = ohlcv_df.tail(30).copy()
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if len(window) < 20 or not all(c in window.columns for c in needed):
            return False

        window = window[needed]
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds", facecolor="black",
            edgecolor="black", figcolor="black", gridcolor="black",
        )
        fig, _ = mpf.plot(
            window, type="candle", style=style, volume=True,
            axisoff=True, tight_layout=True, returnfig=True,
            figsize=(2.24, 2.24),
        )
        fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
        import matplotlib.pyplot as plt
        plt.close(fig)
        return True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_from_features(
        self,
        ticker: str,
        market_feat: pd.DataFrame,
        nlp_feat: pd.Series,
        cv_feat: pd.Series,
        horizon: int = 5,
    ) -> dict:
        """Run model inference from pre-built features."""
        self.load_model(horizon)
        model, feature_cols = self._models[horizon]

        latest_row = market_feat.iloc[-1].copy()
        latest_date = market_feat.index[-1]

        all_feat = pd.concat([latest_row, nlp_feat, cv_feat])
        feature_vec = (
            pd.DataFrame([all_feat])
            .reindex(columns=feature_cols, fill_value=0)
            .fillna(0)
        )

        proba = model.predict_proba(feature_vec)[0]
        classes = model.classes_
        pred_idx = int(proba.argmax())
        prediction = classes[pred_idx]
        confidence = float(proba[pred_idx])

        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

        logger.info(
            "%s [%dd]: %s (conf=%.2f) | DOWN=%.2f UP=%.2f",
            ticker, horizon, prediction, confidence,
            prob_dict.get("DOWN", 0), prob_dict.get("UP", 0),
        )

        return {
            "ticker": ticker,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "n_headlines": int(nlp_feat.get("news_volume_1d", 0)),
            "market_date": str(latest_date.date()),
            "horizon": horizon,
        }

    def predict(self, ticker: str, horizon: int = 5) -> dict:
        """Full pipeline: fetch data, build features, predict (backwards compatible)."""
        self.load_model(horizon)
        self.load_nlp_pca()
        self.load_cv_pca()

        ohlcv = self.fetch_ohlcv(ticker)
        market = self.build_market_features(ticker, ohlcv_df=ohlcv)
        nlp = self.build_nlp_features(ticker)
        cv = self.build_cv_features(ticker, ohlcv_df=ohlcv)

        return self.predict_from_features(ticker, market, nlp, cv, horizon)
