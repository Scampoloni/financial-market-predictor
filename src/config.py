"""
config.py — Central configuration for the Financial Market Predictor.
All paths, constants, ticker lists, and hyperparameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

RAW_MARKET_DIR = RAW_DIR / "market_data"
RAW_NEWS_DIR = RAW_DIR / "news"
RAW_CHARTS_DIR = RAW_DIR / "charts"

MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# Processed feature files
FEATURES_MARKET_PATH = PROCESSED_DIR / "features_market.parquet"
FEATURES_NLP_PATH = PROCESSED_DIR / "features_nlp.parquet"
FEATURES_CV_PATH = PROCESSED_DIR / "features_cv.parquet"
FEATURES_COMBINED_PATH = PROCESSED_DIR / "features_combined.parquet"

# Metadata
TICKERS_CSV_PATH = METADATA_DIR / "tickers.csv"

# Model artifacts
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_final.pkl"
STACKING_MODEL_PATH = MODELS_DIR / "stacking_final.pkl"
MODEL_21D_PATH = MODELS_DIR / "model_21d.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
PCA_NLP_PATH = MODELS_DIR / "pca_nlp.pkl"
PCA_CV_PATH = MODELS_DIR / "pca_cv.pkl"
CHART_CNN_PATH = MODELS_DIR / "chart_cnn.pth"

# ---------------------------------------------------------------------------
# API Keys (loaded from .env)
# ---------------------------------------------------------------------------
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# Data collection constants
# ---------------------------------------------------------------------------
DATA_START_DATE = "2020-01-01"
DATA_END_DATE = "2026-03-21"  # extended to include recent period for NLP feature overlap

# Train / validation / test splits (temporal, no leakage)
TRAIN_START = "2020-01-01"
TRAIN_END = "2024-06-30"
VAL_START = "2024-07-01"
VAL_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# Target definition: binary classification
TARGET_HORIZON_DAYS = 5          # predict 5-trading-day forward return
TARGET_HORIZON_DAYS_LONG = 21   # predict 21-trading-day (~1 month) forward return
TARGET_CLASSES = ["DOWN", "UP"]  # binary: return <= 0 → DOWN, > 0 → UP

# ---------------------------------------------------------------------------
# Ticker universe — ~80 S&P 500 tickers across sectors
# ---------------------------------------------------------------------------
TICKERS_TECH = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    "AMD", "INTC", "QCOM", "CRM", "ORCL", "ADBE", "NOW", "INTU",
]

TICKERS_FINANCE = [
    "JPM", "GS", "BAC", "MS", "V", "MA", "BRK-B",
    "C", "WFC", "AXP", "BLK", "SCHW",
]

TICKERS_INSURANCE = [
    "AIG", "MET", "PRU", "ALL", "TRV",
]

TICKERS_HEALTHCARE = [
    "JNJ", "PFE", "UNH", "ABBV", "MRK",
    "LLY", "TMO", "ABT", "AMGN", "GILD",
]

TICKERS_CONSUMER = [
    "KO", "PEP", "MCD", "NKE", "SBUX",
    "PG", "WMT", "COST", "TGT", "HD",
]

TICKERS_ENERGY = [
    "XOM", "CVX", "COP",
    "SLB", "EOG", "PSX",
]

TICKERS_INDUSTRIAL = [
    "BA", "CAT", "GE", "MMM", "HON",
    "UPS", "FDX", "RTX", "LMT",
]

TICKERS_ALL: list[str] = (
    TICKERS_TECH
    + TICKERS_FINANCE
    + TICKERS_INSURANCE
    + TICKERS_HEALTHCARE
    + TICKERS_CONSUMER
    + TICKERS_ENERGY
    + TICKERS_INDUSTRIAL
)

# Sector mapping (ticker → sector label)
TICKER_SECTOR_MAP: dict[str, str] = {
    **{t: "Technology" for t in TICKERS_TECH},
    **{t: "Finance" for t in TICKERS_FINANCE},
    **{t: "Insurance" for t in TICKERS_INSURANCE},
    **{t: "Healthcare" for t in TICKERS_HEALTHCARE},
    **{t: "Consumer" for t in TICKERS_CONSUMER},
    **{t: "Energy" for t in TICKERS_ENERGY},
    **{t: "Industrial" for t in TICKERS_INDUSTRIAL},
}

# Market index tickers (for VIX and benchmark)
MARKET_INDICES = ["^VIX", "^GSPC"]

# ---------------------------------------------------------------------------
# Company name mapping for news relevance filtering
# ---------------------------------------------------------------------------
COMPANY_KEYWORDS: dict[str, list[str]] = {
    # Technology
    "AAPL": ["apple", "iphone", "ipad", "mac", "tim cook", "ios", "app store", "macbook"],
    "MSFT": ["microsoft", "windows", "azure", "copilot", "satya nadella", "xbox", "office 365"],
    "NVDA": ["nvidia", "jensen huang", "cuda", "geforce", "gpu", "rtx"],
    "GOOGL": ["google", "alphabet", "youtube", "gemini", "sundar pichai", "android", "waymo"],
    "META": ["meta platforms", "meta glasses", "meta ai", "meta vr", "meta trial",
             "meta drops", "meta isn", "facebook", "instagram", "whatsapp", "zuckerberg", "threads", "metaverse"],
    "AMZN": ["amazon", "aws", "jeff bezos", "andy jassy", "prime video"],
    "TSLA": ["tesla", "elon musk", "electric vehicle", "autopilot", "cybertruck"],
    "AMD": ["amd", "advanced micro", "lisa su", "radeon", "ryzen", "epyc"],
    "INTC": ["intel", "pat gelsinger", "core ultra"],
    "QCOM": ["qualcomm", "snapdragon"],
    "CRM": ["salesforce", "marc benioff"],
    "ORCL": ["oracle", "larry ellison"],
    "ADBE": ["adobe", "photoshop", "creative cloud"],
    "NOW": ["servicenow"],
    "INTU": ["intuit", "turbotax", "quickbooks"],
    # Finance
    "JPM": ["jpmorgan", "jp morgan", "jamie dimon", "chase bank"],
    "GS": ["goldman sachs", "goldman"],
    "BAC": ["bank of america"],
    "MS": ["morgan stanley"],
    "V": ["visa"],
    "MA": ["mastercard"],
    "BRK-B": ["berkshire hathaway", "warren buffett", "berkshire"],
    "C": ["citigroup", "citibank", "citi"],
    "WFC": ["wells fargo"],
    "AXP": ["american express", "amex"],
    "BLK": ["blackrock", "larry fink"],
    "SCHW": ["charles schwab", "schwab"],
    # Insurance
    "AIG": ["aig", "american international group"],
    "MET": ["metlife"],
    "PRU": ["prudential"],
    "ALL": ["allstate"],
    "TRV": ["travelers"],
    # Healthcare
    "JNJ": ["johnson & johnson", "johnson and johnson", "j&j"],
    "PFE": ["pfizer"],
    "UNH": ["unitedhealth", "united health"],
    "ABBV": ["abbvie"],
    "MRK": ["merck"],
    "LLY": ["eli lilly", "lilly"],
    "TMO": ["thermo fisher"],
    "ABT": ["abbott"],
    "AMGN": ["amgen"],
    "GILD": ["gilead"],
    # Consumer
    "KO": ["coca-cola", "coca cola", "coke"],
    "PEP": ["pepsi", "pepsico"],
    "MCD": ["mcdonald"],
    "NKE": ["nike"],
    "SBUX": ["starbucks"],
    "PG": ["procter & gamble", "procter and gamble", "p&g"],
    "WMT": ["walmart", "wal-mart"],
    "COST": ["costco"],
    "TGT": ["target corp"],
    "HD": ["home depot"],
    # Energy
    "XOM": ["exxon", "exxonmobil"],
    "CVX": ["chevron"],
    "COP": ["conocophillips", "conoco"],
    "SLB": ["schlumberger"],
    "EOG": ["eog resources"],
    "PSX": ["phillips 66"],
    # Industrial
    "BA": ["boeing"],
    "CAT": ["caterpillar"],
    "GE": ["general electric"],
    "MMM": ["3m company"],
    "HON": ["honeywell"],
    "UPS": ["ups", "united parcel"],
    "FDX": ["fedex", "federal express"],
    "RTX": ["raytheon", "rtx corp"],
    "LMT": ["lockheed martin", "lockheed"],
}

SPAM_KEYWORDS: list[str] = [
    "comedy", "movie", "film", "bachelorette", "celebrity", "glyphosate",
    "recipe", "fashion", "nba ", "nfl ", "soccer", "music", "album",
    "actor", "actress", "kernel", "npm", "pypi", "github release",
    "changelog", "open source", "mixed precision", "numkong", "satirical",
    "marketing ner", "sugar harmony", "singapore air", "roundtrip",
]

# ---------------------------------------------------------------------------
# Feature engineering constants
# ---------------------------------------------------------------------------
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
SMA_SHORT = 20
SMA_LONG = 50
EMA_PERIOD = 12
ATR_PERIOD = 14
VOLATILITY_PERIOD = 20
VOLUME_AVG_PERIOD = 20
RETURN_PERIODS = [1, 5, 20]   # days for rolling return features

NLP_PCA_COMPONENTS = 10
CV_PCA_COMPONENTS = 10

# News deduplication threshold (rapidfuzz ratio)
NEWS_DEDUP_THRESHOLD = 90

# Minimum price filter (exclude penny stocks)
MIN_STOCK_PRICE = 5.0

# ---------------------------------------------------------------------------
# Chart generation (CV block)
# ---------------------------------------------------------------------------
CHART_WINDOW_DAYS = 30          # candlestick chart covers 30 trading days
CHART_IMAGE_SIZE = (224, 224)   # pixels (standard for ResNet/EfficientNet)
CHART_PATTERN_CLASSES = ["uptrend", "downtrend", "sideways", "reversal"]
CHART_LABEL_THRESHOLD = 0.02    # >2% in 5 days → uptrend / downtrend

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
CV_FOLDS = 5  # TimeSeriesSplit folds

LOGISTIC_REGRESSION_PARAMS = {
    "C": [0.01, 0.1, 1, 10],
    "max_iter": 1000,
    "solver": "lbfgs",
    "multi_class": "multinomial",
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": [5, 10, None],
    "min_samples_leaf": [1, 5, 10],
    "random_state": 42,
}

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": [0.01, 0.05],
    "max_depth": [3, 5, 7],
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# NLP constants
# ---------------------------------------------------------------------------
FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 32
FINBERT_MAX_LENGTH = 512

GEMINI_MODEL_NAME = "gemini-pro"

# RSS feed URLs
RSS_FEEDS = {
    "yahoo_finance": "https://feeds.finance.yahoo.com/rss/2.0/headline",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
}

# ---------------------------------------------------------------------------
# Streamlit app constants
# ---------------------------------------------------------------------------
APP_TITLE = "Financial Market Predictor"
APP_ICON = "📈"
LIVE_DATA_LOOKBACK_DAYS = 60   # days of history to fetch for live predictions
