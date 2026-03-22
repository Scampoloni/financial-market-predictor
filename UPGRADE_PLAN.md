# UPGRADE_PLAN.md — Financial Market Predictor Overhaul
# Claude Code: Read this file COMPLETELY before making ANY changes.
# Execute changes in the EXACT order specified. Commit after each section.

---

## CONTEXT: CURRENT STATE OF THE PROJECT

This is a ZHAW AI Applications course project (deadline: June 7, 2026).
It predicts stock price movements by combining 3 AI blocks:
- ML on structured market data (28 technical indicators)
- NLP on financial news (FinBERT + VADER sentiment)
- CV on candlestick chart images (EfficientNet-B0 embeddings)

### Current Performance (BROKEN — needs fixing)
| Config | Features | Test F1 | 
|--------|----------|---------|
| A — Market only | 28 | 0.3415 |
| B — Market + NLP | 46 | 0.3430 |
| C — Market + NLP + CV | 56 | 0.3443 |

These results are barely above random guessing (0.33 for 3 classes).
The NLP delta (+0.0015) and CV delta (+0.0028) are statistically meaningless.

### Root Causes Identified
1. TARGET: Next-day 3-class (UP/DOWN/SIDEWAYS with ±1% threshold) is too noisy
2. NLP COVERAGE: 98.3% of trading days have NO news data — NLP features are zeros
3. CV COVERAGE: Only 26 charts per ticker (1.7% coverage) — CV features are sparse  
4. MODEL: Only RandomForest tested seriously — no LightGBM, no Optuna tuning, no Stacking
5. NLP FEATURES: Raw sentiment levels instead of dynamic features (shifts, surprises)
6. APP DESIGN: Ugly default Streamlit, line chart instead of candlesticks, broken news matching
7. APP SPEED: Too slow — model reloads on every click, no caching

### Goal After This Overhaul
- F1: 0.45+ (realistic target for 5-day binary classification)
- NLP delta: +0.03 or more (visible, meaningful contribution)
- CV delta: +0.01 or more (visible contribution)  
- App: Modern dark UI, candlestick charts, <3s prediction time
- Everything documented, all notebooks re-run, ablation table updated

---

## PHASE 1: TARGET VARIABLE CHANGE (DO THIS FIRST)
Priority: CRITICAL | Estimated time: 1-2 hours | Biggest single impact

### What to change
The prediction target must change from:
- OLD: Next-day return classified as UP (>1%) / DOWN (<-1%) / SIDEWAYS (±1%)
- NEW: Next-5-day return classified as UP (>0%) / DOWN (≤0%) — BINARY

### Why
- 3-class with narrow SIDEWAYS band is near-impossible to predict (Precision 0.26)
- 5-day horizon smooths daily noise and gives NLP sentiment time to materialize
- Binary classification doubles the effective signal-to-noise ratio
- Expected improvement: F1 from 0.34 → 0.42-0.48 from this change alone

### Implementation Steps

1. Find where the target variable is created (likely in feature engineering or data preprocessing).
   Change the target computation to:
   ```python
   # OLD (remove this)
   # df['target'] = pd.cut(df['return_1d'], bins=[-np.inf, -0.01, 0.01, np.inf], labels=['DOWN', 'SIDEWAYS', 'UP'])
   
   # NEW
   df['forward_return_5d'] = df.groupby('ticker')['close'].transform(
       lambda x: x.shift(-5) / x - 1
   )
   df['target'] = (df['forward_return_5d'] > 0).astype(int)  # 1 = UP, 0 = DOWN
   df = df.dropna(subset=['target'])  # drop last 5 rows per ticker (no future data)
   ```

2. Update ALL downstream references:
   - Change from 3-class to 2-class everywhere (model training, evaluation, app)
   - Update metrics: use binary F1, Precision, Recall, AUC-ROC
   - Update confusion matrix visualization (2x2 instead of 3x3)
   - Update class labels in Streamlit app (remove SIDEWAYS, show UP/DOWN only)
   - Update the probability display (2 classes, not 3)

3. CRITICAL: Ensure no data leakage. The forward_return_5d uses future data 
   for the TARGET only. No features should use future data. Verify that:
   - All rolling features use .shift(1) or are computed on past data only
   - The temporal train/test split is preserved (train ≤2024, test = 2025)
   - Drop the last 5 rows per ticker in the dataset (they have no valid target)

4. Re-run the ML baseline with the new target to verify improvement before 
   proceeding to other phases.

### Commit
```
refactor(target): change to 5-day binary classification (UP/DOWN)

- Replace next-day 3-class target with 5-day forward return binary
- UP = return > 0%, DOWN = return <= 0%
- Update all evaluation metrics to binary (F1, AUC-ROC, Precision, Recall)
- Remove SIDEWAYS class from all visualizations and app components
- Verify no data leakage in forward-looking target computation
```

---

## PHASE 2: NLP COVERAGE FIX
Priority: CRITICAL | Estimated time: 3-4 hours | Makes NLP block actually work

### Problem
98.3% of trading days have no news data. NLP features are zero for almost 
every row. The NLP block is essentially a no-op.

### Solution: Multi-layer news coverage strategy

1. **Primary: yfinance built-in news**
   For every ticker, yfinance provides recent news via `ticker.news`.
   This is the easiest source and doesn't require API keys.
   ```python
   import yfinance as yf
   ticker = yf.Ticker("AAPL")
   news = ticker.news  # returns list of dicts with title, link, publisher, date
   ```
   Limitation: only returns recent news (last ~2 weeks), not historical.
   Use this for the LIVE APP predictions.

2. **Secondary: Sector-level news as fallback**
   When a ticker has no news on a given day, use the SECTOR average sentiment.
   For example, if AAPL has no news on 2024-03-15, use the average FinBERT 
   sentiment of all Technology sector headlines on that day.
   This alone should boost coverage from 1.7% to 20-30%.
   ```python
   # For each day, compute sector-level sentiment
   sector_sentiment = news_df.groupby(['date', 'sector'])['finbert_sentiment'].mean()
   # Fill missing ticker-day sentiment with sector average
   df['sentiment'] = df['sentiment'].fillna(df['sector_sentiment'])
   ```

3. **Tertiary: Market-wide sentiment (VIX-like text signal)**
   For days with no sector news either, use the market-wide average sentiment
   across ALL headlines that day. This captures "risk-on" vs "risk-off" mood.
   Coverage should reach 40-60%.

4. **Quaternary: Forward-fill remaining gaps**
   For days still missing (weekends, holidays), forward-fill the last known
   sentiment value. Do NOT use future sentiment. Mark forward-filled rows
   with a flag `is_sentiment_imputed = True` so the model can learn to 
   weight these differently.

5. **Fix news matching (CRITICAL — currently broken)**
   The app currently shows irrelevant headlines like "jax-mps 0.9.10.dev444" 
   for AAPL. Fix the matching logic:
   ```python
   def is_relevant_news(headline: str, ticker: str, company_name: str) -> bool:
       headline_lower = headline.lower()
       # Must contain ticker symbol OR company name
       if ticker.lower() in headline_lower or company_name.lower() in headline_lower:
           return True
       # Filter out obviously non-financial content
       spam_keywords = ['pypi', 'github', 'npm', 'dev', 'release notes', 'changelog']
       if any(kw in headline_lower for kw in spam_keywords):
           return False
       return False
   ```

### New NLP Features (replace or augment existing ones)
Add these dynamic features that capture sentiment CHANGES, not just levels:

```python
# Sentiment momentum: change over last 3 days
df['sentiment_shift_3d'] = df.groupby('ticker')['finbert_sentiment'].transform(
    lambda x: x - x.shift(3)
)

# Sentiment surprise: deviation from 20-day rolling mean
rolling_mean = df.groupby('ticker')['finbert_sentiment'].transform(
    lambda x: x.rolling(20, min_periods=5).mean()
)
rolling_std = df.groupby('ticker')['finbert_sentiment'].transform(
    lambda x: x.rolling(20, min_periods=5).std()
)
df['sentiment_surprise'] = (df['finbert_sentiment'] - rolling_mean) / (rolling_std + 1e-8)

# Sentiment × Volume interaction (high sentiment + high volume = stronger signal)
df['sentiment_x_volume'] = df['finbert_sentiment'] * df['volume_ratio']

# News volume spike (unusual number of articles = something happening)
df['news_volume_zscore'] = df.groupby('ticker')['news_count'].transform(
    lambda x: (x - x.rolling(20).mean()) / (x.rolling(20).std() + 1e-8)
)

# Sentiment dispersion (disagreement across headlines = uncertainty)
df['sentiment_dispersion'] = df.groupby(['ticker', 'date'])['finbert_sentiment'].transform('std')
```

### Commit
```
feat(nlp): fix coverage with sector fallback + add dynamic sentiment features

- Add sector-level sentiment fallback (fills ~30% of missing days)
- Add market-wide sentiment fallback (fills another ~20%)  
- Forward-fill remaining gaps with imputation flag
- Fix news matching: filter spam, require ticker/company name match
- Add 5 new dynamic NLP features: sentiment_shift_3d, sentiment_surprise,
  sentiment_x_volume, news_volume_zscore, sentiment_dispersion
- NLP coverage: 1.7% → ~60%
```

---

## PHASE 3: CV COVERAGE FIX  
Priority: HIGH | Estimated time: 2 hours | Makes CV block relevant

### Problem
Only 26 charts per ticker (monthly) = 2,788 total. 98.3% of rows have no CV data.

### Solution
Generate a chart for EVERY trading day (using the prior 30-day window):

```python
# For each ticker, for each trading day, generate a 30-day lookback chart
for ticker in tickers:
    ticker_data = df[df['ticker'] == ticker].sort_values('date')
    for i in range(30, len(ticker_data)):
        window = ticker_data.iloc[i-30:i]
        date = ticker_data.iloc[i]['date']
        
        # Generate candlestick chart
        chart_path = f"data/raw/charts/{ticker}_{date}.png"
        if not os.path.exists(chart_path):
            mpf.plot(window, type='candle', style='charles',
                     savefig=chart_path, figsize=(2.24, 2.24),
                     axisoff=True)  # No axes for cleaner CNN input
```

This gives ~67 tickers × ~1,200 trading days = ~80,000 charts.
Too many to process with EfficientNet in real-time, so:

1. Pre-compute ALL chart embeddings offline and save to parquet
2. At training time, just join embeddings by (ticker, date)
3. At inference time, generate chart + extract embedding on the fly (cache it)

If generating 80k charts is too slow, sample every 5th trading day = ~16,000 charts.
That's still 6x more than current 2,788.

### Commit
```
feat(cv): increase chart coverage from monthly to weekly generation

- Generate 30-day candlestick charts for every 5th trading day per ticker
- Pre-compute EfficientNet embeddings and save to parquet
- CV coverage: 1.7% → ~20% (weekly) or 100% (daily)
- Update feature pipeline to join pre-computed embeddings
```

---

## PHASE 4: MODEL UPGRADE
Priority: HIGH | Estimated time: 3 hours | Squeezes remaining F1 points

### Add LightGBM
```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    min_child_samples=20,
    is_unbalance=True,  # handles class imbalance
    random_state=42,
    verbose=-1
)
```

### Add Optuna Hyperparameter Tuning
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    model = lgb.LGBMClassifier(**params, is_unbalance=True, verbose=-1)
    # Use TimeSeriesSplit cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='f1')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
best_params = study.best_params
```

### Add Stacking Ensemble
```python
from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced')),
        ('xgb', XGBClassifier(**best_xgb_params)),
        ('lgb', LGBMClassifier(**best_lgb_params)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=TimeSeriesSplit(n_splits=5),
    stack_method='predict_proba',
    n_jobs=-1
)
```

### Updated Ablation Study
Re-run ALL configs with the best model (likely Stacking or tuned LightGBM):

| Config | Features | Models | Test F1 | AUC-ROC | Delta |
|--------|----------|--------|---------|---------|-------|
| A — Market only | ~28 | Stacking | TBD | TBD | Baseline |
| B — Market + NLP | ~38 | Stacking | TBD | TBD | TBD |
| C — Market + NLP + CV | ~48 | Stacking | TBD | TBD | TBD |

Also report per-model comparison:
| Model | Config C F1 | AUC-ROC | Training Time |
|-------|-------------|---------|---------------|
| Logistic Regression | TBD | TBD | TBD |
| RandomForest | TBD | TBD | TBD |
| XGBoost (tuned) | TBD | TBD | TBD |
| LightGBM (tuned) | TBD | TBD | TBD |
| Stacking Ensemble | TBD | TBD | TBD |

### Commit
```
model(upgrade): add LightGBM + Optuna tuning + Stacking ensemble

- Add LightGBM classifier with is_unbalance=True
- Add Optuna hyperparameter tuning (50 trials) for XGBoost and LightGBM
- Add StackingClassifier (RF + XGB + LGB, meta: LogisticRegression)
- Re-run full ablation study with best model
- Update all evaluation notebooks with new results
```

---

## PHASE 5: APP SPEED OPTIMIZATION
Priority: HIGH | Estimated time: 1-2 hours | App becomes usable

### Caching Strategy
Add these caching decorators to the Streamlit app:

```python
import streamlit as st

# Load models ONCE at app startup (never reload)
@st.cache_resource
def load_models():
    """Load all ML models and preprocessors. Called once at startup."""
    import joblib
    model = joblib.load('models/stacking_final.pkl')
    scaler = joblib.load('models/scaler.pkl')
    pca_nlp = joblib.load('models/pca_nlp.pkl')
    pca_cv = joblib.load('models/pca_cv.pkl')
    return model, scaler, pca_nlp, pca_cv

# Cache market data for 1 hour
@st.cache_data(ttl=3600)
def fetch_market_data(ticker: str, period: str = "6mo"):
    """Fetch OHLCV data from Yahoo Finance. Cached for 1 hour."""
    import yfinance as yf
    return yf.Ticker(ticker).history(period=period)

# Cache news for 1 hour
@st.cache_data(ttl=3600)
def fetch_news(ticker: str):
    """Fetch news headlines. Cached for 1 hour."""
    import yfinance as yf
    return yf.Ticker(ticker).news

# Cache FinBERT inference (most expensive operation)
@st.cache_data(ttl=3600)
def compute_sentiment(headlines: tuple):  # tuple for hashability
    """Run FinBERT on headlines. Cached for 1 hour."""
    from transformers import pipeline
    pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return pipe(list(headlines))

# Cache chart rendering
@st.cache_data(ttl=3600)
def render_chart(ticker: str, data_hash: str):
    """Generate candlestick chart. Cached by data hash."""
    # ... chart generation code
    pass
```

### Loading UX
```python
with st.spinner("Loading models..."):
    model, scaler, pca_nlp, pca_cv = load_models()

with st.spinner(f"Fetching {ticker} data..."):
    data = fetch_market_data(ticker)

with st.spinner("Analyzing sentiment..."):
    sentiment = compute_sentiment(tuple(headlines))
```

### Pre-computation Option
If FinBERT is still too slow even with caching, pre-compute sentiments 
for all 67 tickers daily and store in `data/processed/sentiment_cache.parquet`.
At prediction time, just look up the cached value. Only run FinBERT live 
if the cache miss (ticker not in cache or cache older than 1 day).

### Commit
```
perf(app): add caching for models, data, and sentiment — target <3s predictions

- Add @st.cache_resource for model loading (one-time at startup)
- Add @st.cache_data(ttl=3600) for market data, news, and chart rendering
- Add @st.cache_data for FinBERT inference
- Add loading spinners for user feedback during computation
- Target: prediction completes in <3 seconds after initial load
```

---

## PHASE 6: APP DESIGN OVERHAUL
Priority: MEDIUM-HIGH | Estimated time: 3-4 hours | Makes it portfolio-worthy

### Design Direction
Modern financial dashboard. Think Bloomberg Terminal meets Vercel — 
dark background, clean typography, minimal but informative.

### Color Palette
```python
COLORS = {
    'bg': '#0f1117',           # App background (Streamlit dark default)
    'card_bg': '#1a1d23',      # Card/container background
    'card_border': '#2d3139',  # Subtle card borders
    'text': '#e6e6e6',         # Primary text
    'text_dim': '#8b8d93',     # Secondary/muted text
    'up': '#22c55e',           # Green for UP / bullish
    'down': '#ef4444',         # Red for DOWN / bearish
    'accent': '#3b82f6',       # Blue accent for interactive elements
    'warning': '#f59e0b',      # Yellow/amber for warnings
}
```

### Custom CSS (inject at top of app.py)
```python
st.markdown("""
<style>
    /* Card containers */
    .stMetric {
        background-color: #1a1d23;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 16px;
    }
    
    /* Clean up default Streamlit padding */
    .block-container { padding-top: 2rem; max-width: 1200px; }
    
    /* Header styling */
    h1 { font-weight: 600; letter-spacing: -0.02em; }
    
    /* Metric value styling */
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
    
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)
```

### Layout Changes

#### Header Bar (replace current minimal header)
```
┌──────────────────────────────────────────────────────────────┐
│  📈 Market Predictor          [Prediction] [Analysis] [Chat] │
├──────────────────────────────────────────────────────────────┤
│  Ticker: [AAPL ▼]  [Predict]                                │
│                                                               │
│  Apple Inc. · Technology · $248.52 · ▼ -1.2% today           │
│  52W Range: $164.08 — $278.21                                │
└──────────────────────────────────────────────────────────────┘
```

#### Main Content Area
```
┌─────────────────────────────────────────┬────────────────────┐
│                                         │  PREDICTION        │
│   Plotly Candlestick Chart              │  ▲ UP              │
│   (OHLCV with volume bars below)        │  Confidence: 62%   │
│   (last 90 days)                        │  AUC: 0.58         │
│                                         │                    │
│                                         │  ──────────────    │
│                                         │  UP    62% ████░░  │
│                                         │  DOWN  38% ███░░░  │
├─────────────────────────────────────────┼────────────────────┤
│  SHAP Feature Contributions             │  Recent Headlines  │
│  ┌─────────────────────────────┐        │  ┌──────────────┐  │
│  │ RSI_14        ████▓  +0.08 │        │  │ Reuters 3/20 │  │
│  │ VIX_level     ███▓   +0.06 │        │  │ "Apple..."   │  │
│  │ Sentiment     ██▓    +0.04 │        │  │ Score: 0.82  │  │
│  │ Volume_ratio  █▓     -0.03 │        │  ├──────────────┤  │
│  └─────────────────────────────┘        │  │ CNBC 3/19    │  │
│                                         │  │ "iPhone..."  │  │
│                                         │  │ Score: 0.45  │  │
│                                         │  └──────────────┘  │
└─────────────────────────────────────────┴────────────────────┘
```

#### Candlestick Chart (Plotly)
Replace the current matplotlib line chart with a Plotly candlestick:
```python
import plotly.graph_objects as go

fig = go.Figure(data=[go.Candlestick(
    x=df['date'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    increasing_line_color='#22c55e',
    decreasing_line_color='#ef4444',
)])
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis_rangeslider_visible=False,
    height=400,
    margin=dict(l=0, r=0, t=0, b=0),
    yaxis=dict(gridcolor='#2d3139'),
    xaxis=dict(gridcolor='#2d3139'),
)
st.plotly_chart(fig, use_container_width=True)
```

#### Prediction Display (replace ugly probability bar)
```python
col1, col2 = st.columns([1, 2])
with col1:
    direction = "UP" if prediction == 1 else "DOWN"
    color = "#22c55e" if prediction == 1 else "#ef4444"
    arrow = "▲" if prediction == 1 else "▼"
    st.markdown(f"""
    <div style="background: {color}15; border: 1px solid {color}40; 
                border-radius: 12px; padding: 24px; text-align: center;">
        <div style="font-size: 2.5rem;">{arrow}</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{direction}</div>
        <div style="color: #8b8d93; margin-top: 4px;">
            Confidence: {confidence:.0%}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Horizontal confidence bar
    up_prob = proba[1]
    down_prob = proba[0]
    st.markdown(f"""
    <div style="margin-top: 12px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #ef4444;">DOWN {down_prob:.0%}</span>
            <span style="color: #22c55e;">UP {up_prob:.0%}</span>
        </div>
        <div style="background: #2d3139; border-radius: 4px; height: 8px; overflow: hidden;">
            <div style="background: #22c55e; width: {up_prob:.0%}; height: 100%; 
                        float: right; border-radius: 4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
```

#### Headlines Display (fix broken matching + improve design)
```python
st.markdown("#### Recent Headlines")
for article in filtered_news[:5]:  # Only show relevant, filtered news
    sentiment_color = "#22c55e" if article['sentiment'] > 0.1 else \
                      "#ef4444" if article['sentiment'] < -0.1 else "#f59e0b"
    st.markdown(f"""
    <div style="background: #1a1d23; border: 1px solid #2d3139; 
                border-radius: 8px; padding: 12px; margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: 500;">{article['title']}</div>
                <div style="color: #8b8d93; font-size: 0.85rem; margin-top: 4px;">
                    {article['source']} · {article['date']}
                </div>
            </div>
            <div style="background: {sentiment_color}20; color: {sentiment_color}; 
                        padding: 4px 12px; border-radius: 12px; font-size: 0.85rem;
                        font-weight: 600; white-space: nowrap;">
                {article['sentiment']:+.2f}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
```

### Additional Tabs

#### Tab: Model Analysis
Show the ablation study results as a clean table + bar chart.
Show SHAP summary plot (beeswarm) for the full test set.
Show feature importance ranking.

#### Tab: Market Chat (RAG)
If RAG chatbot is implemented, add a chat interface:
```python
st.markdown("#### 💬 Ask about recent market events")
user_question = st.text_input("", placeholder="What happened to NVDA last week?")
if user_question:
    with st.spinner("Searching news and generating answer..."):
        answer = rag_pipeline(user_question)
    st.markdown(answer)
```

#### Tab: About
Project description, data sources, methodology, ethical disclaimer.
Include: "Research prototype — not financial advice."

### Commit
```
feat(app): complete design overhaul — modern dark financial dashboard

- Replace line chart with Plotly candlestick chart
- New prediction display with directional arrow and confidence bar
- Fix news matching: filter irrelevant headlines, require entity match
- Add SHAP waterfall chart for feature explanations
- Add custom CSS: dark cards, clean typography, consistent color palette
- Add ticker info header: company name, sector, price, daily change
- Add Model Analysis tab with ablation results
- Add About tab with methodology and ethical disclaimer
- Remove default Streamlit branding
```

---

## PHASE 7: RE-RUN EVERYTHING AND UPDATE DOCS
Priority: HIGH | Estimated time: 2-3 hours | Final polish

### Steps
1. Re-run ALL notebooks (01 through 06) with the new target, new features,
   new models. Every notebook must execute cleanly top-to-bottom.

2. Update the ablation table in:
   - 06_evaluation_ablation.ipynb
   - README.md
   - The Streamlit Model Analysis tab

3. Update README.md with:
   - New results table
   - Updated methodology description (5-day binary, not 3-class)
   - Updated feature list
   - Ethical considerations section
   - Deployment URL
   - Clear instructions to reproduce

4. Verify deployment works end-to-end on Hugging Face Spaces

5. Final git log review: ensure commit history tells a clear story

### Commit
```
docs: update all notebooks, README, and ablation results with final numbers

- Re-run all 6 notebooks with new target + features + models
- Update ablation table with final F1/AUC scores
- Update README with methodology, results, and ethical considerations
- Verify deployment on Hugging Face Spaces
```

---

## EXECUTION ORDER (STRICT — follow this sequence)

1. PHASE 1: Target change (5-day binary) → commit → verify F1 improvement
2. PHASE 2: NLP coverage fix → commit → verify NLP delta improvement  
3. PHASE 3: CV coverage fix → commit → verify CV delta improvement
4. PHASE 4: Model upgrade (LightGBM + Optuna + Stacking) → commit
5. PHASE 5: App speed (caching) → commit → verify <3s predictions
6. PHASE 6: App design overhaul → commit → verify visual quality
7. PHASE 7: Re-run everything + update docs → final commit

DO NOT skip phases or change the order. Each phase builds on the previous one.
After each phase, run the relevant tests and verify the improvement before 
proceeding.

---

## DEPENDENCIES TO ADD TO requirements.txt
```
lightgbm>=4.0.0
optuna>=3.0.0
```

---

END OF UPGRADE_PLAN.md
