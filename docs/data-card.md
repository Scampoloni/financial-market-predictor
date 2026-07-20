# Data Card

## Sources and scope

The project collects OHLCV and VIX context via `yfinance`; headline metadata from RSS feeds and optionally NewsAPI; FinBERT weights from Hugging Face; and generated candlestick images from OHLCV windows. The configured universe contains 67 US large-cap tickers in seven manually mapped sectors.

## Coverage and missingness

Direct ticker-matched headline coverage is sparse. The NLP pipeline can use sector means, market means, and forward-fill to populate feature rows. Near-complete feature-row coverage therefore does not mean near-complete direct-news coverage. `is_sentiment_imputed` should be retained and reported.

## Time and point-in-time limitations

Headline timestamps are converted to UTC then normalised to a date. The current pipeline does not use an exchange calendar to assign after-close, weekend, or holiday items to the next tradable session. `yfinance` aggregate analyst recommendations and targets may be current values; they are not validated point-in-time history and must not support historical claims.

## Licensing and redistribution

Provider terms govern Yahoo Finance/yfinance data, news headlines, links, and source content. RSS and NewsAPI content may be copyrighted or restricted. Do not commit full article text, undisclosed provider exports, or large raw datasets without a rights review. Generated charts inherit the practical data-rights question of their underlying prices. Hugging Face model licences must be checked before redistributing weights.

This is technical documentation, not legal advice.
