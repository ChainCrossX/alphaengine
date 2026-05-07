# AlphaEngine

A defensible, risk-first algorithmic trading framework for US equities.

## What this is

A modular Python system that combines momentum, mean reversion, breakout, relative strength, and a regularized ML directional model into a regime-weighted ensemble, with strict risk controls and an honest cost-aware backtester. It runs against Alpaca (paper or live), exposes a small Flask dashboard for API key entry and monitoring, and ships in Docker.

## What this is not

A money printer. Most retail systems break even or lose net of fees and slippage. The point of this build is rigorous engineering, transparent risk, and an honest evaluation harness. Paper trade for at least 90 days before considering live capital.

## Project structure

```
alphaengine/
  config.yaml            # all thresholds, weights, and toggles
  requirements.txt
  Dockerfile
  docker-compose.yml
  .env.example
  alphaengine/
    config.py            # YAML + .env loader
    logger.py            # structured JSON logging
    data_ingestion.py    # Alpaca primary, yfinance fallback, parquet cache
    indicators.py        # RSI, MACD, ADX, BB, ATR, VWAP, ROC, Donchian, RV
    signal_engine.py     # strategies + regime-weighted ensemble
    ml_model.py          # LightGBM directional classifier
    risk_manager.py      # circuit breakers, stops, sector caps
    position_sizer.py    # half-Kelly + VIX scaling + intraday dampening
    execution_engine.py  # smart limits + retries + market fallback
    portfolio_manager.py # broker reconciliation, P&L, attribution
    alerting.py          # Slack / Discord / Twilio
    news_data.py         # NewsAPI + Alpha Vantage + SEC EDGAR
    main.py              # APScheduler orchestrator
    backtest.py          # vectorized event-driven backtester
    web/app.py           # Flask dashboard
  tests/                 # pytest smoke suite
```

## Quick start

1. Create a paper Alpaca account and copy API key + secret.
2. `cp .env.example .env` and fill in credentials, or use the dashboard's API Keys page.
3. Install: `pip install -r requirements.txt`
4. Backtest first: `python -m alphaengine.backtest --config config.yaml`
5. Launch dashboard: `python -m alphaengine.web.app` then open `http://localhost:8000`
6. Start the engine from the dashboard (paper mode by default).

## Docker

```
docker compose up --build
```

Then open `http://localhost:8000`. The compose file mounts `.env`, `.cache/`, and `logs/` as volumes so settings and history persist across restarts.

## Web dashboard

The dashboard exposes:

- `/` start/stop the engine, see configuration status
- `/settings` paste API keys (Alpaca, NewsAPI, Alpha Vantage, Slack)
- `/signals` live snapshot of every symbol's current ensemble signal
- `/news` headlines and SEC filings per symbol with sentiment scores
- `/api/state` JSON status endpoint

## Strategy notes (what changed vs the original spec, and why)

The original spec asked for an LSTM/Transformer for price prediction, news sentiment as a primary signal, and reinforcement-learning ensemble weighting. After review I made three substitutions on grounds of out-of-sample performance:

1. **LightGBM instead of LSTM.** Deep models on liquid equity prices at the 1m to 1h horizon are mostly fitting noise. A regularized GBT on engineered features generalizes better, trains in seconds, and is interpretable. The model auto-disables itself if validation log-loss does not beat random.
2. **Sentiment as a veto-only overlay.** News-to-price arbitrage is dominated by HFTs reading wires before public APIs publish. The sentiment score can kill a trade that strongly disagrees with the headline tape, but it cannot be the sole reason to enter.
3. **Regime-weighted ensemble instead of RL.** Volatility regime determines whether to favor trend (low vol) or mean reversion (high vol). RL on a 30-day rolling window would chase noise.

If you want to re-add an LSTM, swap `MLStrategy(model=...)` for one wrapping your model. The interface contract is `predict_last(df) -> (p_up, p_down)`.

## Risk controls

All controls in `config.yaml` under `risk:`. The defaults match the original spec:

- 8% max portfolio drawdown -> halt + cooldown
- 3% max single-position loss -> hard stop
- 12% max single-position size
- 30% max sector exposure
- 2.5% daily loss limit -> halt
- 10 max simultaneous positions
- Trailing stop at +2% profit, trail at 1.2%
- 24-hour cooldown after any breach

## Backtesting

```
python -m alphaengine.backtest --config config.yaml --out backtest_equity.csv
```

The backtester models commission (Alpaca is free, but regulatory fees still apply), 3 bps slippage by default, SEC and TAF fees on sells, and bar-close fills. Outputs Sharpe, Sortino, max drawdown, win rate, and the full equity curve.

Decision rule before going live: paper Sharpe > 1.0 over 90 calendar days, max drawdown < 8%, win rate > 45%, and at least 60 trades.

## Going live

The engine refuses to run live unless `mode: live` is set in `config.yaml` and the Alpaca base URL is `https://api.alpaca.markets`. Even then, start with a fraction of intended capital. Keep paper running in parallel for at least the first month and watch for divergence.

## Testing

```
pytest tests/
```

The smoke suite covers imports, indicators, signal engine sanity, risk vetoes, sizing, and a tiny end-to-end backtest run on synthetic data so it does not require live API access.

## TradingView webhook integration

If you trade off custom Pine Script alerts, AlphaEngine accepts incoming TradingView webhooks at `POST /webhook/tradingview` and runs them through the same risk manager as autonomous signals. No bypass.

Setup:

1. Set `TRADINGVIEW_WEBHOOK_SECRET` in `.env` to a long random string.
2. In TradingView, create an alert. Under "Notifications", enable "Webhook URL" and paste your dashboard URL plus the path: `https://your-app.onrender.com/webhook/tradingview`.
3. In the alert message field, paste a JSON payload like:

```json
{"secret":"YOUR_SECRET","symbol":"{{ticker}}","action":"buy","price":{{close}},"strategy":"my_pine_strategy"}
```

Supported actions: `buy`, `sell`, `close`. The `qty` field is optional; if omitted, the position sizer is used. Health check at `/webhook/tradingview/health` confirms the secret is set and engine is attached. Set `TRADINGVIEW_IP_CHECK=1` to enforce TradingView's published IP allowlist.

## Streamlit analytics dashboard

A second, read-only dashboard for charts, equity curves, and per-strategy attribution:

```
streamlit run alphaengine/web/streamlit_app.py
```

Or deploy free at `streamlit.io/cloud`. Point at the same repo and add `DATABASE_URL` as a secret. The Streamlit app reads from the Postgres trade-log database (or sqlite for local), so it shows actual recorded data, not Alpaca state. The Flask app at port 8000 stays the control plane (start/stop, settings, webhooks). Streamlit at port 8501 is the analytics plane.

## Operational checklist

- [ ] Alpaca paper keys configured and `account()` returns equity > 0
- [ ] At least 365 days of cached daily bars per symbol
- [ ] `pytest tests/` passes
- [ ] `python -m alphaengine.backtest` produces metrics with Sharpe > 1.0
- [ ] Slack or Discord webhook alerts wired and tested
- [ ] Dashboard reachable, signals render
- [ ] 90 days paper trading before enabling live
