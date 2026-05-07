# AlphaEngine Operator Runbook

A single, sequential checklist from zero to live trading. Do these in order. Do not skip steps. The 90-day paper period is not optional.

## Phase 0 — Accounts and credentials (you, 20 minutes)

1. Create a paper Alpaca account at `https://app.alpaca.markets/signup`. Pick "Paper Trading" first.
2. In the Alpaca dashboard, go to "Your API Keys" and click "Generate New Key". Copy the API Key ID and Secret. Treat the secret like a password.
3. Optional but recommended:
   - NewsAPI free tier at `https://newsapi.org/register` for headlines.
   - Alpha Vantage free tier at `https://www.alphavantage.co/support/#api-key` for pre-scored news sentiment.
   - Slack Incoming Webhook at `https://api.slack.com/messaging/webhooks` for alerts.
4. If deploying to a host, create accounts for the host (Render, Railway, or Fly.io) and link your GitHub. Stick with one host.

## Phase 1 — Local install and validation (you, 15 minutes)

1. Clone or download the `alphaengine/` project to your machine.
2. From the project root, run: `./bootstrap.sh --no-dashboard`
3. Confirm the smoke tests pass with `10/10 passed`. If anything fails, do not continue.
4. Edit `.env` and paste your Alpaca paper keys. Leave `ALPACA_BASE_URL` as the paper URL.
5. Re-run `./bootstrap.sh --skip-backtest` to verify the dashboard starts. Open `http://localhost:8000`. Confirm `/settings` shows your keys masked, `/signals` renders without error, and `/api/state` returns JSON.

## Phase 2 — Backtest validation (you, 30 minutes)

1. Stop the dashboard and run: `python -m alphaengine.backtest --config config.yaml --out backtest_out/equity.csv`
2. Look at the printed metrics. The decision rule is: do not advance to live capital unless and until paper Sharpe > 1.0 over 90 days, max drawdown < 8%, win rate > 45%, and at least 60 trades.
3. The first backtest will likely be middling. That is expected and correct. The pessimistic cost model is honest. Do not chase numbers by removing costs.
4. If results look catastrophically bad (Sharpe < 0, drawdown > 20%), the strategy params in `config.yaml` need tuning before paper trading. Open `signal_engine` weights and the per-strategy thresholds.

## Phase 3 — Deploy to a public URL (you, 15 minutes)

Pick one of three. Each gives you HTTPS and a domain.

**Render (easiest):**
1. Push the repo to GitHub.
2. Go to `https://render.com/deploy` and point at the repo.
3. Render reads `render.yaml` and provisions the web service plus Postgres.
4. After the build completes, open `/settings` on the new URL and paste API keys there. Render keeps them as environment variables.

**Railway:**
1. Push to GitHub. Click "Deploy from GitHub repo" on Railway.
2. Railway reads `railway.json` and uses `Dockerfile`.
3. Add a Postgres plugin from the Railway dashboard. Copy its `DATABASE_URL` into the service variables.
4. Visit the assigned URL, go to `/settings`, paste keys.

**Fly.io:**
1. Install `flyctl` locally.
2. From the project root: `flyctl launch --copy-config` then `flyctl secrets set ALPACA_API_KEY=... ALPACA_API_SECRET=...`
3. `flyctl deploy`
4. Open the returned URL.

## Phase 4 — Database (optional but recommended, 5 minutes)

1. Set `DATABASE_URL` in the host's environment. Render and Railway provide one automatically when you attach Postgres.
2. SSH or shell into the running container, or run locally with the same env: `python scripts/migrate_db.py`
3. From now on, every signal, order, fill, and risk event is persisted in Postgres. You can query for attribution with simple SQL.

## Phase 5 — Paper trading watch (you, 90 days)

1. Click "Start engine" on the dashboard. Confirm `/api/state` returns `running: true` and you see signal entries flowing in the logs.
2. Set a calendar reminder for Day 7, Day 30, and Day 90.
3. **Day 7 checkpoint.** Open `/signals` daily for a week. Confirm signals are firing, positions opening and closing, no errors in the logs. If you see no trades for 7 days, lower `ensemble_threshold` from 0.72 to 0.65 in `config.yaml` and restart.
4. **Day 30 checkpoint.** Pull live Sharpe, max drawdown, win rate, and trade count from the backtest module pointed at the order log. If Sharpe is negative or drawdown exceeds 8%, stop the engine, review trades by strategy attribution, and tune. Do not advance.
5. **Day 90 checkpoint.** Compare live paper Sharpe to backtested Sharpe. If live is more than 30% below backtest, the strategies are overfit. Stop. Either re-engineer or accept that retail systematic trading on this universe is not viable for you and stand down. There is no shame in that conclusion. Most retail bots fail this test.

## Phase 6 — Live capital (only if Phase 5 passed cleanly)

1. Open a live Alpaca account, fund it with no more than 25 percent of the capital you eventually want to deploy.
2. Generate live API keys. Put them in `.env` or the host's secret store.
3. In `config.yaml`, change `mode: paper` to `mode: live` and `ALPACA_BASE_URL` to `https://api.alpaca.markets`.
4. Restart. Watch the dashboard for the first 5 trades in real time. Confirm fills match expected slippage.
5. Keep paper running in parallel for the first 30 days of live and watch for divergence between paper and live P&L. If they diverge significantly, your slippage model is wrong; tune `slippage_bps` upward.
6. Scale capital up gradually only if live performance tracks paper within 10 percent.

## Daily ops (5 minutes a day, every trading day)

1. Open the dashboard. Confirm engine status is `RUNNING`.
2. Check `/signals` for any obviously broken signals (all-flat for hours during market hours = data problem).
3. Read the Slack alert for daily P&L summary. If a circuit breaker fired, stop and investigate before resuming.
4. On the first of each month, export the `trade_log` table to CSV for taxes and review.

## Failure modes and what to do

| Symptom | Likely cause | Action |
|---|---|---|
| Dashboard returns 500 | bad `.env` or expired keys | regenerate keys, redeploy |
| No signals for a full session | data feed broken | check Alpaca status page; review logs for fetch errors |
| All trades losing | regime mismatch with strategy weights | stop engine, run backtest on last 30 days, retune |
| Live deviates from paper | slippage model too optimistic | raise `slippage_bps` in config |
| Drawdown breaker fires | strategies are wrong, not the engine | stop, do not override the cooldown |

## Hard rules, never break

- Never override a circuit breaker by editing the cooldown timestamp manually.
- Never increase position sizing caps after a losing streak. The instinct to "make it back" is the surest path to ruin.
- Never deploy a config change directly to live without paper trading the same config for at least 14 days.
- Never give your Alpaca secret to anyone, paste it into a chat tool, or commit it to git. Use the dashboard `/settings` page or the host's secret manager.
- Never use leverage above 1x in this framework. Margin in `config.yaml` is intentionally not exposed.
