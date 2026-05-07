"""
AlphaEngine Web Dashboard.

A small Flask app that lets you:
  * Enter and persist API keys (Alpaca, NewsAPI, Alpha Vantage) to a local .env file
  * Start/stop the trading engine
  * View live signals, positions, P&L, and recent news per symbol
  * Run a backtest from the browser

Run locally:
    python -m alphaengine.web.app

Or via gunicorn for production:
    gunicorn -b 0.0.0.0:8000 alphaengine.web.app:app
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, render_template_string, request, url_for

from ..config import load_config
from ..data_ingestion import DataClient
from ..logger import get_logger
from ..news_data import NewsAggregator
from ..signal_engine import (
    BreakoutStrategy, MeanReversionStrategy, MomentumStrategy,
    RelativeStrengthStrategy, SignalEngine,
)
from ..webhooks import bp as tradingview_bp

log = get_logger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "change-me")
app.register_blueprint(tradingview_bp)

_state: dict[str, Any] = {
    "engine_thread": None,
    "engine_running": False,
    "last_signals": [],
    "last_news": {},
}

ENV_PATH = Path(os.getenv("ENV_FILE", ".env"))


def _read_env() -> dict[str, str]:
    out: dict[str, str] = {}
    if not ENV_PATH.exists():
        return out
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _write_env(updates: dict[str, str]) -> None:
    cur = _read_env()
    cur.update({k: v for k, v in updates.items() if v != ""})
    body = "\n".join(f"{k}={v}" for k, v in cur.items()) + "\n"
    ENV_PATH.write_text(body)


BASE_TMPL = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AlphaEngine Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
           background:#0e1117; color:#e6edf3; margin:0; padding:0; }
    nav { padding: 16px 24px; background:#161b22; display:flex; gap:24px;
          border-bottom: 1px solid #30363d; }
    nav a { color:#7ee787; text-decoration:none; font-weight:600; }
    .container { max-width: 1100px; margin: 24px auto; padding: 0 16px; }
    h1, h2 { color:#e6edf3; }
    .card { background:#161b22; border:1px solid #30363d; border-radius:8px;
            padding:16px; margin-bottom:16px; }
    table { width:100%; border-collapse: collapse; }
    th, td { text-align:left; padding:8px; border-bottom:1px solid #30363d; }
    th { color:#7ee787; font-weight:600; }
    input[type=text], input[type=password] { width:100%; padding:8px; background:#0e1117;
            border:1px solid #30363d; color:#e6edf3; border-radius:4px; }
    button { background:#238636; color:white; border:0; padding:10px 16px;
             border-radius:6px; font-weight:600; cursor:pointer; }
    button.danger { background:#da3633; }
    .badge { padding:2px 8px; border-radius:12px; font-size:12px; }
    .pos { background:#238636; color:white; }
    .neg { background:#da3633; color:white; }
    .neu { background:#6e7681; color:white; }
    code { background:#0e1117; padding:2px 6px; border-radius:4px; }
  </style>
</head>
<body>
  <nav>
    <a href="{{ url_for('home') }}">Dashboard</a>
    <a href="{{ url_for('settings') }}">API Keys</a>
    <a href="{{ url_for('signals') }}">Signals</a>
    <a href="{{ url_for('news_view') }}">News</a>
  </nav>
  <div class="container">{{ body|safe }}</div>
</body>
</html>
"""


def render(body: str) -> str:
    return render_template_string(BASE_TMPL, body=body)


@app.route("/")
def home():
    env = _read_env()
    has_alpaca = bool(env.get("ALPACA_API_KEY"))
    running = _state["engine_running"]
    body = f"""
      <h1>AlphaEngine</h1>
      <div class="card">
        <p><strong>Mode:</strong> {'live' if env.get('ALPACA_BASE_URL', '').endswith('alpaca.markets') and 'paper' not in env.get('ALPACA_BASE_URL', '') else 'paper'}</p>
        <p><strong>Engine:</strong>
          <span class="badge {'pos' if running else 'neu'}">{'RUNNING' if running else 'STOPPED'}</span></p>
        <p><strong>Alpaca keys:</strong>
          <span class="badge {'pos' if has_alpaca else 'neg'}">{'configured' if has_alpaca else 'missing'}</span></p>
        <form method="POST" action="{url_for('toggle')}">
          <button class="{'danger' if running else ''}" type="submit">
            {'Stop engine' if running else 'Start engine'}
          </button>
        </form>
      </div>
      <div class="card">
        <h2>Quick start</h2>
        <ol>
          <li>Add API keys on the <a href="{url_for('settings')}">API Keys</a> page (paper Alpaca minimum).</li>
          <li>Run a backtest: <code>python -m alphaengine.backtest --config config.yaml</code></li>
          <li>Start the engine when paper Sharpe &gt; 1 over 90 days.</li>
        </ol>
      </div>
    """
    return render(body)


@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        updates = {
            "ALPACA_API_KEY": request.form.get("alpaca_key", "").strip(),
            "ALPACA_API_SECRET": request.form.get("alpaca_secret", "").strip(),
            "ALPACA_BASE_URL": request.form.get("alpaca_base_url", "").strip()
                              or "https://paper-api.alpaca.markets",
            "NEWSAPI_KEY": request.form.get("newsapi_key", "").strip(),
            "ALPHAVANTAGE_KEY": request.form.get("alphavantage_key", "").strip(),
            "SLACK_WEBHOOK_URL": request.form.get("slack_webhook_url", "").strip(),
        }
        _write_env(updates)
        return redirect(url_for("settings"))
    env = _read_env()
    body = f"""
      <h1>API Keys</h1>
      <div class="card">
        <p>Keys are written to a local <code>.env</code> file on this server.
        Use a paper Alpaca account first. Live URL is
        <code>https://api.alpaca.markets</code>.</p>
        <form method="POST">
          <p><strong>Alpaca API Key</strong><br>
            <input type="text" name="alpaca_key" value="{env.get('ALPACA_API_KEY', '')}"></p>
          <p><strong>Alpaca API Secret</strong><br>
            <input type="password" name="alpaca_secret" value="{env.get('ALPACA_API_SECRET', '')}"></p>
          <p><strong>Alpaca Base URL</strong><br>
            <input type="text" name="alpaca_base_url"
                   value="{env.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')}"></p>
          <p><strong>NewsAPI Key</strong> (optional)<br>
            <input type="text" name="newsapi_key" value="{env.get('NEWSAPI_KEY', '')}"></p>
          <p><strong>Alpha Vantage Key</strong> (optional, includes news sentiment)<br>
            <input type="text" name="alphavantage_key" value="{env.get('ALPHAVANTAGE_KEY', '')}"></p>
          <p><strong>Slack Webhook</strong> (optional)<br>
            <input type="text" name="slack_webhook_url" value="{env.get('SLACK_WEBHOOK_URL', '')}"></p>
          <button type="submit">Save</button>
        </form>
      </div>
    """
    return render(body)


@app.route("/toggle", methods=["POST"])
def toggle():
    if _state["engine_running"]:
        _state["engine_running"] = False
        return redirect(url_for("home"))
    t = threading.Thread(target=_run_engine, daemon=True)
    _state["engine_thread"] = t
    _state["engine_running"] = True
    t.start()
    return redirect(url_for("home"))


def _run_engine():
    """Lazy import + start. Engine respects mode in config.yaml."""
    try:
        from ..main import AlphaEngine
        engine = AlphaEngine("config.yaml")
        engine.run()
    except Exception as e:
        log.error(f"engine failed: {e}")
    finally:
        _state["engine_running"] = False


@app.route("/signals")
def signals():
    cfg = load_config("config.yaml")
    data = DataClient(cache_dir=cfg.data.cache_dir, paper=True)
    bench_df = None
    try:
        bench_df = data.get_bars(cfg.universe.benchmark, "1Day")
    except Exception:
        pass
    engine = SignalEngine(cfg.signal_engine.to_dict())
    engine.register(MomentumStrategy())
    engine.register(MeanReversionStrategy())
    engine.register(BreakoutStrategy())
    if bench_df is not None:
        engine.register(RelativeStrengthStrategy(benchmark_df=bench_df))

    rows = []
    for s in cfg.universe.symbols:
        try:
            df = data.get_bars(s, "1Day")
            sig = engine.evaluate(s, df)
            badge = "pos" if sig.direction == 1 else ("neg" if sig.direction == -1 else "neu")
            label = {1: "LONG", -1: "SHORT", 0: "FLAT"}[sig.direction]
            rows.append(f"""
                <tr>
                  <td>{s}</td>
                  <td><span class="badge {badge}">{label}</span></td>
                  <td>{sig.confidence:.2f}</td>
                  <td>{sig.regime}</td>
                  <td>{', '.join(sig.reasons[:3])}</td>
                </tr>""")
        except Exception as e:
            rows.append(f"<tr><td>{s}</td><td colspan=4>error: {e}</td></tr>")
    body = f"""
      <h1>Signals</h1>
      <div class="card">
        <table>
          <thead><tr><th>Symbol</th><th>Direction</th><th>Confidence</th>
            <th>Regime</th><th>Why</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    """
    return render(body)


@app.route("/news")
def news_view():
    cfg = load_config("config.yaml")
    aggr = NewsAggregator()
    cards = []
    for s in cfg.universe.symbols:
        try:
            n = aggr.for_symbol(s)
        except Exception as e:
            cards.append(f"<div class='card'><h2>{s}</h2><p>error: {e}</p></div>")
            continue
        badge = "pos" if n.score > 0.1 else ("neg" if n.score < -0.1 else "neu")
        items = "".join(
            f"<li><a href='{h.url}' target='_blank'>{h.title}</a> "
            f"<span class='badge {('pos' if h.sentiment > 0 else 'neg' if h.sentiment < 0 else 'neu')}'>"
            f"{h.sentiment:+.2f}</span></li>"
            for h in n.headlines[:5]
        ) or "<li>no headlines (provide NEWSAPI or ALPHAVANTAGE keys)</li>"
        filings = "".join(
            f"<li>{f.form} on {f.filed_at.date()} "
            f"<a href='{f.url}' target='_blank'>view</a></li>"
            for f in n.filings[:5]
        ) or "<li>no recent filings</li>"
        cards.append(f"""
          <div class="card">
            <h2>{s} <span class="badge {badge}">{n.score:+.2f}</span></h2>
            <h3>Headlines</h3><ul>{items}</ul>
            <h3>SEC Filings</h3><ul>{filings}</ul>
          </div>
        """)
    return render("<h1>News &amp; Filings</h1>" + "".join(cards))


@app.route("/api/state")
def api_state():
    return jsonify({
        "running": _state["engine_running"],
        "has_alpaca_keys": bool(_read_env().get("ALPACA_API_KEY")),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
