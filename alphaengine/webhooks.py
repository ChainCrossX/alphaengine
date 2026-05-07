"""
TradingView webhook handler.

TradingView Pine Script alerts can POST a JSON body to a URL when triggered.
We accept those alerts at /webhook/tradingview, validate a shared secret to
prevent spoofing, run them through the same risk manager as autonomous signals,
and submit via the execution engine. No bypass.

Expected JSON payload from TradingView:
    {
      "secret": "your_shared_secret",
      "symbol": "AAPL",
      "action": "buy" | "sell" | "close",
      "qty": 10,                       // optional; if omitted, sizer is used
      "price": 175.23,                 // optional; for logging only
      "strategy": "tv_supertrend",     // optional label
      "comment": "long entry"          // optional, free-form
    }

Pine Script alert message template:
    {"secret":"REPLACE_ME","symbol":"{{ticker}}","action":"buy",
     "price":{{close}},"strategy":"my_pine_strategy"}

Security:
  * Shared secret in env var TRADINGVIEW_WEBHOOK_SECRET
  * IP allowlist optional (TradingView publishes their IPs)
  * Rate-limited via simple in-memory token bucket
  * All payloads logged to trade_log database if configured
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Any

from flask import Blueprint, jsonify, request

from .logger import get_logger, log_event

log = get_logger(__name__)

bp = Blueprint("webhooks", __name__)

# Module-level handles populated by the web app at startup.
_state: dict[str, Any] = {
    "execution": None,
    "risk": None,
    "portfolio": None,
    "sizer": None,
    "trade_logger": None,
    "sector_map": {},
}

# TradingView's published webhook IPs (subject to change; see TV docs).
_TV_ALLOWED_IPS = {
    "52.89.214.238",
    "34.212.75.30",
    "54.218.53.128",
    "52.32.178.7",
}

# Crude rate limit: max 30 webhooks per minute.
_TIMES: deque[float] = deque(maxlen=30)


def configure(execution=None, risk=None, portfolio=None, sizer=None,
              trade_logger=None, sector_map: dict | None = None) -> None:
    """Wire engine components into the webhook handler."""
    _state["execution"] = execution
    _state["risk"] = risk
    _state["portfolio"] = portfolio
    _state["sizer"] = sizer
    _state["trade_logger"] = trade_logger
    if sector_map:
        _state["sector_map"] = sector_map


def _rate_limited() -> bool:
    now = time.time()
    while _TIMES and now - _TIMES[0] > 60:
        _TIMES.popleft()
    if len(_TIMES) >= _TIMES.maxlen:
        return True
    _TIMES.append(now)
    return False


def _ip_allowed(req) -> bool:
    if os.getenv("TRADINGVIEW_IP_CHECK", "0") != "1":
        return True
    src = req.headers.get("X-Forwarded-For", req.remote_addr or "")
    src = src.split(",")[0].strip()
    return src in _TV_ALLOWED_IPS


def _validate_payload(data: dict) -> tuple[bool, str]:
    secret = os.getenv("TRADINGVIEW_WEBHOOK_SECRET")
    if not secret:
        return False, "server_missing_secret"
    if data.get("secret") != secret:
        return False, "bad_secret"
    if "symbol" not in data:
        return False, "missing_symbol"
    if data.get("action") not in {"buy", "sell", "close"}:
        return False, "bad_action"
    return True, "ok"


@bp.post("/webhook/tradingview")
def tradingview():
    if _rate_limited():
        return jsonify({"ok": False, "error": "rate_limited"}), 429
    if not _ip_allowed(request):
        return jsonify({"ok": False, "error": "ip_not_allowed"}), 403

    try:
        data = request.get_json(force=True, silent=False) or {}
    except Exception:
        # Some TV alert templates send a string body. Accept that too.
        try:
            import json
            data = json.loads(request.data.decode("utf-8"))
        except Exception:
            return jsonify({"ok": False, "error": "invalid_json"}), 400

    valid, reason = _validate_payload(data)
    if not valid:
        log_event(log, "WARNING", "tv_webhook_rejected", reason=reason)
        return jsonify({"ok": False, "error": reason}), 401

    symbol = str(data["symbol"]).upper()
    action = data["action"]
    qty_in = data.get("qty")
    price = float(data.get("price", 0) or 0)
    strategy = data.get("strategy", "tradingview")
    comment = data.get("comment", "")

    log_event(log, "INFO", "tv_webhook_received",
              symbol=symbol, action=action, qty=qty_in, price=price,
              strategy=strategy, comment=comment)

    execution = _state["execution"]
    risk = _state["risk"]
    portfolio = _state["portfolio"]
    sizer = _state["sizer"]
    trade_logger = _state["trade_logger"]
    sector_map = _state["sector_map"]

    if execution is None:
        return jsonify({"ok": False, "error": "engine_not_running"}), 503

    # CLOSE: flatten the position regardless of source.
    if action == "close":
        portfolio.sync() if portfolio else None
        positions = (portfolio.positions if portfolio else {}) or {}
        if symbol not in positions:
            return jsonify({"ok": True, "info": "no_position", "symbol": symbol})
        p = positions[symbol]
        side = "sell" if p.side == "long" else "buy"
        result = execution.submit(symbol, side, p.qty)
        if trade_logger:
            trade_logger.order(
                symbol=symbol, side=side, qty=p.qty, order_type="market",
                status="filled" if result.accepted else "rejected",
                broker_order_id=result.order_id,
                fill_price=result.avg_fill_price,
                filled_qty=result.filled_qty,
                rejection_reason=result.reason, strategy=strategy,
            )
        return jsonify({
            "ok": result.accepted, "symbol": symbol, "action": "close",
            "fill_price": result.avg_fill_price, "qty": result.filled_qty,
            "reason": result.reason,
        })

    # BUY / SELL: pre-trade risk check.
    if portfolio:
        portfolio.sync()
        state = portfolio.state()
    else:
        return jsonify({"ok": False, "error": "no_portfolio"}), 503

    # Determine quantity.
    if qty_in is not None:
        qty = int(qty_in)
    else:
        if not sizer:
            return jsonify({"ok": False, "error": "no_sizer"}), 503
        try:
            from .position_sizer import TradeStat
            stats = TradeStat(win_rate=0.5, avg_win=0.012, avg_loss=0.01)
            est_price = price or 1.0
            qty = sizer.size(equity=state.equity, price=est_price,
                             confidence=0.7, stats=stats)
        except Exception as e:
            return jsonify({"ok": False, "error": f"sizer:{e}"}), 500
        if qty <= 0:
            return jsonify({"ok": False, "error": "zero_qty"}), 400

    notional = qty * (price or 1.0)
    sector = sector_map.get(symbol, "other")
    side_long_short = "long" if action == "buy" else "short"

    if risk:
        check = risk.check_new_position(symbol, side_long_short, notional, sector, state)
        if check.decision == "veto":
            log_event(log, "INFO", "tv_webhook_vetoed",
                      symbol=symbol, reasons=check.reasons)
            return jsonify({"ok": False, "error": "risk_veto",
                            "reasons": check.reasons}), 403

    result = execution.submit(symbol, action, qty)

    if trade_logger:
        trade_logger.order(
            symbol=symbol, side=action, qty=qty, order_type="limit",
            status="filled" if result.accepted else "rejected",
            broker_order_id=result.order_id,
            fill_price=result.avg_fill_price,
            filled_qty=result.filled_qty,
            rejection_reason=result.reason, strategy=strategy,
        )

    return jsonify({
        "ok": result.accepted,
        "symbol": symbol,
        "action": action,
        "qty": qty,
        "fill_price": result.avg_fill_price,
        "order_id": result.order_id,
        "reason": result.reason,
    })


@bp.get("/webhook/tradingview/health")
def health():
    return jsonify({
        "ok": True,
        "secret_configured": bool(os.getenv("TRADINGVIEW_WEBHOOK_SECRET")),
        "engine_attached": _state["execution"] is not None,
    })
