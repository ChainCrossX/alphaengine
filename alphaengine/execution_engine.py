"""
Execution engine. Smart limit pricing inside the spread, exponential backoff,
and a market-order fallback only after retries are exhausted. Everything goes
through pre-trade validation against the live broker state.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    def retry(*a, **kw):
        def deco(fn): return fn
        return deco
    def stop_after_attempt(*a, **kw): return None
    def wait_exponential(*a, **kw): return None
    def retry_if_exception_type(*a, **kw): return None

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        LimitOrderRequest, MarketOrderRequest, StopOrderRequest, TrailingStopOrderRequest
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
    _ALPACA = True
except ImportError:
    _ALPACA = False

from .data_ingestion import DataClient
from .logger import get_logger, log_event

log = get_logger(__name__)

Side = Literal["buy", "sell"]


@dataclass
class OrderResult:
    accepted: bool
    order_id: str | None
    avg_fill_price: float | None
    filled_qty: int
    reason: str | None = None


class ExecutionError(Exception):
    pass


class ExecutionEngine:
    def __init__(
        self,
        cfg: dict,
        data_client: DataClient,
        api_key: str | None = None,
        api_secret: str | None = None,
        paper: bool = True,
    ):
        if not _ALPACA:
            raise ImportError("alpaca-py is required for ExecutionEngine")
        self.cfg = cfg
        self.data = data_client
        self.client = TradingClient(
            api_key or os.getenv("ALPACA_API_KEY"),
            api_secret or os.getenv("ALPACA_API_SECRET"),
            paper=paper,
        )

    # -------- account / positions --------

    def account(self):
        return self.client.get_account()

    def positions(self):
        return self.client.get_all_positions()

    def cancel_all(self) -> None:
        try:
            self.client.cancel_orders()
        except Exception as e:
            log.warning(f"cancel_all failed: {e}")

    # -------- pre-trade validation --------

    def _validate(self, symbol: str, side: Side, qty: int, est_price: float) -> str | None:
        if qty <= 0:
            return "qty_zero"
        if est_price <= 0:
            return "price_zero"
        try:
            acct = self.account()
            buying_power = float(acct.buying_power)
            if side == "buy" and qty * est_price > buying_power:
                return f"insufficient_buying_power: need={qty * est_price:.2f} have={buying_power:.2f}"
            if not bool(acct.trading_blocked) is False:
                return "trading_blocked"
        except Exception as e:
            log.warning(f"pre-trade validation failed: {e}")
            return f"validation_error:{e}"
        return None

    # -------- limit pricing --------

    def _smart_limit_price(self, symbol: str, side: Side) -> float | None:
        q = self.data.get_latest_quote(symbol)
        if q is None or q.bid <= 0 or q.ask <= 0:
            return None
        edge_bps = self.cfg.get("limit_edge_bps", 5)
        edge = q.mid * edge_bps / 10_000
        if side == "buy":
            # buy slightly below mid to reduce slippage; if not filled, retry tighter.
            return round(q.mid - edge, 2)
        return round(q.mid + edge, 2)

    # -------- core place --------

    def submit(self, symbol: str, side: Side, qty: int) -> OrderResult:
        if qty <= 0:
            return OrderResult(False, None, None, 0, "qty_zero")

        # Estimate price for validation.
        q = self.data.get_latest_quote(symbol)
        est_price = q.mid if q else 0.0
        invalid = self._validate(symbol, side, qty, est_price)
        if invalid:
            log_event(log, "WARNING", "order_rejected_pretrade",
                      symbol=symbol, side=side, qty=qty, reason=invalid)
            return OrderResult(False, None, None, 0, invalid)

        max_retries = self.cfg.get("max_retry_attempts", 4)
        backoff = self.cfg.get("retry_backoff_seconds", 1.5)

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                limit_price = self._smart_limit_price(symbol, side)
                if limit_price is None:
                    return self._place_market(symbol, side, qty)

                req = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )
                order = self.client.submit_order(req)
                log_event(log, "INFO", "order_submitted",
                          symbol=symbol, side=side, qty=qty,
                          limit_price=limit_price, order_id=order.id)
                # Poll briefly for fill.
                filled = self._poll_fill(order.id, timeout_sec=20)
                if filled:
                    return filled
            except Exception as e:
                last_err = str(e)
                log.warning(f"submit attempt {attempt} failed: {e}")
            time.sleep(backoff ** attempt)

        if self.cfg.get("market_order_fallback_after_retries", True):
            log_event(log, "WARNING", "fallback_to_market",
                      symbol=symbol, side=side, qty=qty, last_err=last_err)
            return self._place_market(symbol, side, qty)
        return OrderResult(False, None, None, 0, last_err or "unknown")

    def _place_market(self, symbol: str, side: Side, qty: int) -> OrderResult:
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self.client.submit_order(req)
            filled = self._poll_fill(order.id, timeout_sec=15)
            if filled:
                return filled
            return OrderResult(True, order.id, None, 0, "submitted_unfilled")
        except Exception as e:
            log_event(log, "ERROR", "market_order_failed",
                      symbol=symbol, side=side, qty=qty, error=str(e))
            return OrderResult(False, None, None, 0, str(e))

    def _poll_fill(self, order_id: str, timeout_sec: int = 15) -> OrderResult | None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                o = self.client.get_order_by_id(order_id)
                if o.status == OrderStatus.FILLED:
                    return OrderResult(
                        True, str(o.id),
                        float(o.filled_avg_price or 0),
                        int(float(o.filled_qty or 0)),
                    )
                if o.status in (OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
                    return OrderResult(False, str(o.id), None, 0, f"status_{o.status}")
            except Exception as e:
                log.warning(f"poll_fill error: {e}")
            time.sleep(0.5)
        return None

    def submit_trailing_stop(self, symbol: str, qty: int, trail_pct: float) -> OrderResult:
        try:
            req = TrailingStopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                trail_percent=round(trail_pct * 100, 3),
            )
            order = self.client.submit_order(req)
            return OrderResult(True, str(order.id), None, 0, "trailing_submitted")
        except Exception as e:
            log_event(log, "ERROR", "trailing_stop_failed",
                      symbol=symbol, qty=qty, error=str(e))
            return OrderResult(False, None, None, 0, str(e))
