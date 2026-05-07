"""
Thin recorder facade. Wire this into main.py / execution_engine.py to persist
every signal, order, fill, and risk event. If DATABASE_URL is unset, all calls
become no-ops, so the engine still runs without a database.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from .models import (
    OrderLog, PortfolioSnapshot, RiskEvent, SignalLog, TradeLog,
    init_db, make_session,
)
from ..logger import get_logger

log = get_logger(__name__)


class TradeLogger:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.Session = None
        if not self.database_url:
            log.info("DATABASE_URL not set; trade logger is a no-op")
            return
        try:
            init_db(self.database_url)
            self.Session = make_session(self.database_url)
            log.info(f"trade logger ready: {self.database_url.split('@')[-1]}")
        except Exception as e:
            log.error(f"trade logger init failed: {e}")
            self.Session = None

    def _commit(self, obj):
        if self.Session is None:
            return
        try:
            with self.Session() as s:
                s.add(obj)
                s.commit()
        except Exception as e:
            log.error(f"db write failed: {e}")

    def signal(self, symbol: str, direction: int, confidence: float,
               regime: str, components: dict, reasons: list,
               actioned: bool = False) -> None:
        self._commit(SignalLog(
            symbol=symbol, direction=direction, confidence=confidence,
            regime=regime, components=components, reasons=reasons,
            actioned=actioned,
        ))

    def order(self, symbol: str, side: str, qty: int, order_type: str,
              status: str, broker_order_id: str | None = None,
              limit_price: float | None = None,
              fill_price: float | None = None, filled_qty: int = 0,
              rejection_reason: str | None = None,
              strategy: str | None = None,
              signal_id: int | None = None) -> None:
        self._commit(OrderLog(
            symbol=symbol, side=side, qty=qty, order_type=order_type,
            status=status, broker_order_id=broker_order_id,
            limit_price=limit_price, fill_price=fill_price,
            filled_qty=filled_qty, rejection_reason=rejection_reason,
            strategy=strategy, signal_id=signal_id,
        ))

    def closed_trade(self, symbol: str, side: str, qty: int,
                     entry_ts: datetime, exit_ts: datetime,
                     entry_price: float, exit_price: float,
                     realized_pnl: float, cost: float,
                     strategy: str, exit_reason: str) -> None:
        self._commit(TradeLog(
            symbol=symbol, side=side, qty=qty,
            entry_ts=entry_ts, exit_ts=exit_ts,
            entry_price=entry_price, exit_price=exit_price,
            realized_pnl=realized_pnl, cost=cost,
            strategy=strategy, exit_reason=exit_reason,
        ))

    def snapshot(self, equity: float, cash: float, open_positions: int,
                 daily_pnl: float, drawdown_pct: float,
                 sector_exposure: dict) -> None:
        self._commit(PortfolioSnapshot(
            equity=equity, cash=cash, open_positions=open_positions,
            daily_pnl=daily_pnl, drawdown_pct=drawdown_pct,
            sector_exposure=sector_exposure,
        ))

    def risk(self, event_type: str, symbol: str | None, detail: dict[str, Any]) -> None:
        self._commit(RiskEvent(event_type=event_type, symbol=symbol, detail=detail))
