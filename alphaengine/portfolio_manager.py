"""
Portfolio manager. Single source of truth for live state. Mirrors broker
positions and tracks daily P&L, sector exposure, and per-strategy attribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from .execution_engine import ExecutionEngine
from .risk_manager import PortfolioState
from .logger import get_logger, log_event

log = get_logger(__name__)


SECTOR_MAP: dict[str, str] = {
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "GOOGL": "tech", "META": "tech",
    "AMZN": "consumer_disc", "TSLA": "consumer_disc",
    "JPM": "financials",
    "XOM": "energy",
    "UNH": "healthcare",
    "SPY": "etf", "QQQ": "etf",
}


@dataclass
class PositionTracker:
    symbol: str
    side: str            # 'long' or 'short'
    qty: int
    avg_price: float
    sector: str
    opened_at: datetime
    peak_price: float
    strategy: str
    realized_pnl: float = 0.0


@dataclass
class PortfolioManager:
    execution: ExecutionEngine
    starting_equity: float
    peak_equity: float = 0.0
    today: date = field(default_factory=lambda: date.today())
    daily_start_equity: float = 0.0
    positions: dict[str, PositionTracker] = field(default_factory=dict)
    strategy_pnl: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.peak_equity = self.starting_equity
        self.daily_start_equity = self.starting_equity

    def sync(self) -> None:
        """Pull broker state and reconcile with local positions."""
        try:
            positions = self.execution.positions()
        except Exception as e:
            log.error(f"sync failed: {e}")
            return
        live_symbols = set()
        for p in positions:
            live_symbols.add(p.symbol)
            qty = int(float(p.qty))
            if qty == 0:
                continue
            existing = self.positions.get(p.symbol)
            if existing is None:
                self.positions[p.symbol] = PositionTracker(
                    symbol=p.symbol,
                    side="long" if qty > 0 else "short",
                    qty=abs(qty),
                    avg_price=float(p.avg_entry_price),
                    sector=SECTOR_MAP.get(p.symbol, "other"),
                    opened_at=datetime.now(timezone.utc),
                    peak_price=float(p.current_price or p.avg_entry_price),
                    strategy="external",
                )
            else:
                existing.qty = abs(qty)
                existing.avg_price = float(p.avg_entry_price)
                cp = float(p.current_price or existing.peak_price)
                if existing.side == "long":
                    existing.peak_price = max(existing.peak_price, cp)
                else:
                    existing.peak_price = min(existing.peak_price, cp)

        # Drop closed positions.
        for sym in list(self.positions.keys()):
            if sym not in live_symbols:
                self.positions.pop(sym, None)

    # -------- state snapshot for risk manager --------

    def equity(self) -> float:
        try:
            acct = self.execution.account()
            return float(acct.equity)
        except Exception:
            return self.starting_equity

    def state(self) -> PortfolioState:
        eq = self.equity()
        self.peak_equity = max(self.peak_equity, eq)
        self._roll_day_if_needed(eq)

        sector_exposure: dict[str, float] = {}
        positions_by_symbol: dict[str, dict[str, Any]] = {}
        for p in self.positions.values():
            notional = p.qty * p.avg_price
            sector_exposure[p.sector] = sector_exposure.get(p.sector, 0.0) + notional
            positions_by_symbol[p.symbol] = {
                "qty": p.qty,
                "avg_price": p.avg_price,
                "side": p.side,
                "sector": p.sector,
                "peak_price": p.peak_price,
                "strategy": p.strategy,
            }

        return PortfolioState(
            equity=eq,
            peak_equity=self.peak_equity,
            cash=eq,
            open_positions=len(self.positions),
            daily_pnl=eq - self.daily_start_equity,
            sector_exposure=sector_exposure,
            positions_by_symbol=positions_by_symbol,
        )

    def _roll_day_if_needed(self, eq: float) -> None:
        today = date.today()
        if today != self.today:
            self.today = today
            self.daily_start_equity = eq

    # -------- attribution --------

    def record_trade(self, symbol: str, strategy: str, pnl: float) -> None:
        self.strategy_pnl[strategy] = self.strategy_pnl.get(strategy, 0.0) + pnl
        log_event(log, "INFO", "trade_attribution",
                  symbol=symbol, strategy=strategy, pnl=pnl)

    # -------- weekly rebalance hook --------

    def rebalance_targets(self) -> dict[str, float]:
        """
        Stub for rebalancing. The default is to hold whatever the strategies have
        picked. Override or extend for sector targeting.
        """
        return {}
