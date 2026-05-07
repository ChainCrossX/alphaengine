"""
Risk manager. All limits are enforced here. Nothing else in the system has
permission to bypass these rules. Approve/Veto results are logged in full.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from .logger import get_logger, log_event

log = get_logger(__name__)

Decision = Literal["approve", "veto"]


@dataclass
class RiskCheck:
    decision: Decision
    reasons: list[str]


@dataclass
class PortfolioState:
    equity: float
    peak_equity: float
    cash: float
    open_positions: int
    daily_pnl: float
    sector_exposure: dict[str, float]   # symbol -> sector mapping handled upstream
    positions_by_symbol: dict[str, dict]  # {symbol: {qty, avg_price, sector, ...}}


class RiskManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cooldown_until: datetime | None = None

    # -------- circuit breakers --------

    def portfolio_drawdown_pct(self, state: PortfolioState) -> float:
        if state.peak_equity <= 0:
            return 0.0
        return (state.peak_equity - state.equity) / state.peak_equity

    def is_in_cooldown(self) -> bool:
        if self.cooldown_until is None:
            return False
        return datetime.now(timezone.utc) < self.cooldown_until

    def trigger_cooldown(self, reason: str) -> None:
        from datetime import timedelta
        minutes = self.cfg.get("cooldown_after_breach_minutes", 1440)
        self.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        log_event(log, "WARNING", "cooldown_triggered",
                  reason=reason, until=self.cooldown_until.isoformat())

    # -------- pre-trade gate --------

    def check_new_position(
        self,
        symbol: str,
        side: str,                   # 'long' or 'short'
        notional: float,
        sector: str,
        state: PortfolioState,
    ) -> RiskCheck:
        reasons: list[str] = []

        if self.is_in_cooldown():
            return RiskCheck("veto", ["in_cooldown"])

        # Drawdown breaker
        dd = self.portfolio_drawdown_pct(state)
        if dd >= self.cfg["max_portfolio_drawdown_pct"]:
            self.trigger_cooldown(f"max_drawdown {dd:.2%}")
            return RiskCheck("veto", [f"drawdown_{dd:.2%}_>=_max"])

        # Daily loss breaker
        if state.equity > 0:
            daily_pct = state.daily_pnl / state.equity
            if daily_pct <= -self.cfg["max_daily_loss_pct"]:
                self.trigger_cooldown(f"daily_loss {daily_pct:.2%}")
                return RiskCheck("veto", [f"daily_loss_{daily_pct:.2%}"])

        # Open positions cap
        if state.open_positions >= self.cfg["max_open_positions"]:
            return RiskCheck("veto", [f"max_open_positions={state.open_positions}"])

        # Single position size cap
        if state.equity > 0:
            pos_pct = notional / state.equity
            if pos_pct > self.cfg["max_single_position_size_pct"]:
                return RiskCheck("veto", [f"position_pct_{pos_pct:.2%}_above_max"])

        # Sector concentration
        sector_exposure = state.sector_exposure.get(sector, 0.0)
        new_sector_pct = (sector_exposure + notional) / max(state.equity, 1)
        if new_sector_pct > self.cfg["max_sector_exposure_pct"]:
            return RiskCheck("veto", [f"sector_{sector}_{new_sector_pct:.2%}_above_max"])

        # Already in this symbol? Block stacking.
        if symbol in state.positions_by_symbol:
            return RiskCheck("veto", [f"already_in_{symbol}"])

        reasons.append("ok")
        return RiskCheck("approve", reasons)

    # -------- exit logic --------

    def should_stop_out(self, position: dict, current_price: float) -> tuple[bool, str]:
        """Hard stop based on percent loss from entry."""
        entry = position["avg_price"]
        side = position["side"]
        if side == "long":
            loss_pct = (entry - current_price) / entry
        else:
            loss_pct = (current_price - entry) / entry
        if loss_pct >= self.cfg["hard_stop_loss_pct"]:
            return True, f"hard_stop_{loss_pct:.2%}"
        return False, ""

    def trailing_stop_price(self, position: dict, peak_price: float) -> float | None:
        """
        Once a position has gained at least `activation_pct`, trail at `distance_pct`.
        Returns None if trailing not active.
        """
        entry = position["avg_price"]
        side = position["side"]
        act = self.cfg["trailing_stop_activation_pct"]
        dist = self.cfg["trailing_stop_distance_pct"]

        if side == "long":
            gain = (peak_price - entry) / entry
            if gain < act:
                return None
            return peak_price * (1 - dist)
        else:
            gain = (entry - peak_price) / entry
            if gain < act:
                return None
            return peak_price * (1 + dist)
