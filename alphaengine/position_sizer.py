"""
Position sizing. Half-Kelly is dangerous when the win rate is overestimated, so
we cap aggressively. We also scale down with VIX and dampen at the open/close.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np

Method = Literal["half_kelly", "fixed_fractional", "vol_target"]


@dataclass
class TradeStat:
    win_rate: float
    avg_win: float    # in % return
    avg_loss: float   # in % return, positive number


class PositionSizer:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def kelly_fraction(self, stats: TradeStat) -> float:
        if stats.avg_loss <= 0 or stats.win_rate <= 0:
            return 0.0
        b = stats.avg_win / stats.avg_loss
        p = stats.win_rate
        q = 1 - p
        f = (b * p - q) / b
        f = max(0.0, f) * 0.5  # half-Kelly
        return min(f, self.cfg.get("kelly_max_fraction", 0.10))

    def vix_scalar(self, vix: float | None) -> float:
        cfg = self.cfg.get("vix_scaling", {})
        if not cfg.get("enabled", False) or vix is None:
            return 1.0
        ref = cfg.get("reference_vix", 18)
        # Inverse scaling: high VIX -> smaller size.
        scalar = ref / max(vix, 1.0)
        return float(np.clip(scalar, cfg.get("min_scalar", 0.4),
                             cfg.get("max_scalar", 1.2)))

    def time_of_day_scalar(self, now: datetime, market_open: datetime,
                           market_close: datetime) -> float:
        d_open = (now - market_open).total_seconds() / 60
        d_close = (market_close - now).total_seconds() / 60
        cfg = self.cfg.get("intraday_dampening", {})
        ow = cfg.get("open_window_minutes", 15)
        cw = cfg.get("close_window_minutes", 15)
        s = cfg.get("open_close_scalar", 0.5)
        if 0 <= d_open <= ow or 0 <= d_close <= cw:
            return s
        return 1.0

    def size(
        self,
        equity: float,
        price: float,
        confidence: float,
        stats: TradeStat | None = None,
        vix: float | None = None,
        now: datetime | None = None,
        market_open: datetime | None = None,
        market_close: datetime | None = None,
        method: Method | None = None,
    ) -> int:
        method = method or self.cfg.get("method", "half_kelly")

        if method == "fixed_fractional":
            frac = self.cfg.get("fixed_fraction", 0.05)
        elif method == "vol_target":
            target = self.cfg.get("vol_target_annual", 0.15)
            frac = min(target / max(0.01, _annualize_estimate(stats)), 0.20)
        else:
            frac = self.kelly_fraction(stats) if stats else 0.0

        # Always scale by confidence so weak signals get smaller size.
        frac *= float(np.clip(confidence, 0.0, 1.0))

        # VIX scaling.
        frac *= self.vix_scalar(vix)

        # Open/close dampening.
        if now and market_open and market_close:
            frac *= self.time_of_day_scalar(now, market_open, market_close)

        # Hard cap on single-position fraction (safety).
        frac = min(frac, self.cfg.get("kelly_max_fraction", 0.10))

        if frac <= 0 or price <= 0 or equity <= 0:
            return 0
        notional = equity * frac
        return max(0, int(notional // price))


def _annualize_estimate(stats: TradeStat | None) -> float:
    if stats is None:
        return 0.20
    # Crude annualized vol estimate from avg_loss as a proxy.
    return max(0.05, stats.avg_loss * np.sqrt(252))
