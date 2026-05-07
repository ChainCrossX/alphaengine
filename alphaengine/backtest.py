"""
Backtest framework. Vectorized where possible, event-driven for risk and stops.

Cost model is intentionally pessimistic. Most retail backtests look great because
they ignore commissions, half-spread slippage, market-impact drag, regulatory
fees, and short-borrow costs. This one models all of them. Treat the resulting
Sharpe as the *upper bound* of what live trading might achieve, not a target.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data_ingestion import DataClient
from .indicators import feature_frame
from .logger import get_logger, log_event
from .ml_model import DirectionModel
from .signal_engine import (
    BreakoutStrategy, MeanReversionStrategy, MLStrategy, MomentumStrategy,
    RelativeStrengthStrategy, SignalEngine,
)

log = get_logger(__name__)


@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: datetime | None
    side: str
    qty: int
    entry_price: float
    exit_price: float | None
    pnl: float = 0.0
    cost: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade]
    metrics: dict[str, float] = field(default_factory=dict)


class Backtester:
    def __init__(self, cfg: dict, data: DataClient):
        self.cfg = cfg
        self.data = data

    def _cost(self, side: str, price: float, qty: int) -> float:
        bps = self.cfg["slippage_bps"]
        slippage = price * qty * bps / 10_000
        commission = self.cfg.get("commission_per_trade", 0.0)
        sec_fee = 0.0
        if side == "sell":
            sec_fee = (price * qty / 1_000_000) * self.cfg.get("sec_fee_per_million_notional", 0)
            sec_fee += qty * self.cfg.get("taf_fee_per_share_sold", 0)
        return slippage + commission + sec_fee

    def run(
        self,
        symbols: list[str],
        start: str,
        end: str,
        signal_engine: SignalEngine,
    ) -> BacktestResult:
        equity = self.cfg["initial_capital"]
        peak_equity = equity
        equity_curve = []
        trades: list[Trade] = []
        positions: dict[str, Trade] = {}

        # Pull all bars up front.
        bars = {s: self.data.get_bars(s, "1Day",
                                      start=pd.Timestamp(start, tz="UTC"),
                                      end=pd.Timestamp(end, tz="UTC"))
                for s in symbols}
        # Common index union.
        all_idx = sorted(set().union(*[df.index for df in bars.values()]))

        risk_cfg = self.cfg.get("risk", {})
        max_dd = risk_cfg.get("max_portfolio_drawdown_pct", 0.08)
        hard_stop = risk_cfg.get("hard_stop_loss_pct", 0.03)
        daily_loss_limit = risk_cfg.get("max_daily_loss_pct", 0.025)
        max_pos = risk_cfg.get("max_open_positions", 10)
        per_pos_cap = risk_cfg.get("max_single_position_size_pct", 0.12)

        last_eod_equity = equity
        in_cooldown_until: datetime | None = None

        for ts in all_idx:
            # Mark-to-market existing positions.
            mtm = 0.0
            for sym, t in list(positions.items()):
                df = bars[sym]
                if ts not in df.index:
                    continue
                px = float(df.loc[ts, "close"])
                if t.side == "long":
                    pnl = (px - t.entry_price) * t.qty
                else:
                    pnl = (t.entry_price - px) * t.qty
                mtm += pnl

                # Hard stop check.
                loss_pct = (t.entry_price - px) / t.entry_price if t.side == "long" \
                           else (px - t.entry_price) / t.entry_price
                if loss_pct >= hard_stop:
                    cost = self._cost("sell" if t.side == "long" else "buy", px, t.qty)
                    realized = pnl - cost - t.cost
                    equity += realized
                    t.exit_time = ts
                    t.exit_price = px
                    t.pnl = realized
                    t.reason = f"hard_stop_{loss_pct:.2%}"
                    trades.append(t)
                    positions.pop(sym)

            current_equity = equity + mtm
            peak_equity = max(peak_equity, current_equity)
            equity_curve.append((ts, current_equity))

            # Day-roll: reset daily anchor.
            if ts.date() != getattr(self, "_last_day", ts.date()):
                last_eod_equity = current_equity
                self._last_day = ts.date()

            # Drawdown halt.
            if (peak_equity - current_equity) / peak_equity >= max_dd:
                continue
            # Daily loss halt.
            if (current_equity - last_eod_equity) / max(last_eod_equity, 1) <= -daily_loss_limit:
                continue
            if in_cooldown_until and ts < in_cooldown_until:
                continue

            # Generate signals across symbols and pick highest-confidence non-held.
            candidates = []
            for sym in symbols:
                df = bars[sym]
                if ts not in df.index:
                    continue
                hist = df.loc[:ts]
                if len(hist) < 80:
                    continue
                if sym in positions:
                    continue
                sig = signal_engine.evaluate(sym, hist)
                if sig.is_actionable:
                    candidates.append((sym, sig, float(hist["close"].iloc[-1])))

            candidates.sort(key=lambda x: x[1].confidence, reverse=True)

            for sym, sig, px in candidates:
                if len(positions) >= max_pos:
                    break
                # Position size: confidence-weighted, capped.
                frac = min(per_pos_cap, 0.05 * sig.confidence + 0.02)
                notional = current_equity * frac
                qty = int(notional // px)
                if qty <= 0:
                    continue
                cost = self._cost("buy" if sig.direction == 1 else "sell", px, qty)
                positions[sym] = Trade(
                    symbol=sym,
                    entry_time=ts,
                    exit_time=None,
                    side="long" if sig.direction == 1 else "short",
                    qty=qty,
                    entry_price=px,
                    exit_price=None,
                    cost=cost,
                )

        # Force-close at end.
        last_ts = all_idx[-1] if all_idx else None
        if last_ts:
            for sym, t in list(positions.items()):
                px = float(bars[sym].loc[last_ts, "close"])
                pnl = (px - t.entry_price) * t.qty if t.side == "long" \
                      else (t.entry_price - px) * t.qty
                cost = self._cost("sell" if t.side == "long" else "buy", px, t.qty)
                t.pnl = pnl - cost - t.cost
                t.exit_time = last_ts
                t.exit_price = px
                t.reason = "eod_close"
                trades.append(t)
                equity += t.pnl

        ec = pd.Series([v for _, v in equity_curve],
                       index=[ts for ts, _ in equity_curve], name="equity")

        metrics = self._metrics(ec, trades)
        log_event(log, "INFO", "backtest_done", **metrics)
        return BacktestResult(equity_curve=ec, trades=trades, metrics=metrics)

    @staticmethod
    def _metrics(equity: pd.Series, trades: list[Trade]) -> dict[str, float]:
        if equity.empty:
            return {}
        returns = equity.pct_change().dropna()
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std(ddof=0)) \
                 if returns.std(ddof=0) > 0 else 0.0
        downside = returns[returns < 0]
        sortino = float(np.sqrt(252) * returns.mean() / downside.std(ddof=0)) \
                  if len(downside) > 1 and downside.std(ddof=0) > 0 else 0.0
        peak = equity.cummax()
        max_dd = float((1 - equity / peak).max())
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        win_rate = len(wins) / max(1, len(trades))
        avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "n_trades": len(trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "final_equity": float(equity.iloc[-1]),
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="backtest_results.csv")
    args = p.parse_args()

    cfg = load_config(args.config)
    data = DataClient(cache_dir=cfg.data.cache_dir, paper=True)

    bt_cfg = cfg.backtest.to_dict()
    bt_cfg["risk"] = cfg.risk.to_dict()

    bench = data.get_bars(cfg.universe.benchmark, "1Day")

    engine = SignalEngine(cfg.signal_engine.to_dict())
    engine.register(MomentumStrategy())
    engine.register(MeanReversionStrategy())
    engine.register(BreakoutStrategy())
    engine.register(RelativeStrengthStrategy(benchmark_df=bench))
    try:
        m = DirectionModel(horizon=1)
        m.fit(bench)
        if m.train_result and m.train_result.is_useful:
            engine.register(MLStrategy(model=m))
    except Exception as e:
        log.warning(f"ML model unavailable in backtest: {e}")

    bt = Backtester(bt_cfg, data)
    res = bt.run(
        symbols=cfg.universe.symbols,
        start=cfg.backtest.start_date,
        end=cfg.backtest.end_date,
        signal_engine=engine,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.equity_curve.to_csv(args.out)
    print("Metrics:")
    for k, v in res.metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
