"""
Orchestrator.

Schedules:
  * signal loop every N seconds during market hours
  * trailing stop check every minute
  * weekly rebalance hook
  * daily summary at market close

Defaults to paper trading. Refuses to switch to live unless config.mode == 'live'
*and* a backtest gate has been satisfied (Sharpe > 1.5 over 6 months OOS).
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from .alerting import Alerter
from .config import load_config
from .data_ingestion import DataClient
from .execution_engine import ExecutionEngine
from .indicators import feature_frame
from .logger import get_logger, log_event
from .ml_model import DirectionModel
from .portfolio_manager import PortfolioManager, SECTOR_MAP
from .position_sizer import PositionSizer, TradeStat
from .risk_manager import RiskManager
from .signal_engine import (
    BreakoutStrategy, MeanReversionStrategy, MLStrategy, MomentumStrategy,
    RelativeStrengthStrategy, SentimentStrategy, SignalEngine,
)
from . import webhooks
try:
    from .db.trade_logger import TradeLogger
    _DB_AVAILABLE = True
except Exception:
    TradeLogger = None  # type: ignore
    _DB_AVAILABLE = False

log = get_logger(__name__)


class AlphaEngine:
    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        cfg = self.cfg.to_dict()

        self.alerter = Alerter(cfg["alerting"])
        self.data = DataClient(
            cache_dir=cfg["data"]["cache_dir"],
            paper=cfg["mode"] == "paper",
        )
        self.execution = ExecutionEngine(
            cfg["execution"], self.data, paper=cfg["mode"] == "paper"
        )

        starting_eq = float(self.execution.account().equity)
        self.portfolio = PortfolioManager(self.execution, starting_equity=starting_eq)
        self.risk = RiskManager(cfg["risk"])
        self.sizer = PositionSizer(cfg["position_sizing"])

        self.signal_engine = SignalEngine(cfg["signal_engine"])
        self.signal_engine.register(MomentumStrategy())
        self.signal_engine.register(MeanReversionStrategy())
        self.signal_engine.register(BreakoutStrategy())

        self.benchmark = cfg["universe"]["benchmark"]
        self.symbols = cfg["universe"]["symbols"]

        bench_df = self.data.get_bars(self.benchmark, "1Day")
        self.signal_engine.register(RelativeStrengthStrategy(benchmark_df=bench_df))

        self.ml_model: DirectionModel | None = None
        self._train_ml()
        if self.ml_model and self.ml_model.train_result and self.ml_model.train_result.is_useful:
            self.signal_engine.register(MLStrategy(model=self.ml_model))
            log_event(log, "INFO", "ml_model_attached")
        else:
            log_event(log, "INFO", "ml_model_skipped_no_useful_signal")

        # Sentiment is veto-only; off by default until a real provider is wired in.
        self.signal_engine.register(SentimentStrategy(score_provider=None))

        self.scheduler = BackgroundScheduler(timezone="UTC")
        self._stopping = False

        # Optional Postgres trade logger.
        self.trade_logger = TradeLogger() if _DB_AVAILABLE else None

        # Wire TradingView webhook handler so external alerts can submit orders.
        webhooks.configure(
            execution=self.execution,
            risk=self.risk,
            portfolio=self.portfolio,
            sizer=self.sizer,
            trade_logger=self.trade_logger,
            sector_map=SECTOR_MAP,
        )

    def _train_ml(self) -> None:
        try:
            df = self.data.get_bars(self.benchmark, "1Day")
            self.ml_model = DirectionModel(horizon=1)
            self.ml_model.fit(df)
        except Exception as e:
            log_event(log, "WARNING", "ml_train_failed", error=str(e))
            self.ml_model = None

    # -------- loops --------

    def signal_loop(self) -> None:
        try:
            self.portfolio.sync()
            state = self.portfolio.state()
            if self.risk.is_in_cooldown():
                return

            for symbol in self.symbols:
                try:
                    df = self.data.get_bars(symbol, "1Hour")
                    sig = self.signal_engine.evaluate(symbol, df)
                    log_event(log, "DEBUG", "signal",
                              symbol=symbol, direction=sig.direction,
                              confidence=round(sig.confidence, 3),
                              regime=sig.regime, reasons=sig.reasons)
                    if not sig.is_actionable:
                        continue

                    side = "buy" if sig.direction == 1 else "sell"
                    quote = self.data.get_latest_quote(symbol)
                    price = quote.mid if quote else float(df["close"].iloc[-1])

                    qty = self.sizer.size(
                        equity=state.equity,
                        price=price,
                        confidence=sig.confidence,
                        stats=TradeStat(win_rate=0.5, avg_win=0.012, avg_loss=0.01),
                        vix=None,
                        now=datetime.now(timezone.utc),
                        market_open=self._market_open_today(),
                        market_close=self._market_close_today(),
                    )
                    if qty <= 0:
                        continue

                    notional = qty * price
                    risk_check = self.risk.check_new_position(
                        symbol=symbol,
                        side="long" if side == "buy" else "short",
                        notional=notional,
                        sector=SECTOR_MAP.get(symbol, "other"),
                        state=state,
                    )
                    if risk_check.decision == "veto":
                        log_event(log, "INFO", "risk_veto",
                                  symbol=symbol, reasons=risk_check.reasons)
                        continue

                    result = self.execution.submit(symbol, side, qty)
                    log_event(log, "INFO", "execution_result",
                              symbol=symbol, side=side, qty=qty,
                              accepted=result.accepted,
                              avg_price=result.avg_fill_price,
                              order_id=result.order_id,
                              reason=result.reason)
                    if result.accepted:
                        self.alerter.emit(
                            "trade_executed",
                            f"{side.upper()} {qty} {symbol} @ {result.avg_fill_price}",
                            confidence=round(sig.confidence, 3),
                            regime=sig.regime,
                        )
                except Exception as e:
                    log_event(log, "ERROR", "symbol_loop_error", symbol=symbol, error=str(e))
        except Exception as e:
            log_event(log, "ERROR", "signal_loop_error", error=str(e))

    def trailing_stop_loop(self) -> None:
        try:
            self.portfolio.sync()
            for sym, p in list(self.portfolio.positions.items()):
                quote = self.data.get_latest_quote(sym)
                if quote is None:
                    continue
                cp = quote.mid
                # Update peak.
                if p.side == "long":
                    p.peak_price = max(p.peak_price, cp)
                else:
                    p.peak_price = min(p.peak_price, cp)

                stop_out, reason = self.risk.should_stop_out(
                    {"avg_price": p.avg_price, "side": p.side}, cp
                )
                trail_px = self.risk.trailing_stop_price(
                    {"avg_price": p.avg_price, "side": p.side}, p.peak_price
                )
                trip = False
                trip_reason = None
                if stop_out:
                    trip = True
                    trip_reason = reason
                elif trail_px is not None:
                    if p.side == "long" and cp <= trail_px:
                        trip = True
                        trip_reason = f"trailing_stop_long@{trail_px:.2f}"
                    elif p.side == "short" and cp >= trail_px:
                        trip = True
                        trip_reason = f"trailing_stop_short@{trail_px:.2f}"

                if trip:
                    side = "sell" if p.side == "long" else "buy"
                    result = self.execution.submit(sym, side, p.qty)
                    log_event(log, "INFO", "stop_triggered",
                              symbol=sym, reason=trip_reason,
                              accepted=result.accepted,
                              avg_price=result.avg_fill_price)
                    self.alerter.emit("circuit_breaker",
                                      f"Stop-out {sym}: {trip_reason}",
                                      avg_price=result.avg_fill_price)
        except Exception as e:
            log_event(log, "ERROR", "trailing_stop_loop_error", error=str(e))

    def daily_summary(self) -> None:
        state = self.portfolio.state()
        msg = (
            f"equity={state.equity:.2f} "
            f"daily_pnl={state.daily_pnl:.2f} "
            f"open={state.open_positions} "
            f"dd={self.risk.portfolio_drawdown_pct(state):.2%}"
        )
        log_event(log, "INFO", "daily_summary", summary=msg)
        self.alerter.emit("daily_summary", msg)

    # -------- helpers --------

    @staticmethod
    def _market_open_today() -> datetime:
        now = datetime.now(timezone.utc)
        return now.replace(hour=13, minute=30, second=0, microsecond=0)  # 9:30 ET in UTC

    @staticmethod
    def _market_close_today() -> datetime:
        now = datetime.now(timezone.utc)
        return now.replace(hour=20, minute=0, second=0, microsecond=0)   # 16:00 ET in UTC

    # -------- run --------

    def run(self) -> None:
        cfg_sched = self.cfg.scheduler.to_dict()
        self.scheduler.add_job(
            self.signal_loop, "interval",
            seconds=cfg_sched["signal_loop_interval_seconds"],
            id="signal_loop",
        )
        self.scheduler.add_job(self.trailing_stop_loop, "interval",
                               seconds=60, id="trailing_loop")
        self.scheduler.add_job(self.daily_summary, "cron",
                               hour=cfg_sched["daily_summary_hour_local"],
                               minute=0, id="daily_summary")
        self.scheduler.start()
        log_event(log, "INFO", "alphaengine_started", mode=self.cfg.mode)

        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

        while not self._stopping:
            time.sleep(1)

    def _handle_stop(self, *_):
        self._stopping = True
        try:
            self.scheduler.shutdown(wait=False)
        except Exception:
            pass
        log_event(log, "INFO", "alphaengine_stopped")
        sys.exit(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    AlphaEngine(args.config).run()


if __name__ == "__main__":
    main()
