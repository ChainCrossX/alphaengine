"""
Smoke tests for AlphaEngine. No live API calls. Uses synthetic data so the
suite passes deterministically in CI and on a fresh checkout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaengine import indicators as ind
from alphaengine.signal_engine import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    SignalEngine,
)
from alphaengine.position_sizer import PositionSizer, TradeStat
from alphaengine.risk_manager import PortfolioState, RiskManager


def _synthetic_df(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=0.0005, scale=0.012, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.01, size=n))
    low = close * (1 - rng.uniform(0, 0.01, size=n))
    open_ = close * (1 + rng.normal(0, 0.003, size=n))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_imports():
    """Every module imports without error."""
    from alphaengine import (
        config, data_ingestion, indicators, signal_engine, ml_model,
        risk_manager, position_sizer, execution_engine, portfolio_manager,
        alerting, news_data, backtest, main,
    )
    assert all([
        config, data_ingestion, indicators, signal_engine, ml_model,
        risk_manager, position_sizer, execution_engine, portfolio_manager,
        alerting, news_data, backtest, main,
    ])


def test_indicators_shapes():
    df = _synthetic_df()
    feats = ind.feature_frame(df).dropna()
    assert len(feats) > 100
    for col in ["rsi_14", "macd", "signal", "hist", "adx",
                "bb_upper", "bb_lower", "atr_14", "dc_upper", "rv_20"]:
        assert col in feats.columns
        assert feats[col].notna().any()
    # RSI bounded
    assert feats["rsi_14"].between(0, 100).all()
    # ADX nonneg
    assert (feats["adx"].dropna() >= 0).all()


def test_signal_engine_runs():
    df = _synthetic_df()
    cfg = {
        "ensemble_threshold": 0.0,
        "min_confirming_strategies": 1,
        "high_vol_quantile": 0.75,
        "low_vol_quantile": 0.35,
        "weights": {
            "momentum": 0.4, "mean_reversion": 0.3, "breakout": 0.3,
            "ml_model": 0.0, "sentiment": 0.0, "relative_strength": 0.0,
        },
        "strategies": {
            "momentum": {"rsi_lower": 35, "rsi_upper": 65, "adx_min_trend": 10,
                         "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
                         "rsi_period": 14, "adx_period": 14},
            "mean_reversion": {"vwap_zscore_threshold": 1.0,
                               "bb_period": 20, "bb_stddev": 2.0},
            "breakout": {"volume_multiplier": 1.0, "donchian_period": 20},
        },
    }
    engine = SignalEngine(cfg)
    engine.register(MomentumStrategy())
    engine.register(MeanReversionStrategy())
    engine.register(BreakoutStrategy())
    sig = engine.evaluate("TEST", df)
    assert sig is not None
    assert sig.symbol == "TEST"
    assert sig.regime in ("low_vol", "high_vol", "neutral")


def test_signal_engine_requires_two_categories():
    """With only one strategy registered, signal must be flat regardless of confidence."""
    df = _synthetic_df()
    cfg = {
        "ensemble_threshold": 0.0,
        "min_confirming_strategies": 2,
        "high_vol_quantile": 0.75,
        "low_vol_quantile": 0.35,
        "weights": {"momentum": 1.0, "mean_reversion": 0, "breakout": 0,
                    "ml_model": 0, "sentiment": 0, "relative_strength": 0},
        "strategies": {
            "momentum": {"rsi_lower": 35, "rsi_upper": 65, "adx_min_trend": 1,
                         "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
                         "rsi_period": 14, "adx_period": 14},
        },
    }
    engine = SignalEngine(cfg)
    engine.register(MomentumStrategy())
    sig = engine.evaluate("TEST", df)
    assert sig.direction == 0


def test_risk_manager_position_caps():
    cfg = {
        "max_portfolio_drawdown_pct": 0.08,
        "max_single_position_loss_pct": 0.03,
        "max_single_position_size_pct": 0.10,
        "max_sector_exposure_pct": 0.30,
        "max_daily_loss_pct": 0.025,
        "max_open_positions": 10,
        "trailing_stop_activation_pct": 0.02,
        "trailing_stop_distance_pct": 0.012,
        "hard_stop_loss_pct": 0.03,
        "cooldown_after_breach_minutes": 1,
    }
    rm = RiskManager(cfg)
    state = PortfolioState(
        equity=100_000, peak_equity=100_000, cash=100_000,
        open_positions=0, daily_pnl=0,
        sector_exposure={}, positions_by_symbol={},
    )
    # 15% of equity must be vetoed.
    decision = rm.check_new_position("AAPL", "long", 15_000, "tech", state)
    assert decision.decision == "veto"
    # 8% of equity passes.
    decision = rm.check_new_position("AAPL", "long", 8_000, "tech", state)
    assert decision.decision == "approve"


def test_risk_manager_drawdown_breaker():
    cfg = {
        "max_portfolio_drawdown_pct": 0.08,
        "max_single_position_loss_pct": 0.03,
        "max_single_position_size_pct": 0.20,
        "max_sector_exposure_pct": 0.50,
        "max_daily_loss_pct": 0.05,
        "max_open_positions": 10,
        "trailing_stop_activation_pct": 0.02,
        "trailing_stop_distance_pct": 0.012,
        "hard_stop_loss_pct": 0.03,
        "cooldown_after_breach_minutes": 1,
    }
    rm = RiskManager(cfg)
    state = PortfolioState(
        equity=90_000, peak_equity=100_000, cash=90_000,
        open_positions=0, daily_pnl=0,
        sector_exposure={}, positions_by_symbol={},
    )
    decision = rm.check_new_position("AAPL", "long", 5_000, "tech", state)
    assert decision.decision == "veto"
    assert any("drawdown" in r for r in decision.reasons)


def test_position_sizer_caps():
    cfg = {
        "method": "half_kelly",
        "kelly_max_fraction": 0.10,
        "vix_scaling": {"enabled": False},
        "intraday_dampening": {"open_close_scalar": 0.5,
                               "open_window_minutes": 15,
                               "close_window_minutes": 15},
    }
    ps = PositionSizer(cfg)
    # Reasonable stats.
    stats = TradeStat(win_rate=0.55, avg_win=0.02, avg_loss=0.01)
    qty = ps.size(equity=100_000, price=100, confidence=1.0, stats=stats)
    notional = qty * 100
    # Hard cap is 10% of equity.
    assert notional <= 10_000


def test_backtester_runs():
    """End-to-end backtest on synthetic data, no API."""
    from alphaengine.backtest import Backtester
    from alphaengine.signal_engine import SignalEngine, MomentumStrategy, \
        MeanReversionStrategy, BreakoutStrategy

    df = _synthetic_df(n=300, seed=11)

    class FakeData:
        def get_bars(self, symbol, timeframe="1Day", start=None, end=None, **_):
            return df

    bt = Backtester(
        cfg={
            "initial_capital": 100_000,
            "slippage_bps": 3,
            "commission_per_trade": 0,
            "sec_fee_per_million_notional": 8,
            "taf_fee_per_share_sold": 0.000166,
            "risk": {
                "max_portfolio_drawdown_pct": 0.20,
                "max_daily_loss_pct": 0.10,
                "hard_stop_loss_pct": 0.03,
                "max_open_positions": 5,
                "max_single_position_size_pct": 0.15,
            },
        },
        data=FakeData(),
    )

    cfg = {
        "ensemble_threshold": 0.3,
        "min_confirming_strategies": 1,
        "high_vol_quantile": 0.75,
        "low_vol_quantile": 0.35,
        "weights": {"momentum": 0.5, "mean_reversion": 0.3, "breakout": 0.2,
                    "ml_model": 0, "sentiment": 0, "relative_strength": 0},
        "strategies": {
            "momentum": {"rsi_lower": 35, "rsi_upper": 65, "adx_min_trend": 10,
                         "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
                         "rsi_period": 14, "adx_period": 14},
            "mean_reversion": {"vwap_zscore_threshold": 1.5,
                               "bb_period": 20, "bb_stddev": 2.0},
            "breakout": {"volume_multiplier": 1.0, "donchian_period": 20},
        },
    }
    engine = SignalEngine(cfg)
    engine.register(MomentumStrategy())
    engine.register(MeanReversionStrategy())
    engine.register(BreakoutStrategy())
    res = bt.run(symbols=["TEST"], start="2023-01-01", end="2024-01-01",
                 signal_engine=engine)
    assert res.equity_curve is not None and not res.equity_curve.empty
    assert "sharpe" in res.metrics
    assert "max_drawdown" in res.metrics


def test_news_aggregator_no_keys_safe():
    """Without API keys, the aggregator must not throw."""
    from alphaengine.news_data import NewsAggregator
    aggr = NewsAggregator()
    n = aggr.for_symbol("AAPL")
    assert n.symbol == "AAPL"
    assert isinstance(n.score, float)
