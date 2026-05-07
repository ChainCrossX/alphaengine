"""
Self-contained smoke test runner. No pytest required. Imports only stdlib +
numpy/pandas/pyyaml/python-dotenv/requests. Optional packages (alpaca, tenacity,
lightgbm, flask, apscheduler) are tolerated as missing. Use this to verify the
engine modules wire together correctly on any machine.
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

PASS = 0
FAIL = 0
ERRORS: list[tuple[str, str]] = []


def _check(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAIL += 1
        ERRORS.append((name, f"AssertionError: {e}"))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        FAIL += 1
        tb = traceback.format_exc(limit=2)
        ERRORS.append((name, tb))
        print(f"  FAIL  {name}: {e}")


def _synthetic(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# -------- tests --------

def t_imports():
    from alphaengine import (
        config, data_ingestion, indicators, signal_engine, ml_model,
        risk_manager, position_sizer, execution_engine, portfolio_manager,
        alerting, news_data, backtest,
    )
    assert all([
        config, data_ingestion, indicators, signal_engine, ml_model,
        risk_manager, position_sizer, execution_engine, portfolio_manager,
        alerting, news_data, backtest,
    ])


def t_indicators():
    from alphaengine import indicators as ind
    df = _synthetic()
    feats = ind.feature_frame(df).dropna()
    assert len(feats) > 100
    assert feats["rsi_14"].between(0, 100).all()
    assert (feats["adx"].dropna() >= 0).all()
    assert "bb_upper" in feats and "atr_14" in feats and "dc_upper" in feats


def t_signal_basic():
    from alphaengine.signal_engine import (
        SignalEngine, MomentumStrategy, MeanReversionStrategy, BreakoutStrategy,
    )
    df = _synthetic()
    cfg = {
        "ensemble_threshold": 0.0,
        "min_confirming_strategies": 1,
        "high_vol_quantile": 0.75,
        "low_vol_quantile": 0.35,
        "weights": {"momentum": 0.4, "mean_reversion": 0.3, "breakout": 0.3,
                    "ml_model": 0, "sentiment": 0, "relative_strength": 0},
        "strategies": {
            "momentum": {"rsi_lower": 35, "rsi_upper": 65, "adx_min_trend": 10,
                         "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
                         "rsi_period": 14, "adx_period": 14},
            "mean_reversion": {"vwap_zscore_threshold": 1.0,
                               "bb_period": 20, "bb_stddev": 2.0},
            "breakout": {"volume_multiplier": 1.0, "donchian_period": 20},
        },
    }
    eng = SignalEngine(cfg)
    eng.register(MomentumStrategy())
    eng.register(MeanReversionStrategy())
    eng.register(BreakoutStrategy())
    sig = eng.evaluate("TEST", df)
    assert sig.symbol == "TEST"
    assert sig.regime in ("low_vol", "high_vol", "neutral")


def t_signal_requires_two_categories():
    from alphaengine.signal_engine import SignalEngine, MomentumStrategy
    df = _synthetic()
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
    eng = SignalEngine(cfg)
    eng.register(MomentumStrategy())
    sig = eng.evaluate("TEST", df)
    assert sig.direction == 0


def t_risk_position_caps():
    from alphaengine.risk_manager import RiskManager, PortfolioState
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
    state = PortfolioState(equity=100_000, peak_equity=100_000, cash=100_000,
                           open_positions=0, daily_pnl=0,
                           sector_exposure={}, positions_by_symbol={})
    assert rm.check_new_position("AAPL", "long", 15_000, "tech", state).decision == "veto"
    assert rm.check_new_position("AAPL", "long", 8_000, "tech", state).decision == "approve"


def t_risk_drawdown_breaker():
    from alphaengine.risk_manager import RiskManager, PortfolioState
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
    state = PortfolioState(equity=90_000, peak_equity=100_000, cash=90_000,
                           open_positions=0, daily_pnl=0,
                           sector_exposure={}, positions_by_symbol={})
    d = rm.check_new_position("AAPL", "long", 5_000, "tech", state)
    assert d.decision == "veto"
    assert any("drawdown" in r for r in d.reasons)


def t_position_sizer_caps():
    from alphaengine.position_sizer import PositionSizer, TradeStat
    cfg = {
        "method": "half_kelly", "kelly_max_fraction": 0.10,
        "vix_scaling": {"enabled": False},
        "intraday_dampening": {"open_close_scalar": 0.5,
                               "open_window_minutes": 15,
                               "close_window_minutes": 15},
    }
    ps = PositionSizer(cfg)
    stats = TradeStat(win_rate=0.55, avg_win=0.02, avg_loss=0.01)
    qty = ps.size(equity=100_000, price=100, confidence=1.0, stats=stats)
    assert qty * 100 <= 10_000


def t_backtester_runs():
    from alphaengine.backtest import Backtester
    from alphaengine.signal_engine import (
        SignalEngine, MomentumStrategy, MeanReversionStrategy, BreakoutStrategy,
    )
    df = _synthetic(n=300, seed=11)

    class FakeData:
        def get_bars(self, symbol, timeframe="1Day", start=None, end=None, **_):
            return df

    bt = Backtester(
        cfg={"initial_capital": 100_000, "slippage_bps": 3,
             "commission_per_trade": 0,
             "sec_fee_per_million_notional": 8,
             "taf_fee_per_share_sold": 0.000166,
             "risk": {"max_portfolio_drawdown_pct": 0.20,
                      "max_daily_loss_pct": 0.10,
                      "hard_stop_loss_pct": 0.03,
                      "max_open_positions": 5,
                      "max_single_position_size_pct": 0.15}},
        data=FakeData(),
    )
    cfg = {
        "ensemble_threshold": 0.3, "min_confirming_strategies": 1,
        "high_vol_quantile": 0.75, "low_vol_quantile": 0.35,
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
    eng = SignalEngine(cfg)
    eng.register(MomentumStrategy())
    eng.register(MeanReversionStrategy())
    eng.register(BreakoutStrategy())
    res = bt.run(symbols=["TEST"], start="2023-01-01", end="2024-01-01",
                 signal_engine=eng)
    assert res.equity_curve is not None and not res.equity_curve.empty
    assert "sharpe" in res.metrics and "max_drawdown" in res.metrics


def t_news_aggregator_safe():
    from alphaengine.news_data import NewsAggregator
    aggr = NewsAggregator()
    n = aggr.for_symbol("AAPL")
    assert n.symbol == "AAPL"
    assert isinstance(n.score, float)


def t_config_load():
    from alphaengine.config import load_config
    cfg = load_config("config.yaml")
    assert cfg.mode in ("paper", "live")
    risk = cfg.risk.to_dict()
    assert risk["max_portfolio_drawdown_pct"] == 0.08
    assert risk["max_open_positions"] == 10


# -------- runner --------

TESTS = [
    ("imports", t_imports),
    ("indicators", t_indicators),
    ("config_load", t_config_load),
    ("signal_basic", t_signal_basic),
    ("signal_requires_two_categories", t_signal_requires_two_categories),
    ("risk_position_caps", t_risk_position_caps),
    ("risk_drawdown_breaker", t_risk_drawdown_breaker),
    ("position_sizer_caps", t_position_sizer_caps),
    ("backtester_runs", t_backtester_runs),
    ("news_aggregator_safe", t_news_aggregator_safe),
]


def main(run_label: str = "1"):
    global PASS, FAIL, ERRORS
    PASS = FAIL = 0
    ERRORS = []
    print(f"=== Smoke run {run_label} @ {datetime.now().isoformat(timespec='seconds')} ===")
    for name, fn in TESTS:
        _check(name, fn)
    print(f"--- run {run_label}: {PASS} passed, {FAIL} failed ---")
    if FAIL:
        for n, e in ERRORS:
            print(f"\n[{n}]\n{e}")
    return FAIL


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "1"
    sys.exit(main(label))
