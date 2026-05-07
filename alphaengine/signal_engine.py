"""
Signal engine. Each strategy returns a Score with direction and confidence in [0, 1].
The ensemble combines them with regime-aware weights and a hard rule:
NEVER fire a trade unless at least 2 different strategy categories confirm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from . import indicators as ind
from .logger import get_logger

log = get_logger(__name__)


DIRECTIONS = {1: "long", -1: "short", 0: "flat"}


@dataclass
class Score:
    name: str
    category: str
    direction: int        # -1, 0, 1
    confidence: float     # 0..1
    reasons: list[str] = field(default_factory=list)


@dataclass
class Signal:
    symbol: str
    direction: int
    confidence: float
    components: list[Score]
    regime: str
    reasons: list[str]

    @property
    def is_actionable(self) -> bool:
        return self.direction != 0 and self.confidence > 0


# ---------- strategies ----------

class Strategy:
    name = "base"
    category = "base"

    def score(self, df: pd.DataFrame, params: dict) -> Score:
        raise NotImplementedError


class MomentumStrategy(Strategy):
    name = "momentum"
    category = "trend"

    def score(self, df: pd.DataFrame, params: dict) -> Score:
        rsi_lo = params["rsi_lower"]
        rsi_hi = params["rsi_upper"]
        adx_min = params["adx_min_trend"]

        last = df.iloc[-1]
        macd_hist = last["hist"]
        adx_val = last["adx"]
        rsi_val = last["rsi_14"]
        plus_di, minus_di = last["+di"], last["-di"]
        roc_val = last["roc_12"]

        reasons = []
        direction = 0
        c = 0.0

        if pd.isna(adx_val) or adx_val < adx_min:
            return Score(self.name, self.category, 0, 0.0, ["adx_below_threshold"])

        bullish = macd_hist > 0 and plus_di > minus_di and rsi_val > 50 and roc_val > 0
        bearish = macd_hist < 0 and plus_di < minus_di and rsi_val < 50 and roc_val < 0

        if bullish:
            direction = 1
            c = float(np.clip(
                0.4 + 0.3 * (adx_val - adx_min) / 30
                + 0.2 * np.tanh(macd_hist * 5)
                + 0.1 * (rsi_val - 50) / (rsi_hi - 50),
                0, 1
            ))
            reasons = [f"adx={adx_val:.1f}", f"rsi={rsi_val:.1f}", "macd_pos", "roc_pos"]
        elif bearish:
            direction = -1
            c = float(np.clip(
                0.4 + 0.3 * (adx_val - adx_min) / 30
                + 0.2 * np.tanh(-macd_hist * 5)
                + 0.1 * (50 - rsi_val) / (50 - rsi_lo),
                0, 1
            ))
            reasons = [f"adx={adx_val:.1f}", f"rsi={rsi_val:.1f}", "macd_neg", "roc_neg"]

        return Score(self.name, self.category, direction, c, reasons)


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    category = "reversion"

    def score(self, df: pd.DataFrame, params: dict) -> Score:
        zthr = params["vwap_zscore_threshold"]
        last = df.iloc[-1]
        pct_b = last.get("bb_pct_b", np.nan)
        rsi_val = last.get("rsi_14", np.nan)

        # Local zscore from BB position
        try:
            z = ind.vwap_zscore(df, 20).iloc[-1]
        except Exception:
            z = np.nan

        reasons = []
        direction = 0
        c = 0.0

        if pd.isna(z) or pd.isna(pct_b):
            return Score(self.name, self.category, 0, 0.0, ["insufficient_data"])

        # Stretched short-term: fade
        if z > zthr and rsi_val > 65 and pct_b > 0.95:
            direction = -1
            c = float(np.clip((z - zthr) / 2 + 0.3, 0, 1))
            reasons = [f"z={z:.2f}", f"rsi={rsi_val:.1f}", "above_upper_band"]
        elif z < -zthr and rsi_val < 35 and pct_b < 0.05:
            direction = 1
            c = float(np.clip((-z - zthr) / 2 + 0.3, 0, 1))
            reasons = [f"z={z:.2f}", f"rsi={rsi_val:.1f}", "below_lower_band"]

        return Score(self.name, self.category, direction, c, reasons)


class BreakoutStrategy(Strategy):
    name = "breakout"
    category = "trend"

    def score(self, df: pd.DataFrame, params: dict) -> Score:
        vol_mult = params["volume_multiplier"]
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else last

        upper = last.get("dc_upper", np.nan)
        lower = last.get("dc_lower", np.nan)
        vol = last["volume"]
        avg_vol = df["volume"].rolling(20).mean().iloc[-1]

        reasons = []
        direction = 0
        c = 0.0

        if pd.isna(upper) or pd.isna(avg_vol) or avg_vol == 0:
            return Score(self.name, self.category, 0, 0.0, ["insufficient_data"])

        vol_ratio = vol / avg_vol
        if last["close"] >= upper and prev["close"] < upper and vol_ratio >= vol_mult:
            direction = 1
            c = float(np.clip(0.4 + 0.3 * (vol_ratio - vol_mult), 0, 1))
            reasons = [f"break_upper", f"vol_ratio={vol_ratio:.2f}"]
        elif last["close"] <= lower and prev["close"] > lower and vol_ratio >= vol_mult:
            direction = -1
            c = float(np.clip(0.4 + 0.3 * (vol_ratio - vol_mult), 0, 1))
            reasons = [f"break_lower", f"vol_ratio={vol_ratio:.2f}"]

        return Score(self.name, self.category, direction, c, reasons)


class RelativeStrengthStrategy(Strategy):
    """Long the symbol when it is outperforming the benchmark, short otherwise."""
    name = "relative_strength"
    category = "trend"

    def __init__(self, benchmark_df: pd.DataFrame | None = None):
        self.benchmark_df = benchmark_df

    def score(self, df: pd.DataFrame, params: dict) -> Score:
        if self.benchmark_df is None or len(self.benchmark_df) < 20:
            return Score(self.name, self.category, 0, 0.0, ["no_benchmark"])
        sym_ret = df["close"].pct_change(20).iloc[-1]
        bench_ret = self.benchmark_df["close"].pct_change(20).iloc[-1]
        if pd.isna(sym_ret) or pd.isna(bench_ret):
            return Score(self.name, self.category, 0, 0.0, ["nan"])
        diff = sym_ret - bench_ret
        direction = 1 if diff > 0.01 else (-1 if diff < -0.01 else 0)
        c = float(np.clip(abs(diff) * 5, 0, 1))
        return Score(self.name, self.category, direction, c,
                     [f"rs_diff={diff*100:.2f}%"])


class MLStrategy(Strategy):
    name = "ml_model"
    category = "ml"

    def __init__(self, model=None):
        self.model = model

    def score(self, df: pd.DataFrame, params: dict) -> Score:
        if self.model is None:
            return Score(self.name, self.category, 0, 0.0, ["no_model"])
        try:
            prob_up, prob_down = self.model.predict_last(df)
        except Exception as e:
            return Score(self.name, self.category, 0, 0.0, [f"model_error:{e}"])
        min_conf = params.get("min_confidence", 0.55)
        if prob_up >= min_conf:
            return Score(self.name, self.category, 1, float(prob_up),
                         [f"p_up={prob_up:.2f}"])
        if prob_down >= min_conf:
            return Score(self.name, self.category, -1, float(prob_down),
                         [f"p_down={prob_down:.2f}"])
        return Score(self.name, self.category, 0, 0.0, [f"low_conf"])


class SentimentStrategy(Strategy):
    """Veto-only by default. Returns 0 unless a strong tilt is provided."""
    name = "sentiment"
    category = "alt"

    def __init__(self, score_provider=None):
        self.provider = score_provider

    def score(self, df: pd.DataFrame, params: dict) -> Score:
        if self.provider is None:
            return Score(self.name, self.category, 0, 0.0, ["no_provider"])
        try:
            s = float(self.provider())  # in [-1, 1]
        except Exception:
            return Score(self.name, self.category, 0, 0.0, ["provider_error"])
        if abs(s) < 0.4:
            return Score(self.name, self.category, 0, 0.0, [f"neutral={s:.2f}"])
        return Score(
            self.name, self.category,
            1 if s > 0 else -1, min(1.0, abs(s)), [f"sent={s:.2f}"]
        )


# ---------- ensemble ----------

class SignalEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.strategies: list[Strategy] = []
        self.weights: dict[str, float] = dict(cfg["weights"])

    def register(self, strategy: Strategy) -> None:
        self.strategies.append(strategy)

    # Regime detection: classify recent realized vol against history.
    def _regime(self, df: pd.DataFrame) -> str:
        rv = ind.realized_vol(df["close"], 20)
        if rv.dropna().empty:
            return "neutral"
        last = rv.iloc[-1]
        hi = rv.quantile(self.cfg["high_vol_quantile"])
        lo = rv.quantile(self.cfg["low_vol_quantile"])
        if last >= hi:
            return "high_vol"
        if last <= lo:
            return "low_vol"
        return "neutral"

    def _regime_weights(self, regime: str) -> dict[str, float]:
        w = dict(self.weights)
        if regime == "low_vol":
            w["momentum"] *= 1.3
            w["breakout"] *= 1.2
            w["mean_reversion"] *= 0.6
        elif regime == "high_vol":
            w["mean_reversion"] *= 1.4
            w["momentum"] *= 0.7
            w["breakout"] *= 0.7
        # normalize
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    def evaluate(self, symbol: str, df: pd.DataFrame) -> Signal:
        feats = ind.feature_frame(df).dropna()
        if len(feats) < 50:
            return Signal(symbol, 0, 0.0, [], "insufficient", ["insufficient_history"])

        regime = self._regime(feats)
        weights = self._regime_weights(regime)

        params_map = self.cfg.get("strategies", {})
        scores: list[Score] = []
        for strat in self.strategies:
            params = params_map.get(strat.name, {})
            try:
                s = strat.score(feats, params)
            except Exception as e:
                log.error(f"{strat.name} failed for {symbol}: {e}")
                s = Score(strat.name, strat.category, 0, 0.0, [f"error:{e}"])
            scores.append(s)

        # Sentiment is veto-only when negative; never the sole reason to enter.
        sentiment = next((s for s in scores if s.name == "sentiment"), None)

        # Combine: weighted directional vote.
        weighted_long = 0.0
        weighted_short = 0.0
        for s in scores:
            w = weights.get(s.name, 0.0)
            if s.direction == 1:
                weighted_long += w * s.confidence
            elif s.direction == -1:
                weighted_short += w * s.confidence

        net = weighted_long - weighted_short
        confidence = min(1.0, abs(net))
        direction = 1 if net > 0 else (-1 if net < 0 else 0)

        # Hard rule: 2+ confirming categories.
        confirming_categories = {
            s.category for s in scores
            if s.direction == direction and s.confidence > 0 and s.category != "alt"
        }
        min_confirming = self.cfg.get("min_confirming_strategies", 2)
        if len(confirming_categories) < min_confirming:
            return Signal(
                symbol, 0, 0.0, scores, regime,
                [f"only_{len(confirming_categories)}_confirming_categories"],
            )

        # Ensemble threshold gate.
        threshold = self.cfg.get("ensemble_threshold", 0.72)
        if confidence < threshold:
            return Signal(
                symbol, 0, 0.0, scores, regime,
                [f"conf_{confidence:.2f}_below_{threshold}"],
            )

        # Sentiment veto: if sentiment strongly disagrees, kill the signal.
        if sentiment and sentiment.direction != 0 and sentiment.direction != direction \
                and sentiment.confidence > 0.7:
            return Signal(
                symbol, 0, 0.0, scores, regime,
                [f"sentiment_veto={sentiment.reasons}"],
            )

        reasons = [
            f"regime={regime}",
            f"net={net:.3f}",
            f"confirming={sorted(confirming_categories)}",
        ]
        return Signal(symbol, direction, confidence, scores, regime, reasons)
