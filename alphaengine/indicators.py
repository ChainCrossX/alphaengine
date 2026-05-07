"""Vectorized technical indicators. Pure functions over a price DataFrame."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return pd.DataFrame({"macd": line, "signal": sig, "hist": hist})


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder's ADX. df must have high/low/close columns."""
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    tr = _true_range(df)
    atr_n = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_n)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_n)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_n = dx.ewm(alpha=1 / period, adjust=False).mean()
    return pd.DataFrame({"adx": adx_n, "+di": plus_di, "-di": minus_di})


def bollinger(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    width = (upper - lower) / ma
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"bb_mid": ma, "bb_upper": upper, "bb_lower": lower,
                         "bb_width": width, "bb_pct_b": pct_b})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP. Resets each trading day."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    day = df.index.date if hasattr(df.index, "date") else df.index.normalize()
    grouped = pd.Series(pv).groupby(day).cumsum()
    vol = pd.Series(df["volume"]).groupby(day).cumsum()
    return (grouped / vol.replace(0, np.nan)).set_axis(df.index)


def vwap_zscore(df: pd.DataFrame, window: int = 20) -> pd.Series:
    v = vwap(df)
    diff = df["close"] - v
    return (diff - diff.rolling(window).mean()) / diff.rolling(window).std(ddof=0)


def roc(close: pd.Series, period: int = 12) -> pd.Series:
    return (close / close.shift(period) - 1) * 100


def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    mid = (upper + lower) / 2
    return pd.DataFrame({"dc_upper": upper, "dc_lower": lower, "dc_mid": mid})


def realized_vol(close: pd.Series, period: int = 20, annualize: int = 252) -> pd.Series:
    ret = np.log(close / close.shift(1))
    return ret.rolling(period).std(ddof=0) * np.sqrt(annualize)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prev_close).abs()
    c = (df["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a standard feature set used by both signals and the ML model."""
    out = df.copy()
    out["ret_1"] = out["close"].pct_change()
    out["ret_5"] = out["close"].pct_change(5)
    out["ret_20"] = out["close"].pct_change(20)
    out["rsi_14"] = rsi(out["close"], 14)
    macd_df = macd(out["close"])
    out = out.join(macd_df)
    out = out.join(adx(out, 14))
    out = out.join(bollinger(out["close"], 20, 2.0))
    out["atr_14"] = atr(out, 14)
    out["roc_12"] = roc(out["close"], 12)
    out = out.join(donchian(out, 20))
    out["rv_20"] = realized_vol(out["close"], 20)
    out["volume_z"] = (
        (out["volume"] - out["volume"].rolling(20).mean())
        / out["volume"].rolling(20).std(ddof=0)
    )
    return out
