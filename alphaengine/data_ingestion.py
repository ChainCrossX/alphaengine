"""Data ingestion. Alpaca primary, yfinance fallback. Built to fail gracefully."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from tenacity import (
        retry, retry_if_exception_type, stop_after_attempt, wait_exponential
    )
    _TENACITY = True
except ImportError:  # graceful no-op if tenacity is missing
    _TENACITY = False
    def retry(*a, **kw):  # type: ignore
        def deco(fn):
            return fn
        return deco
    def stop_after_attempt(*a, **kw):  # type: ignore
        return None
    def wait_exponential(*a, **kw):  # type: ignore
        return None
    def retry_if_exception_type(*a, **kw):  # type: ignore
        return None

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.live import StockDataStream
    from alpaca.trading.client import TradingClient
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

try:
    import yfinance as yf
    _YF = True
except ImportError:
    _YF = False
    yf = None  # type: ignore

from .logger import get_logger

log = get_logger(__name__)


_TF_MAP = {
    "1Min": ("1m", (1, "Minute")),
    "5Min": ("5m", (5, "Minute")),
    "15Min": ("15m", (15, "Minute")),
    "1Hour": ("1h", (1, "Hour")),
    "1Day": ("1d", (1, "Day")),
}


class DataError(Exception):
    """Raised when both primary and fallback data sources fail."""


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_bps(self) -> float:
        if self.mid <= 0:
            return 0.0
        return (self.ask - self.bid) / self.mid * 10_000


class DataClient:
    """
    Unified data client. All higher layers should use this rather than vendor SDKs
    so that fallback and caching are consistent.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        cache_dir: str | Path = ".cache",
        paper: bool = True,
    ):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.paper = paper

        self._alpaca = None
        self._trading = None
        if _ALPACA_AVAILABLE and self.api_key and self.api_secret:
            try:
                self._alpaca = StockHistoricalDataClient(self.api_key, self.api_secret)
                self._trading = TradingClient(self.api_key, self.api_secret, paper=paper)
            except Exception as e:
                log.warning(f"Alpaca init failed, falling back to yfinance: {e}")

    # -------- historical bars --------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: datetime | None = None,
        end: datetime | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return OHLCV bars indexed by UTC timestamp."""
        if timeframe not in _TF_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = end - timedelta(days=365)

        cache_path = self._cache_path(symbol, timeframe, start, end)
        if use_cache and cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                cache_path.unlink(missing_ok=True)

        df: pd.DataFrame | None = None
        if self._alpaca is not None:
            try:
                df = self._fetch_alpaca(symbol, timeframe, start, end)
            except Exception as e:
                log.warning(f"Alpaca fetch failed for {symbol} {timeframe}: {e}")

        if df is None or df.empty:
            df = self._fetch_yfinance(symbol, timeframe, start, end)

        if df is None or df.empty:
            raise DataError(f"No data for {symbol} {timeframe}")

        df = self._normalize(df)
        if use_cache:
            try:
                df.to_parquet(cache_path)
            except Exception as e:
                log.warning(f"Cache write failed: {e}")
        return df

    def get_bars_multi(
        self, symbols: Iterable[str], timeframe: str = "1Day", **kwargs
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for s in symbols:
            try:
                out[s] = self.get_bars(s, timeframe=timeframe, **kwargs)
            except Exception as e:
                log.error(f"Failed to fetch {s}: {e}")
        return out

    # -------- quotes --------

    def get_latest_quote(self, symbol: str) -> Quote | None:
        if self._alpaca is None:
            return None
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            resp = self._alpaca.get_stock_latest_quote(req)
            q = resp[symbol]
            return Quote(
                symbol=symbol,
                bid=float(q.bid_price),
                ask=float(q.ask_price),
                bid_size=float(q.bid_size),
                ask_size=float(q.ask_size),
                timestamp=q.timestamp,
            )
        except Exception as e:
            log.warning(f"Latest quote failed for {symbol}: {e}")
            return None

    # -------- streaming --------

    def stream(self, symbols: Iterable[str], on_bar) -> None:
        """Subscribe to live minute bars. Blocks. Caller is responsible for thread."""
        if not _ALPACA_AVAILABLE or not self.api_key:
            raise DataError("Streaming requires Alpaca credentials")
        stream = StockDataStream(self.api_key, self.api_secret)

        async def handler(bar):
            await on_bar(bar)

        for s in symbols:
            stream.subscribe_bars(handler, s)
        stream.run()

    # -------- internals --------

    def _fetch_alpaca(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        amount, unit = _TF_MAP[timeframe][1]
        tf = TimeFrame(amount, getattr(TimeFrameUnit, unit))
        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=tf, start=start, end=end
        )
        bars = self._alpaca.get_stock_bars(req).df
        if bars.empty:
            return bars
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level=0)
        return bars

    def _fetch_yfinance(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        if not _YF:
            return pd.DataFrame()
        yf_tf = _TF_MAP[timeframe][0]
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=yf_tf,
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        return df

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        keep = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]]
        df = df.dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df

    def _cache_path(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> Path:
        key = f"{symbol}_{timeframe}_{start.date()}_{end.date()}.parquet"
        return self.cache_dir / key
