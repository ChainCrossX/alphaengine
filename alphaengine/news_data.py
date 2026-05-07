"""
News + government data aggregation.

Sources (each optional, gated by API key presence):
  * NewsAPI            - general news headlines per symbol
  * Alpha Vantage      - news-sentiment endpoint (already scored)
  * SEC EDGAR          - latest filings per ticker (8-K, 10-Q, 10-K)
  * FRED               - macro releases that move the market (CPI, payrolls, FOMC)

Output: a single normalized score in [-1, 1] per symbol, plus a list of headlines
and filings for the dashboard. The SignalEngine treats this as a *veto-only*
overlay; we never enter a trade based on news alone, because the market has
already moved by the time the headline reaches a public API.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .logger import get_logger

log = get_logger(__name__)


@dataclass
class NewsItem:
    source: str
    title: str
    url: str
    published_at: datetime
    sentiment: float = 0.0     # in [-1, 1]
    relevance: float = 1.0     # in [0, 1]


@dataclass
class FilingItem:
    cik: str
    form: str
    filed_at: datetime
    url: str


@dataclass
class SymbolNews:
    symbol: str
    headlines: list[NewsItem] = field(default_factory=list)
    filings: list[FilingItem] = field(default_factory=list)
    score: float = 0.0          # aggregated, in [-1, 1]


_POSITIVE_WORDS = {
    "beats", "beat", "surges", "surge", "rally", "rallies", "upgrades",
    "upgrade", "raises", "expands", "record", "strong", "growth", "gain",
    "profit", "wins", "buyback", "outperforms",
}
_NEGATIVE_WORDS = {
    "miss", "missed", "drops", "plunges", "downgrade", "downgrades", "cuts",
    "lawsuit", "fraud", "investigation", "probe", "loss", "warns", "guidance",
    "weak", "decline", "fall", "halt", "recall", "bankruptcy",
}


def _heuristic_score(text: str) -> float:
    if not text:
        return 0.0
    t = text.lower()
    pos = sum(1 for w in _POSITIVE_WORDS if w in t)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in t)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / max(1, pos + neg)


class NewsAggregator:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.newsapi_key = os.getenv("NEWSAPI_KEY") or self.cfg.get("newsapi_key")
        self.av_key = os.getenv("ALPHAVANTAGE_KEY") or self.cfg.get("alphavantage_key")
        self.sec_user_agent = os.getenv(
            "SEC_USER_AGENT",
            "AlphaEngine research@example.com",
        )

    # -------- NewsAPI --------

    def newsapi_headlines(self, symbol: str, hours: int = 24) -> list[NewsItem]:
        if not self.newsapi_key:
            return []
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "from": since,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 25,
            "apiKey": self.newsapi_key,
        }
        try:
            r = requests.get(url, params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning(f"newsapi failed for {symbol}: {e}")
            return []
        items: list[NewsItem] = []
        for a in data.get("articles", []):
            title = a.get("title") or ""
            score = _heuristic_score(title + " " + (a.get("description") or ""))
            try:
                pub = datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00"))
            except Exception:
                pub = datetime.now(timezone.utc)
            items.append(NewsItem(
                source=a.get("source", {}).get("name", "newsapi"),
                title=title, url=a.get("url", ""), published_at=pub,
                sentiment=score,
            ))
        return items

    # -------- Alpha Vantage news sentiment --------

    def alphavantage_sentiment(self, symbol: str) -> list[NewsItem]:
        if not self.av_key:
            return []
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": self.av_key,
            "limit": 25,
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning(f"alphavantage failed for {symbol}: {e}")
            return []
        items: list[NewsItem] = []
        for f in data.get("feed", []):
            title = f.get("title") or ""
            try:
                pub = datetime.strptime(f["time_published"], "%Y%m%dT%H%M%S") \
                    .replace(tzinfo=timezone.utc)
            except Exception:
                pub = datetime.now(timezone.utc)
            ticker_block = next(
                (t for t in f.get("ticker_sentiment", [])
                 if t.get("ticker", "").upper() == symbol.upper()),
                None,
            )
            sentiment = float(ticker_block["ticker_sentiment_score"]) \
                if ticker_block else float(f.get("overall_sentiment_score", 0))
            relevance = float(ticker_block["relevance_score"]) \
                if ticker_block else 1.0
            items.append(NewsItem(
                source=f.get("source", "alphavantage"),
                title=title,
                url=f.get("url", ""),
                published_at=pub,
                sentiment=max(-1.0, min(1.0, sentiment)),
                relevance=max(0.0, min(1.0, relevance)),
            ))
        return items

    # -------- SEC EDGAR filings --------

    def sec_filings(self, symbol: str, days: int = 7) -> list[FilingItem]:
        try:
            tickers_resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": self.sec_user_agent},
                timeout=8,
            )
            tickers_resp.raise_for_status()
            tickers_data = tickers_resp.json()
        except Exception as e:
            log.warning(f"sec ticker map failed: {e}")
            return []

        cik = None
        for entry in tickers_data.values():
            if entry.get("ticker", "").upper() == symbol.upper():
                cik = str(entry["cik_str"]).zfill(10)
                break
        if not cik:
            return []

        try:
            r = requests.get(
                f"https://data.sec.gov/submissions/CIK{cik}.json",
                headers={"User-Agent": self.sec_user_agent},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning(f"sec submissions failed for {symbol}: {e}")
            return []

        items: list[FilingItem] = []
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        for form, acc, d in zip(forms, accessions, dates):
            try:
                filed = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if filed < cutoff:
                continue
            acc_clean = acc.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{acc}-index.htm"
            items.append(FilingItem(cik=cik, form=form, filed_at=filed, url=url))
        return items

    # -------- aggregate --------

    def for_symbol(self, symbol: str) -> SymbolNews:
        headlines = self.alphavantage_sentiment(symbol) or self.newsapi_headlines(symbol)
        filings = self.sec_filings(symbol)

        if not headlines:
            score = 0.0
        else:
            weights = [h.relevance for h in headlines]
            scores = [h.sentiment for h in headlines]
            total_w = sum(weights) or 1.0
            score = sum(s * w for s, w in zip(scores, weights)) / total_w
            score = max(-1.0, min(1.0, score))

        # 8-K filings are material; treat as a small negative tilt by default
        # because they are filed for unscheduled material events more often than not.
        if any(f.form.startswith("8-K") for f in filings):
            score -= 0.15
            score = max(-1.0, min(1.0, score))

        return SymbolNews(symbol=symbol, headlines=headlines, filings=filings, score=score)

    def score_provider(self, symbol: str):
        """Return a callable suitable for SentimentStrategy(score_provider=...)."""
        def provider() -> float:
            try:
                return self.for_symbol(symbol).score
            except Exception as e:
                log.warning(f"news provider error for {symbol}: {e}")
                return 0.0
        return provider
