"""
SQLAlchemy ORM models for the trade-log database.

Schema goals:
  * Every signal evaluation logged, even if no trade was placed
  * Every order submission and fill recorded with full provenance
  * Daily portfolio snapshot for performance attribution
  * Risk events (circuit breakers, stops) auditable

Use SQLite for local dev (DATABASE_URL=sqlite:///alpha.db) or Postgres in prod.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker,
)


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SignalLog(Base):
    __tablename__ = "signal_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    direction: Mapped[int] = mapped_column(Integer)            # -1, 0, 1
    confidence: Mapped[float] = mapped_column(Float)
    regime: Mapped[str] = mapped_column(String(16))
    components: Mapped[dict] = mapped_column(JSON)             # per-strategy scores
    reasons: Mapped[list] = mapped_column(JSON)
    actioned: Mapped[bool] = mapped_column(Boolean, default=False)


class OrderLog(Base):
    __tablename__ = "order_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    broker_order_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(8))               # buy/sell
    qty: Mapped[int] = mapped_column(Integer)
    order_type: Mapped[str] = mapped_column(String(16))        # limit/market/trailing
    limit_price: Mapped[Optional[float]] = mapped_column(Float)
    fill_price: Mapped[Optional[float]] = mapped_column(Float)
    filled_qty: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16))            # accepted/rejected/filled/canceled
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text)
    strategy: Mapped[Optional[str]] = mapped_column(String(32))
    signal_id: Mapped[Optional[int]] = mapped_column(ForeignKey("signal_log.id"))


class TradeLog(Base):
    """Closed trade. Realized P&L and attribution."""
    __tablename__ = "trade_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(8))
    qty: Mapped[int] = mapped_column(Integer)
    entry_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    exit_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float)
    realized_pnl: Mapped[float] = mapped_column(Float)
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    strategy: Mapped[str] = mapped_column(String(32))
    exit_reason: Mapped[str] = mapped_column(String(64))


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    open_positions: Mapped[int] = mapped_column(Integer)
    daily_pnl: Mapped[float] = mapped_column(Float)
    drawdown_pct: Mapped[float] = mapped_column(Float)
    sector_exposure: Mapped[dict] = mapped_column(JSON)


class RiskEvent(Base):
    __tablename__ = "risk_event"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    event_type: Mapped[str] = mapped_column(String(32), index=True)  # cooldown, stop_out, veto
    symbol: Mapped[Optional[str]] = mapped_column(String(16))
    detail: Mapped[dict] = mapped_column(JSON)


def make_engine(database_url: str):
    return create_engine(database_url, pool_pre_ping=True, future=True)


def make_session(database_url: str):
    eng = make_engine(database_url)
    return sessionmaker(eng, expire_on_commit=False, future=True)


def init_db(database_url: str) -> None:
    eng = make_engine(database_url)
    Base.metadata.create_all(eng)
