"""
AlphaEngine Streamlit Dashboard.

A read-only analytics dashboard. Reads from the Postgres trade-log database
(or sqlite for local) and renders:
  * Equity curve
  * Daily P&L histogram
  * Open positions
  * Recent trades with realized P&L
  * Per-strategy attribution
  * Signal heatmap across the universe (last evaluation per symbol)
  * Risk events (cooldowns, stops, vetoes)

Run locally:
    streamlit run alphaengine/web/streamlit_app.py

Or deploy free to Streamlit Cloud:
    1. Push this repo to GitHub
    2. https://streamlit.io/cloud -> New app -> point at alphaengine/web/streamlit_app.py
    3. Add DATABASE_URL secret in app settings

This dashboard is observational; it does NOT execute trades. The Flask app is
the control plane (start/stop, settings, webhooks). Streamlit is the analytics
plane.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AlphaEngine Analytics",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

DATABASE_URL = os.getenv("DATABASE_URL")


# ---------------- DB helpers ----------------

@st.cache_resource
def get_engine(url: str):
    from sqlalchemy import create_engine
    return create_engine(url, pool_pre_ping=True, future=True)


@st.cache_data(ttl=60)
def query(sql: str, _engine) -> pd.DataFrame:
    return pd.read_sql(sql, _engine)


def banner_no_db():
    st.warning(
        "DATABASE_URL is not configured. The Streamlit dashboard reads from the "
        "Postgres trade-log database. Set DATABASE_URL in your environment or in "
        "Streamlit Cloud secrets, then run `python scripts/migrate_db.py` once."
    )


# ---------------- pages ----------------

def page_overview(eng):
    st.title("AlphaEngine: Overview")

    col1, col2, col3, col4 = st.columns(4)

    snaps = query(
        "SELECT * FROM portfolio_snapshot ORDER BY ts DESC LIMIT 1000",
        eng,
    )
    if snaps.empty:
        st.info("No portfolio snapshots yet.")
        return

    latest = snaps.iloc[0]
    col1.metric("Equity", f"${latest['equity']:,.2f}")
    col2.metric("Daily P&L",
                f"${latest['daily_pnl']:,.2f}",
                delta=f"{latest['daily_pnl']:+,.2f}")
    col3.metric("Open Positions", int(latest["open_positions"]))
    col4.metric("Drawdown", f"{latest['drawdown_pct']*100:.2f}%")

    st.subheader("Equity Curve")
    snaps_sorted = snaps.sort_values("ts")
    chart_df = snaps_sorted.set_index("ts")[["equity"]]
    st.line_chart(chart_df, height=320)

    st.subheader("Daily P&L")
    daily = (
        snaps_sorted.assign(date=pd.to_datetime(snaps_sorted["ts"]).dt.date)
        .groupby("date")["daily_pnl"].last()
        .reset_index()
    )
    st.bar_chart(daily.set_index("date"), height=240)


def page_trades(eng):
    st.title("Trades")
    trades = query(
        "SELECT * FROM trade_log ORDER BY exit_ts DESC LIMIT 500",
        eng,
    )
    if trades.empty:
        st.info("No closed trades yet.")
        return

    win_rate = (trades["realized_pnl"] > 0).mean() * 100
    avg_win = trades.loc[trades["realized_pnl"] > 0, "realized_pnl"].mean()
    avg_loss = trades.loc[trades["realized_pnl"] < 0, "realized_pnl"].mean()
    total_pnl = trades["realized_pnl"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Realized P&L", f"${total_pnl:,.2f}")
    c2.metric("Win Rate", f"{win_rate:.1f}%")
    c3.metric("Avg Win", f"${avg_win:,.2f}" if pd.notna(avg_win) else "-")
    c4.metric("Avg Loss", f"${avg_loss:,.2f}" if pd.notna(avg_loss) else "-")

    st.subheader("Recent trades")
    show_cols = ["symbol", "side", "qty", "entry_price", "exit_price",
                 "realized_pnl", "strategy", "exit_reason", "exit_ts"]
    st.dataframe(trades[show_cols], height=480, use_container_width=True)

    st.subheader("Per-strategy attribution")
    by_strat = trades.groupby("strategy")["realized_pnl"].agg(
        ["sum", "count", "mean"]
    ).rename(columns={"sum": "total_pnl", "count": "n", "mean": "avg_pnl"})
    st.bar_chart(by_strat["total_pnl"], height=240)
    st.dataframe(by_strat, use_container_width=True)


def page_signals(eng):
    st.title("Signals")
    sigs = query(
        "SELECT * FROM signal_log ORDER BY ts DESC LIMIT 2000",
        eng,
    )
    if sigs.empty:
        st.info("No signal evaluations yet.")
        return

    latest_per_symbol = (
        sigs.sort_values("ts").groupby("symbol").tail(1)
        .sort_values("ts", ascending=False)
    )
    st.subheader("Latest signal per symbol")
    show = latest_per_symbol[["symbol", "direction", "confidence",
                              "regime", "actioned", "ts"]]
    st.dataframe(show, use_container_width=True)

    st.subheader("Signal volume over time")
    sigs["date"] = pd.to_datetime(sigs["ts"]).dt.date
    by_day = sigs.groupby("date").size().rename("count")
    st.bar_chart(by_day, height=200)

    st.subheader("Confidence distribution (last 7 days)")
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    recent = sigs[pd.to_datetime(sigs["ts"], utc=True) > cutoff]
    if not recent.empty:
        st.bar_chart(
            recent["confidence"].round(1).value_counts().sort_index(),
            height=240,
        )


def page_positions(eng):
    st.title("Open positions")
    snaps = query(
        "SELECT * FROM portfolio_snapshot ORDER BY ts DESC LIMIT 1",
        eng,
    )
    if snaps.empty:
        st.info("No portfolio snapshot yet.")
        return
    sect = snaps.iloc[0]["sector_exposure"] or {}
    if isinstance(sect, str):
        import json
        sect = json.loads(sect)
    if not sect:
        st.info("No sector exposure recorded.")
        return
    df = pd.DataFrame(
        sorted(sect.items(), key=lambda x: -x[1]),
        columns=["sector", "notional"],
    )
    st.bar_chart(df.set_index("sector"), height=300)
    st.dataframe(df, use_container_width=True)


def page_risk(eng):
    st.title("Risk events")
    events = query(
        "SELECT * FROM risk_event ORDER BY ts DESC LIMIT 200",
        eng,
    )
    if events.empty:
        st.success("No risk events recorded. Either you're early days, or things are quiet.")
        return

    by_type = events.groupby("event_type").size().rename("count")
    st.bar_chart(by_type, height=200)

    st.dataframe(
        events[["ts", "event_type", "symbol", "detail"]],
        use_container_width=True, height=480,
    )


# ---------------- router ----------------

def main():
    st.sidebar.title("AlphaEngine")
    st.sidebar.caption("Read-only analytics. Use the Flask dashboard at port 8000 to control the engine.")

    if not DATABASE_URL:
        banner_no_db()
        return

    try:
        eng = get_engine(DATABASE_URL)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return

    page = st.sidebar.radio(
        "View",
        ["Overview", "Trades", "Signals", "Positions", "Risk"],
        index=0,
    )

    if page == "Overview":
        page_overview(eng)
    elif page == "Trades":
        page_trades(eng)
    elif page == "Signals":
        page_signals(eng)
    elif page == "Positions":
        page_positions(eng)
    elif page == "Risk":
        page_risk(eng)


if __name__ == "__main__":
    main()
