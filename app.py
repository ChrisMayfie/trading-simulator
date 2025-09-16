import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta

# Import local modules
from data.fetch_data import get_data
from strategies.momentum import momentum_strategy
from strategies.mean_reversion import mean_reversion_strategy
from simulation.simulator import run_simulation
from utils.plot import plot_trades_matplotlib, show_trade_table

st.set_page_config(page_title="Trading Simulator — Web App", layout="wide")

st.title("Trading Simulator — Momentum vs. Mean Reversion")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    col1, col2 = st.columns(2)
    with col1:
        end = st.date_input("End date", value=date.today())
    with col2:
        start = st.date_input("Start date", value=end - timedelta(days=365))
    strategies = st.multiselect("Strategies", ["Momentum", "Mean Reversion"], default=["Momentum", "Mean Reversion"])
    show_ma = st.checkbox("Show moving average (10d)", value=True)
    run_btn = st.button("Run Backtest", type="primary")

# Main area
if run_btn:
    try:
        with st.spinner("Fetching data..."):
            df = get_data(ticker, start_date=str(start), end_date=str(end))

        if df is None or df.empty:
            st.warning("No data returned. Try a different date range or ticker.")
        else:
            results = []
            if "Momentum" in strategies:
                trades_mom, stats_mom = run_simulation(df.copy(), momentum_strategy, label="Momentum")
                results.append(("Momentum", trades_mom, stats_mom))
            if "Mean Reversion" in strategies:
                trades_rev, stats_rev = run_simulation(df.copy(), mean_reversion_strategy, label="Mean Reversion")
                results.append(("Mean Reversion", trades_rev, stats_rev))

            # Prepare three columns for up to two plots + table
            left, middle, right = st.columns(3)

            # Plot each selected strategy (up to two)
            for idx, (name, trades, stats) in enumerate(results[:2]):
                fig, ax = plt.subplots(figsize=(7, 4))
                title = f"{ticker} — {name}\nProfit: ${stats['profit']:.2f} | Trades: {stats['trades']}"
                plot_trades_matplotlib(df.copy(), trades, title=title, show_ma=show_ma, ax=ax, cumulative_stats=stats)
                if idx == 0:
                    with left: st.pyplot(fig, clear_figure=True)
                elif idx == 1:
                    with middle: st.pyplot(fig, clear_figure=True)

            # Comparison table if both strategies are present
            if len(results) >= 2:
                (_, t1, _), (_, t2, _) = results[0], results[1]
                fig_table, ax_table = plt.subplots(figsize=(7, 4))
                show_trade_table(ax_table, t1, t2, title=f"{ticker} Trades")
                with right:
                    st.pyplot(fig_table, clear_figure=True)

            # Downloads
            st.subheader("Downloads")
            for name, trades, _ in results:
                csv = trades.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download {ticker} {name} trades CSV",
                    data=csv,
                    file_name=f"{ticker}_{name.lower().replace(' ', '_')}_trades.csv",
                    mime="text/csv",
                )

            # Metrics
            st.subheader("Summary")
            for name, _, stats in results:
                m1, m2, m3 = st.columns(3)
                with m1: st.metric(f"{name} — Profit ($)", f"{stats.get('profit', 0):.2f}")
                with m2: st.metric(f"{name} — Trades", f"{stats.get('trades', 0)}")
                with m3: st.metric(f"{name} — Return (%)", f"{stats.get('return_pct', 0):.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

st.caption("Tip: Use the sidebar to adjust ticker and dates. Wide layout is enabled for best viewing.")