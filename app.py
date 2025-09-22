import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta

# local modules
from data.fetch_data import get_data
from simulation.simulator import run_simulation
from strategies.momentum import momentum_strategy
from strategies.mean_reversion import mean_reversion_strategy
from utils.plot import plot_trades_matplotlib, show_trade_table

st.set_page_config(page_title="Trading Simulator", layout="wide")
st.title("Trading Simulator — Momentum vs. Mean Reversion")

# --- minimal state ---
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"  # default
if "start_date" not in st.session_state:
    st.session_state.start_date = date.today() - timedelta(days=365)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()

with st.sidebar:
    st.header("Controls")

    # =========== Quick tickers ===========
    st.caption("Quick tickers")
    col_qt1, col_qt2, col_qt3 = st.columns(3)
    if col_qt1.button("AAPL"):
        st.session_state.selected_ticker = "AAPL"
        st.rerun()
    if col_qt2.button("TSLA"):
        st.session_state.selected_ticker = "TSLA"
        st.rerun()
    if col_qt3.button("SPY"):
        st.session_state.selected_ticker = "SPY"
        st.rerun()

    # text input stays the source of truth
    ticker = st.text_input("Ticker", value=st.session_state.selected_ticker).upper().strip()
    if ticker:
        st.session_state.selected_ticker = ticker

    # idea: buttons tweak start_date in state, then rerun; inputs read state
    st.caption("Quick ranges")
    q1, q2, q3, q4 = st.columns(4)
    if q1.button("3M"):
        st.session_state.start_date = date.today() - timedelta(days=90)
        st.rerun()
    if q2.button("YTD"):
        st.session_state.start_date = date(date.today().year, 1, 1)
        st.rerun()
    if q3.button("1Y"):
        st.session_state.start_date = date.today() - timedelta(days=365)
        st.rerun()
    if q4.button("5Y"):
        st.session_state.start_date = date.today() - timedelta(days=365 * 5)
        st.rerun()

    # render the actual inputs with different keys so buttons don’t fight them
    start = st.date_input("Start date", value=st.session_state.start_date, key="start_date_input")
    end = st.date_input("End date", value=st.session_state.end_date, key="end_date_input")

    # keep state in sync with any manual edits
    st.session_state.start_date = start
    st.session_state.end_date = end

    show_ma = st.checkbox("Show moving average line", value=True)
    run_btn = st.button("Run Backtest", type="primary")

    st.divider()
    st.subheader("Batch run (no charts)")
    batch_tickers = st.text_input("Tickers", value="AAPL, MSFT, GOOG")
    run_batch = st.button("Run Batch Summary")

# use the state-backed dates for the run
start = st.session_state.start_date
end = st.session_state.end_date
ticker = st.session_state.selected_ticker

# run on button click only
if run_btn:
    try:
        with st.spinner("Fetching data..."):
            df = get_data(ticker, start_date=str(start), end_date=str(end))

        if df is None or df.empty:
            st.warning("No data returned. Try a different date range or ticker.")
        else:
            # run both strategies with the basic simulator
            trades_mom, stats_mom = run_simulation(df.copy(), momentum_strategy)
            trades_rev, stats_rev = run_simulation(df.copy(), mean_reversion_strategy)

            # --- Simple buy & hold benchmark (metrics-only)
            initial_equity = 10_000.0
            start_price = float(df["Close"].iloc[0])
            end_price = float(df["Close"].iloc[-1])
            shares_bh = int(initial_equity // start_price)
            rem_cash = initial_equity - shares_bh * start_price
            final_bh = rem_cash + shares_bh * end_price
            bh_profit = final_bh - initial_equity
            bh_return_pct = (final_bh / initial_equity - 1.0) * 100.0

            # ========== Charts (left/middle) + Trade table (right) ==========
            left, middle, right = st.columns(3)

            with left:
                fig, ax = plt.subplots(figsize=(7, 4))
                title = f"{ticker} — Momentum\nProfit: ${stats_mom['profit']:.2f} | Trades: {stats_mom['trades']}"
                plot_trades_matplotlib(
                    df.copy(), trades_mom,
                    title=title, show_ma=show_ma, ax=ax, cumulative_stats=stats_mom
                )
                st.pyplot(fig, clear_figure=True)

            with middle:
                fig2, ax2 = plt.subplots(figsize=(7, 4))
                title2 = f"{ticker} — Mean Reversion\nProfit: ${stats_rev['profit']:.2f} | Trades: {stats_rev['trades']}"
                plot_trades_matplotlib(
                    df.copy(), trades_rev,
                    title=title2, show_ma=show_ma, ax=ax2, cumulative_stats=stats_rev
                )
                st.pyplot(fig2, clear_figure=True)

            with right:
                fig_table, ax_table = plt.subplots(figsize=(7, 5))
                show_trade_table(ax_table, trades_mom, trades_rev, title=f"{ticker} Trades")
                st.pyplot(fig_table, clear_figure=True)

            # ---- Minimal metrics (under the charts)
            st.markdown("---")
            st.subheader("Metrics (quick)")

            def basic_metrics(stats: dict):
                ret_pct = (stats.get("profit", 0.0) / 10_000.0) * 100.0
                c1, c2, c3 = st.columns(3)
                c1.metric("Profit ($)", f"{stats.get('profit', 0.0):,.2f}")
                c2.metric("Trades", f"{stats.get('trades', 0)}")
                c3.metric("Return (%)", f"{ret_pct:.2f}")

            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Momentum**")
                basic_metrics(stats_mom)
            with colB:
                st.markdown("**Mean Reversion**")
                basic_metrics(stats_rev)

            # tiny benchmark row
            st.markdown("&nbsp;")
            c1, c2, c3 = st.columns(3)
            c1.metric("Benchmark Profit ($)", f"{bh_profit:,.2f}")
            c2.metric("Benchmark Return (%)", f"{bh_return_pct:.2f}")
            beats = "Yes" if (stats_mom["profit"] > bh_profit) or (stats_rev["profit"] > bh_profit) else "No"
            c3.metric("Beats Buy & Hold?", beats)

            # downloads (kept small)
            st.subheader("Downloads")
            for name, trades in [("Momentum", trades_mom), ("Mean Reversion", trades_rev)]:
                csv = trades.reset_index().to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download {ticker} {name} trades CSV",
                    data=csv,
                    file_name=f"{ticker}_{name.lower().replace(' ', '_')}_trades.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

# ----- Batch Summary (v1): tiny table of returns, no charts -----
if run_batch:
    st.markdown("---")
    st.subheader("Batch Summary (no charts)")

    def run_one(tk: str):
        tk = tk.strip().upper()
        if not tk:
            return None
        dfb = get_data(tk, start_date=str(start), end_date=str(end))
        if dfb is None or dfb.empty:
            return {"Ticker": tk, "Mom Return (%)": "—", "MR Return (%)": "—", "Best": "No data"}

        t_m, s_m = run_simulation(dfb.copy(), momentum_strategy)
        t_r, s_r = run_simulation(dfb.copy(), mean_reversion_strategy)

        mom_ret = (s_m.get("profit", 0.0) / 10_000.0) * 100.0
        mr_ret = (s_r.get("profit", 0.0) / 10_000.0) * 100.0
        best_name = "Momentum" if mom_ret >= mr_ret else "MeanRev"
        best_val = mom_ret if mom_ret >= mr_ret else mr_ret

        return {
            "Ticker": tk,
            "Mom Return (%)": f"{mom_ret:.2f}",
            "MR Return (%)": f"{mr_ret:.2f}",
            "Best": f"{best_name} ({best_val:.2f}%)"
        }

    tickers_list = [t for t in (batch_tickers.split(",") if batch_tickers else [])]
    rows = []
    for tk in tickers_list:
        row = run_one(tk)
        if row is not None:
            rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No tickers to run. Add some symbols like: `AAPL, MSFT, GOOG`")

st.caption("Quick tickers, quick ranges, simple metrics, and a tiny batch summary. Keeping the core flow minimal.")