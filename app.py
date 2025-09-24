import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta
from io import BytesIO

# local modules
from data.fetch_data import get_data
from simulation.simulator import run_simulation
from strategies.momentum import momentum_strategy
from strategies.mean_reversion import mean_reversion_strategy
from utils.plot import plot_trades_matplotlib, show_trade_table

st.set_page_config(page_title="Trading Simulator", layout="wide")
st.title("Trading Simulator — Momentum vs. Mean Reversion")

# --- helper: save a matplotlib figure to PNG bytes (for downloads)
def fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    return buf.read()

# --- minimal state ---
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"  # default
if "start_date" not in st.session_state:
    st.session_state.start_date = date.today() - timedelta(days=365)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()

with st.sidebar:
    st.header("Controls")

    # Quick tickers
    st.caption("Quick tickers")
    col_qt1, col_qt2, col_qt3 = st.columns(3)
    if col_qt1.button("AAPL"):
        st.session_state.selected_ticker = "AAPL"; st.rerun()
    if col_qt2.button("TSLA"):
        st.session_state.selected_ticker = "TSLA"; st.rerun()
    if col_qt3.button("SPY"):
        st.session_state.selected_ticker = "SPY"; st.rerun()

    ticker = st.text_input("Ticker", value=st.session_state.selected_ticker).upper().strip()
    if ticker:
        st.session_state.selected_ticker = ticker

    # Quick ranges
    st.caption("Quick ranges")
    qrow1c1, qrow1c2 = st.columns(2)
    if qrow1c1.button("3M"):
        st.session_state.start_date = date.today() - timedelta(days=90); st.rerun()
    if qrow1c2.button("YTD"):
        st.session_state.start_date = date(date.today().year, 1, 1); st.rerun()
    qrow2c1, qrow2c2 = st.columns(2)
    if qrow2c1.button("1Y"):
        st.session_state.start_date = date.today() - timedelta(days=365); st.rerun()
    if qrow2c2.button("5Y"):
        st.session_state.start_date = date.today() - timedelta(days=365 * 5); st.rerun()

    # date inputs (keys differ from state to avoid widget mutation errors)
    start = st.date_input("Start date", value=st.session_state.start_date, key="start_date_input")
    end = st.date_input("End date", value=st.session_state.end_date, key="end_date_input")
    st.session_state.start_date = start
    st.session_state.end_date = end

    show_ma = st.checkbox("Show moving average line", value=True)
    run_btn = st.button("Run Backtest", type="primary")

    st.divider()
    st.subheader("Batch run (no charts)")
    batch_tickers = st.text_input("Tickers", value="AAPL, MSFT, GOOG")
    run_batch = st.button("Run Batch Summary")

# state-backed vars
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
            # strategies with simple simulator
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

                png_bytes_mom = fig_to_png_bytes(fig)
                st.download_button(
                    label="Download Momentum chart (PNG)",
                    data=png_bytes_mom,
                    file_name=f"{ticker}_momentum_{start}_{end}.png",
                    mime="image/png",
                )

            with middle:
                fig2, ax2 = plt.subplots(figsize=(7, 4))
                title2 = f"{ticker} — Mean Reversion\nProfit: ${stats_rev['profit']:.2f} | Trades: {stats_rev['trades']}"
                plot_trades_matplotlib(
                    df.copy(), trades_rev,
                    title=title2, show_ma=show_ma, ax=ax2, cumulative_stats=stats_rev
                )
                st.pyplot(fig2, clear_figure=True)

                png_bytes_rev = fig_to_png_bytes(fig2)
                st.download_button(
                    label="Download Mean Reversion chart (PNG)",
                    data=png_bytes_rev,
                    file_name=f"{ticker}_mean_reversion_{start}_{end}.png",
                    mime="image/png",
                )

            with right:
                fig_table, ax_table = plt.subplots(figsize=(7, 5))
                show_trade_table(ax_table, trades_mom, trades_rev, title=f"{ticker} Trades")
                st.pyplot(fig_table, clear_figure=True)

            # ---- Mini metrics
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

            # Benchmark tiles
            st.markdown("&nbsp;")
            c1, c2, c3 = st.columns(3)
            c1.metric("Benchmark Profit ($)", f"{bh_profit:,.2f}")
            c2.metric("Benchmark Return (%)", f"{bh_return_pct:.2f}")
            beats = "Yes" if (stats_mom["profit"] > bh_profit) or (stats_rev["profit"] > bh_profit) else "No"
            c3.metric("Beats Buy & Hold?", beats)

            st.markdown("&nbsp;")
            comp1, comp2 = st.columns(2)
            mom_ret = (stats_mom.get("profit", 0.0) / 10_000.0) * 100.0
            rev_ret = (stats_rev.get("profit", 0.0) / 10_000.0) * 100.0
            better_profit = "Momentum" if stats_mom.get("profit", 0.0) >= stats_rev.get("profit", 0.0) else "Mean Reversion"
            profit_delta = stats_mom.get("profit", 0.0) - stats_rev.get("profit", 0.0)
            comp1.metric("Better Profit", better_profit, delta=f"${profit_delta:,.2f}")
            better_return = "Momentum" if mom_ret >= rev_ret else "Mean Reversion"
            ret_delta = mom_ret - rev_ret
            comp2.metric("Higher Return (%)", better_return, delta=f"{ret_delta:.2f} pp")

            # downloads (trade logs)
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

# ----- Batch Summary (v2): returns + buy&hold + winner + CSV -----
if run_batch:
    st.markdown("---")
    st.subheader("Batch Summary (returns + winner)")

    def run_one(tk: str):
        tk = tk.strip().upper()
        if not tk:
            return None

        dfb = get_data(tk, start_date=str(start), end_date=str(end))
        if dfb is None or dfb.empty:
            return {
                "Ticker": tk,
                "Mom Return (%)": None,
                "MR Return (%)": None,
                "Buy&Hold (%)": None,
                "Winner": "No data",
            }

        t_m, s_m = run_simulation(dfb.copy(), momentum_strategy)
        t_r, s_r = run_simulation(dfb.copy(), mean_reversion_strategy)

        mom_ret = (s_m.get("profit", 0.0) / 10_000.0) * 100.0
        mr_ret  = (s_r.get("profit", 0.0) / 10_000.0) * 100.0

        # buy & hold return %
        try:
            start_px = float(dfb["Close"].iloc[0])
            end_px   = float(dfb["Close"].iloc[-1])
            bh_ret   = ((end_px / start_px) - 1.0) * 100.0
        except Exception:
            bh_ret = None

        # pick a winner by highest return %
        candidates = {
            "Momentum": mom_ret if mom_ret is not None else float("-inf"),
            "MeanRev": mr_ret if mr_ret is not None else float("-inf"),
            "Buy&Hold": bh_ret if bh_ret is not None else float("-inf"),
        }
        winner = max(candidates, key=candidates.get)

        return {
            "Ticker": tk,
            "Mom Return (%)": None if mom_ret is None else round(mom_ret, 2),
            "MR Return (%)":  None if mr_ret  is None else round(mr_ret, 2),
            "Buy&Hold (%)":   None if bh_ret  is None else round(bh_ret, 2),
            "Winner": winner,
        }

    tickers_list = [t for t in (batch_tickers.split(",") if batch_tickers else [])]
    rows = [r for tk in tickers_list if (r := run_one(tk)) is not None]

    if rows:
        df_batch = pd.DataFrame(rows)
        st.dataframe(df_batch, use_container_width=True)

        # CSV download
        csv_bytes = df_batch.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download batch summary CSV",
            data=csv_bytes,
            file_name=f"batch_summary_{str(start)}_{str(end)}.csv",
            mime="text/csv",
        )
    else:
        st.info("No tickers to run. Add some symbols like: `AAPL, MSFT, GOOG`")