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
from utils.plot import plot_trades_matplotlib  # table graphic not used in this goal

st.set_page_config(page_title="Trading Simulator", layout="wide")
st.title("Trading Simulator — Momentum vs. Mean Reversion")

# -------- utilities --------
def fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    return buf.read()

def _close_series(df: pd.DataFrame) -> pd.Series:
    """Return a 1-D float Series of Close prices with tz-naive DateTime index."""
    s = None
    if isinstance(df, pd.DataFrame):
        if "Close" in df.columns:
            s = df["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
        elif df.shape[1] >= 1:
            s = df.iloc[:, 0]
    if s is None:
        s = pd.Series(getattr(df, "values", df).ravel(), index=getattr(df, "index", None))
    s = pd.to_numeric(s.squeeze(), errors="coerce")
    try:
        s.index = pd.to_datetime(s.index).tz_localize(None)
    except Exception:
        try:
            s.index = pd.to_datetime(s.index).tz_convert(None)
        except Exception:
            s.index = pd.to_datetime(s.index, errors="coerce")
    s.name = "Close"
    return s

def equity_curve_from_trades(df_prices: pd.DataFrame, trades: pd.DataFrame, initial_equity: float = 10_000.0) -> pd.Series:
    """Long-only, all-in on BUY, flat on SELL equity curve from trade log."""
    if df_prices is None or df_prices.empty:
        return pd.Series(dtype=float)
    close = _close_series(df_prices).sort_index()

    if trades is None or trades.empty:
        return pd.Series([initial_equity] * len(close), index=close.index, name="equity")

    tr = trades.copy()
    try:
        tr.index = pd.to_datetime(tr.index, errors="coerce")
        if getattr(tr.index, "tz", None) is not None:
            tr.index = tr.index.tz_convert(None)
    except Exception:
        try:
            tr.index = tr.index.tz_localize(None)
        except Exception:
            pass
    tr = tr.sort_index()

    cash = float(initial_equity)
    shares = 0
    vals = []

    it = iter(tr.itertuples(index=True, name="T"))
    cur = next(it, None)

    for dt, px in close.items():
        while cur is not None and cur.Index == dt:
            act = str(getattr(cur, "Action", "")).lower()
            tpx = float(getattr(cur, "Price", px))
            if act == "buy" and shares == 0:
                qty = int(cash // tpx)
                if qty > 0:
                    cash -= qty * tpx
                    shares = qty
            elif act == "sell" and shares > 0:
                cash += shares * tpx
                shares = 0
            cur = next(it, None)
        vals.append(cash + shares * float(px))

    return pd.Series(vals, index=close.index, name="equity")

def equity_curve_buy_hold(df_prices: pd.DataFrame, initial_equity: float = 10_000.0) -> pd.Series:
    if df_prices is None or df_prices.empty:
        return pd.Series(dtype=float)
    close = _close_series(df_prices).sort_index()
    start = float(close.iloc[0])
    shares = int(initial_equity // start)
    rem = initial_equity - shares * start
    eq = rem + shares * close.astype(float)
    return pd.Series(eq.values, index=close.index, name="buy_hold")

def max_drawdown(eq: pd.Series) -> float:
    if eq is None or eq.empty:
        return 0.0
    cm = eq.cummax()
    dd = (eq / cm) - 1.0
    return float(dd.min())  # negative (e.g., -0.1234)

def sharpe_ratio(eq: pd.Series, periods_per_year: int = 252) -> float:
    if eq is None or len(eq) < 2:
        return 0.0
    rets = eq.pct_change().dropna()
    if rets.std() == 0:
        return 0.0
    return float((rets.mean() / rets.std()) * (periods_per_year ** 0.5))

# -------- state --------
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"
if "start_date" not in st.session_state:
    st.session_state.start_date = date.today() - timedelta(days=365)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()

with st.sidebar:
    st.header("Controls")

    # Quick tickers
    st.caption("Quick tickers")
    col_qt1, col_qt2, col_qt3 = st.columns(3)
    if col_qt1.button("AAPL"): st.session_state.selected_ticker = "AAPL"; st.rerun()
    if col_qt2.button("TSLA"): st.session_state.selected_ticker = "TSLA"; st.rerun()
    if col_qt3.button("SPY"):  st.session_state.selected_ticker = "SPY";  st.rerun()

    ticker = st.text_input("Ticker", value=st.session_state.selected_ticker).upper().strip()
    if ticker:
        st.session_state.selected_ticker = ticker

    # Quick ranges
    st.caption("Quick ranges")

    def _apply_range(start_date_val, end_date_val=None):
        end_date_val = end_date_val or st.session_state.get("end_date", date.today())
        st.session_state.start_date = start_date_val
        st.session_state.end_date = end_date_val
        st.session_state["start_date_input"] = start_date_val
        st.session_state["end_date_input"] = end_date_val
        st.rerun()

    qrow1c1, qrow1c2 = st.columns(2)
    if qrow1c1.button("3M"):
        _apply_range(date.today() - timedelta(days=90))
    if qrow1c2.button("YTD"):
        _apply_range(date(date.today().year, 1, 1))

    qrow2c1, qrow2c2 = st.columns(2)
    if qrow2c1.button("1Y"):
        _apply_range(date.today() - timedelta(days=365))
    if qrow2c2.button("5Y"):
        _apply_range(date.today() - timedelta(days=365 * 5))

    # date inputs
    start = st.date_input(
        "Start date",
        value=st.session_state.get("start_date", date.today() - timedelta(days=365)),
        key="start_date_input",
    )
    end = st.date_input(
        "End date",
        value=st.session_state.get("end_date", date.today()),
        key="end_date_input",
    )
    if start != st.session_state.start_date:
        st.session_state.start_date = start
    if end != st.session_state.end_date:
        st.session_state.end_date = end

    # Strategy parameters (compact w/ tooltips)
    st.subheader("Strategy Parameters")

    with st.expander("Momentum", expanded=False):
        mom_lookback = st.number_input(
            "Lookback (days)",
            min_value=1, max_value=252, value=20, step=1,
            help="How many past days to compare. Higher = smoother & fewer trades; lower = faster but noisier."
        )
        mom_min_gap = st.number_input(
            "Min gap vs reference (fraction)",
            min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f",
            help="Smallest relative gap to trigger a trade. Example: 0.05 = 5%."
        )

    with st.expander("Mean Reversion", expanded=False):
        mr_ma_window = st.number_input(
            "Moving-average window (days)",
            min_value=2, max_value=252, value=20, step=1,
            help="Days in the moving average. Higher = smoother & fewer trades; lower = faster but choppier."
        )
        mr_z_entry = st.number_input(
            "Z-score entry",
            min_value=0.1, max_value=5.0, value=1.0, step=0.1, format="%.1f",
            help="Enter when price is this many std-devs from the mean. Higher = rarer, more extreme setups."
        )
        mr_z_exit = st.number_input(
            "Z-score exit",
            min_value=0.05, max_value=3.0, value=0.25, step=0.05, format="%.2f",
            help="Close when distance shrinks to this level. Lower = exit sooner; higher = ride longer."
        )

    st.subheader("Portfolio")
    starting_cash = st.number_input(
        "Starting cash ($)",
        min_value=1_000, max_value=10_000_000, value=10_000, step=500,
        help="Initial equity for the backtest. Default is $10,000."
    )

    show_ma = st.checkbox("Show moving average line", value=True)
    run_btn = st.button("Run Backtest", type="primary")

    st.divider()
    st.subheader("Batch run - No Charts & CSV Download")
    batch_tickers = st.text_input("Tickers", value="AAPL, MSFT, GOOG")
    run_batch = st.button("Run Batch Summary")

# share state-backed vars
start = st.session_state.start_date
end = st.session_state.end_date
ticker = st.session_state.selected_ticker
initial_equity = float(starting_cash)  # <-- available to both single-ticker and batch paths

# -------- single-ticker run --------
if run_btn:
    try:
        with st.spinner("Fetching data..."):
            df = get_data(ticker, start_date=str(start), end_date=str(end))

        if df is None or df.empty:
            st.warning("No data returned. Try a different date range or ticker.")
        else:
            # strategies
            trades_mom, stats_mom = run_simulation(
                df.copy(),
                lambda d: momentum_strategy(d, lookback=int(mom_lookback), min_gap=float(mom_min_gap)),
                initial_equity=initial_equity
            )
            trades_rev, stats_rev = run_simulation(
                df.copy(),
                lambda d: mean_reversion_strategy(
                    d,
                    ma_window=int(mr_ma_window),
                    z_entry=float(mr_z_entry),
                    z_exit=float(mr_z_exit),
                ),
                initial_equity=initial_equity
            )

            # buy & hold benchmark
            close = _close_series(df)
            start_price = float(close.iloc[0])
            end_price = float(close.iloc[-1])
            shares_bh = int(initial_equity // start_price)
            rem_cash = initial_equity - shares_bh * start_price
            final_bh = rem_cash + shares_bh * end_price
            bh_profit = final_bh - initial_equity
            bh_return_pct = (final_bh / initial_equity - 1.0) * 100.0

            # top row: Momentum | Mean Reversion | Equity Curves
            left, middle, right = st.columns(3)

            with left:
                fig, ax = plt.subplots(figsize=(7, 4))
                title = f"{ticker} — Momentum\nProfit: ${stats_mom['profit']:.2f} | Trades: {stats_mom['trades']}"
                plot_trades_matplotlib(df.copy(), trades_mom, title=title, show_ma=show_ma, ax=ax, cumulative_stats=stats_mom)
                st.pyplot(fig, clear_figure=True)
                st.download_button(
                    label="Download Momentum chart (PNG)",
                    data=fig_to_png_bytes(fig),
                    file_name=f"{ticker}_momentum_{start}_{end}.png",
                    mime="image/png",
                )

            with middle:
                fig2, ax2 = plt.subplots(figsize=(7, 4))
                title2 = f"{ticker} — Mean Reversion\nProfit: ${stats_rev['profit']:.2f} | Trades: {stats_rev['trades']}"
                plot_trades_matplotlib(df.copy(), trades_rev, title=title2, show_ma=show_ma, ax=ax2, cumulative_stats=stats_rev)
                st.pyplot(fig2, clear_figure=True)
                st.download_button(
                    label="Download Mean Reversion chart (PNG)",
                    data=fig_to_png_bytes(fig2),
                    file_name=f"{ticker}_mean_reversion_{start}_{end}.png",
                    mime="image/png",
                )

            with right:
                st.markdown("**Normalized Equity Curves**")
                eq_mom = stats_mom.get("equity_curve")
                eq_rev = stats_rev.get("equity_curve")
                if not isinstance(eq_mom, pd.Series):
                    eq_mom = equity_curve_from_trades(df, trades_mom, initial_equity=initial_equity)
                if not isinstance(eq_rev, pd.Series):
                    eq_rev = equity_curve_from_trades(df, trades_rev, initial_equity=initial_equity)
                eq_bh  = equity_curve_buy_hold(df, initial_equity=initial_equity)

                def _norm(s: pd.Series) -> pd.Series:
                    if s is None or s.empty:
                        return s
                    base = s.iloc[0] if s.iloc[0] != 0 else 1.0
                    return s / base

                fig_ec, ax_ec = plt.subplots(figsize=(7, 4))
                if isinstance(eq_mom, pd.Series) and not eq_mom.empty:
                    ax_ec.plot(eq_mom.index, _norm(eq_mom), label="Momentum")
                if isinstance(eq_rev, pd.Series) and not eq_rev.empty:
                    ax_ec.plot(eq_rev.index, _norm(eq_rev), label="Mean Reversion")
                if isinstance(eq_bh, pd.Series) and not eq_bh.empty:
                    ax_ec.plot(eq_bh.index, _norm(eq_bh), label="Buy & Hold")
                ax_ec.set_xlabel("Date"); ax_ec.set_ylabel("Equity (normalized)")
                ax_ec.legend(loc="best"); ax_ec.grid(True, linestyle="--", alpha=0.3)
                st.pyplot(fig_ec, clear_figure=True)

                # risk metrics for single-ticker view
                mom_sharpe = sharpe_ratio(eq_mom) if isinstance(eq_mom, pd.Series) else 0.0
                rev_sharpe = sharpe_ratio(eq_rev) if isinstance(eq_rev, pd.Series) else 0.0
                mom_mdd = max_drawdown(eq_mom) * 100.0 if isinstance(eq_mom, pd.Series) else 0.0
                rev_mdd = max_drawdown(eq_rev) * 100.0 if isinstance(eq_rev, pd.Series) else 0.0

            st.markdown("---")
            st.subheader("Metrics")

            def basic_metrics(stats: dict):
                ret_pct = (stats.get("profit", 0.0) / initial_equity) * 100.0
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

            # risk tiles under each strategy
            riskA, riskB = st.columns(2)
            with riskA:
                st.caption("Momentum — Risk")
                r1, r2 = st.columns(2)
                r1.metric("Sharpe", f"{mom_sharpe:.2f}")
                r2.metric("Max DD (%)", f"{mom_mdd:.2f}")
            with riskB:
                st.caption("Mean Reversion — Risk")
                r1, r2 = st.columns(2)
                r1.metric("Sharpe", f"{rev_sharpe:.2f}")
                r2.metric("Max DD (%)", f"{rev_mdd:.2f}")

            # benchmark tiles
            st.markdown("&nbsp;")
            c1, c2, c3 = st.columns(3)
            c1.metric("Benchmark Profit ($)", f"{bh_profit:,.2f}")
            c2.metric("Benchmark Return (%)", f"{bh_return_pct:.2f}")
            beats = "Yes" if (stats_mom["profit"] > bh_profit) or (stats_rev["profit"] > bh_profit) else "No"
            c3.metric("Beats Buy & Hold?", beats)

            # comparison row
            st.markdown("&nbsp;")
            comp1, comp2 = st.columns(2)
            mom_ret = (stats_mom.get("profit", 0.0) / initial_equity) * 100.0
            rev_ret = (stats_rev.get("profit", 0.0) / initial_equity) * 100.0
            better_profit = "Momentum" if stats_mom.get("profit", 0.0) >= stats_rev.get("profit", 0.0) else "Mean Reversion"
            profit_delta = stats_mom.get("profit", 0.0) - stats_rev.get("profit", 0.0)
            comp1.metric("Better Profit", better_profit, delta=f"${profit_delta:,.2f}")
            better_return = "Momentum" if mom_ret >= rev_ret else "Mean Reversion"
            ret_delta = mom_ret - rev_ret
            comp2.metric("Higher Return (%)", better_return, delta=f"{ret_delta:.2f} pp")

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

if run_batch:
    st.markdown("---")
    st.subheader("Batch Summary (Sharpe & Max DD)")

    def run_one(tk: str):
        tk = tk.strip().upper()
        if not tk:
            return None
        dfb = get_data(tk, start_date=str(start), end_date=str(end))
        if dfb is None or dfb.empty:
            return {
                "Ticker": tk, "Mom Return (%)": "—", "MR Return (%)": "—",
                "Mom Sharpe": "—", "MR Sharpe": "—",
                "Mom Max DD (%)": "—", "MR Max DD (%)": "—",
                "Best": "No data"
            }

        t_m, s_m = run_simulation(
            dfb.copy(),
            lambda d: momentum_strategy(d, lookback=int(mom_lookback), min_gap=float(mom_min_gap)),
            initial_equity=initial_equity
        )
        t_r, s_r = run_simulation(
            dfb.copy(),
            lambda d: mean_reversion_strategy(
                d, ma_window=int(mr_ma_window), z_entry=float(mr_z_entry), z_exit=float(mr_z_exit)
            ),
            initial_equity=initial_equity
        )

        mom_ret = (s_m.get("profit", 0.0) / initial_equity) * 100.0
        mr_ret  = (s_r.get("profit", 0.0) / initial_equity) * 100.0
        mom_eq  = s_m.get("equity_curve"); mr_eq = s_r.get("equity_curve")

        mom_sharpe = sharpe_ratio(mom_eq)
        mr_sharpe  = sharpe_ratio(mr_eq)
        mom_mdd    = max_drawdown(mom_eq) * 100.0
        mr_mdd     = max_drawdown(mr_eq) * 100.0

        best_name = "Momentum" if mom_ret >= mr_ret else "MeanRev"
        best_val  = mom_ret if mom_ret >= mr_ret else mr_ret

        return {
            "Ticker": tk,
            "Mom Return (%)": f"{mom_ret:.2f}",
            "MR Return (%)": f"{mr_ret:.2f}",
            "Mom Sharpe": f"{mom_sharpe:.2f}",
            "MR Sharpe": f"{mr_sharpe:.2f}",
            "Mom Max DD (%)": f"{mom_mdd:.2f}",
            "MR Max DD (%)": f"{mr_mdd:.2f}",
            "Best": f"{best_name} ({best_val:.2f}%)"
        }

    tickers_list = [t for t in (batch_tickers.split(",") if batch_tickers else [])]
    rows = [r for tk in tickers_list if (r := run_one(tk)) is not None]

    if rows:
        df_out = pd.DataFrame(rows)
        st.dataframe(df_out, use_container_width=True)
        st.download_button(
            "Download batch summary (CSV)",
            df_out.to_csv(index=False).encode("utf-8"),
            file_name=f"batch_summary_{start}_{end}.csv",
            mime="text/csv"
        )
    else:
        st.info("No tickers to run. Add some symbols like: `AAPL, MSFT, GOOG`")
