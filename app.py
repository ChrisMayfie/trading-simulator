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
from utils.plot import plot_trades_matplotlib

# page config + title
st.set_page_config(page_title="Trading Simulator", layout="wide")
st.title("Trading Simulator — Momentum vs. Mean Reversion")

# utils
def fig_to_png_bytes(fig) -> bytes:
    # Create a bytes buffer in memory instead of saving to disk
    buf = BytesIO()
    # Save the figure into that buffere as a PNG
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    # Reset buffer pointer to start to allow Streamlit to read
    buf.seek(0)
    return buf.read()

def _close_series(df: pd.DataFrame) -> pd.Series:
    """
    Extracts 'Close' price series from price DataFrame.
    Handles irregular cases such as extra columns, wrong types, etc,
    allows downstream logic to get float series with correct datetime index.
    """
    s = None

    # Pull the 'Close' column
    if isinstance(df, pd.DataFrame):
        if "Close" in df.columns:
            s = df["Close"]

            # Some data sources may return a single column DataFrame, flattens it
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
        # If no 'Close' column, grab first column
        elif df.shape[1] >= 1:
            s = df.iloc[:, 0]

    # if df isn't standard, flatten anything with values and index attributes
    if s is None:
        s = pd.Series(getattr(df, "values", df).ravel(), index=getattr(df, "index", None))

    # Force numeric type, drops bad data
    s = pd.to_numeric(s.squeeze(), errors="coerce")

    # Standardize the index as timezome-free datetime
    try:
        s.index = pd.to_datetime(s.index).tz_localize(None)
    except Exception:
        try:
            s.index = pd.to_datetime(s.index).tz_convert(None)
        except Exception:
            s.index = pd.to_datetime(s.index, errors="coerce")
    s.name = "Close"
    return s

def equity_curve_buy_hold(df_prices: pd.DataFrame, initial_equity: float = 10_000.0) -> pd.Series:
    """
    Buy and Hold benchmark
    This buys as many shares as possible at first available price,
    tracks total account value over time with: shares # price + leftover cash
    Used to compare strategy performance against just holding
    """
    if df_prices is None or df_prices.empty:
        return pd.Series(dtype=float)
    
    # Gets Close series, sorts chronologically
    close = _close_series(df_prices).sort_index()

    # makes initial purchase, buys as many shares as possible 
    start = float(close.iloc[0])
    shares = int(initial_equity // start)
    rem = initial_equity - shares * start

    # Portfolio value = remaining cash + value of held shares over time
    eq = rem + shares * close.astype(float)

    # Returns time series of total equity
    return pd.Series(eq.values, index=close.index, name="buy_hold")

def max_drawdown(eq: pd.Series) -> float:
    """
    Calculates max drawdown
    Worst percentage dropped from a portfolio's peak equity to the lowest point that follows.
    This is expressed as a negative value
    """
    if eq is None or eq.empty:
        return 0.0
    
    # Tracks running max, seen as peak equity over time
    cm = eq.cummax()

    # Calculates drawdowns as % below peak
    dd = (eq / cm) - 1.0

    # Return most worst loss
    return float(dd.min())  # negative (e.g., -0.1234)

def sharpe_ratio(eq: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculates annual Sharpe ratio for a given equity curve.
    This measures how much excess return the strategy earns per unit of volatility.
    Generally, higher is better.
    """
    if eq is None or len(eq) < 2:
        return 0.0
    
    # Converts equity values to periodic returns
    rets = eq.pct_change().dropna()

    # Avoids division by zero if there is no volatility
    if rets.std() == 0:
        return 0.0
    
    # Returns mean divided by standard deviation, per year
    return float((rets.mean() / rets.std()) * (periods_per_year ** 0.5))

def _normalize_trades_df(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes a trade DataFrame to ensure consistent formatting.
    Makes sure all trades have 'Date', 'Action', 'Price', and 'PnL' columns
    and are correctly typed and sorted by date.
    """
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["Date", "Action", "Price", "PnL"])
    
    # Works on a copy to avoid changing original data
    df = trades.copy().reset_index()

    # Some trade logs use 'index' as timestamp, rename to 'Date' column
    if "Date" not in df.columns:
        df = df.rename(columns={"index": "Date"})

    # Adds any missing columns
    for col in ["Action", "Price", "PnL"]:
        if col not in df.columns:
            df[col] = pd.NA
    
    # Ensures proper data types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Action"] = df["Action"].astype(str)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["PnL"]   = pd.to_numeric(df["PnL"], errors="coerce")

    # Sorts chronologically for plotting and reporting
    return df.sort_values("Date").reset_index(drop=True)

def best_trades_desc(trades: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Returns top N best trades by profit from a trade log.
    Only includes completed 'Sell' actions with PnL values.
    """

    # Standardizes trade data
    df = _normalize_trades_df(trades)

    # Focus on completed sells with a valid profit and\or loss
    sells = df[df["Action"].str.lower() == "sell"].dropna(subset=["PnL"])
    if sells.empty:
        return sells[["Date", "Price", "PnL"]]
    
    # Sort PnL descending and return top N
    top = sells.sort_values("PnL", ascending=False).head(n).copy()

    # Rounds numerical column
    top["Price"] = top["Price"].round(2)
    top["PnL"]   = top["PnL"].round(2)

    # Returns columns by best trades
    return top[["Date", "Price", "PnL"]]

def all_trades_table(trades: pd.DataFrame) -> pd.DataFrame:
    """ 
    Returns formatted table of all trades for display or export.
    Ensures standardized columns and rounds values
    """

    # Copy trade data to avoid changing orriginal
    df = _normalize_trades_df(trades).copy()

    # Round numeric columns
    df["Price"] = df["Price"].round(2)
    df["PnL"]   = df["PnL"].round(2)

    # Return trade information in order
    return df[["Date", "Action", "Price", "PnL"]]

"""
Ensure the app has ticker selected with valid input
"""
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"
if "start_date" not in st.session_state:
    st.session_state.start_date = date.today() - timedelta(days=365)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()

# sidebar controls
with st.sidebar:
    st.header("Controls")

    # Ticker and Dates
    with st.expander("Ticker & Dates", expanded=True):

        # Quick tickers
        st.caption("Quick Tickers")

        # Generates columns 
        col_qt1, col_qt2, col_qt3 = st.columns(3)

        # Sets quick ticker to selected ticker state
        if col_qt1.button("AAPL"): st.session_state.selected_ticker = "AAPL"; st.rerun()
        if col_qt2.button("TSLA"): st.session_state.selected_ticker = "TSLA"; st.rerun()
        if col_qt3.button("SPY"):  st.session_state.selected_ticker = "SPY";  st.rerun()

        # Manual ticker input
        ticker = st.text_input("Ticker", value=st.session_state.selected_ticker).upper().strip()
        if ticker:
            st.session_state.selected_ticker = ticker

        # Quick ranges
        st.caption("Quick Ranges")

        # Helper functions to update global date range, and reruns app
        def _apply_range(start_date_val, end_date_val=None):
            end_date_val = end_date_val or st.session_state.get("end_date", date.today())
            st.session_state.start_date = start_date_val
            st.session_state.end_date = end_date_val
            st.session_state["start_date_input"] = start_date_val
            st.session_state["end_date_input"] = end_date_val
            st.rerun()

        # Quick time presets
        qrow1c1, qrow1c2 = st.columns(2)
        if qrow1c1.button("3M"):  _apply_range(date.today() - timedelta(days=90))
        if qrow1c2.button("YTD"): _apply_range(date(date.today().year, 1, 1))
        qrow2c1, qrow2c2 = st.columns(2)
        if qrow2c1.button("1Y"):  _apply_range(date.today() - timedelta(days=365))
        if qrow2c2.button("5Y"):  _apply_range(date.today() - timedelta(days=365 * 5))

        # Manual date inputs
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

        # Keeps session state in sync with user inputs
        if start != st.session_state.start_date:
            st.session_state.start_date = start
        if end != st.session_state.end_date:
            st.session_state.end_date = end

    # Portfolio
    with st.expander("Portfolio", expanded=True):
        starting_cash = st.number_input(
            "Starting cash ($)",
            min_value=1_000, max_value=10_000_000, value=10_000, step=500,
            help="Initial equity for the backtest. Default is $10,000."
        )

    # Strategy Parameters
    """
    Controls for both strategies
    These sliders allow users to change variables for sensitivity
    """
    with st.expander("Strategy Parameters", expanded=False):

        # Looks at short term price direction and trades when prices move away from recent history
        st.caption("Momentum")

        # Number of days to look back for momentum (mom) calculation
        # A longer lookback reduces noise, but also reduces responsiveness
        mom_lookback = st.number_input(
            "Lookback (days)",
            min_value=1, max_value=252, value=20, step=1,
            help="How many past days to compare. Higher = smoother & fewer trades; lower = faster but noisier."
        )

        # Minimum gap between current and reference price before triggering a trade
        # Filters out small, mostly insignificant moves
        mom_min_gap = st.number_input(
            "Min gap vs ref (fraction)",
            min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f",
            help="Smallest relative gap to trigger a trade. Example: 0.05 = 5%."
        )
        st.divider()

        # Trades against short term extremes under the assumption that prices tend to revert back to average
        st.caption("Mean Reversion")

        # Moving average (MA) window defines what 'normal' looks like
        # A longer MA has a smoother baseline with fewer trades, where shorter reacts faster
        mr_ma_window = st.number_input(
            "MA window (days)",
            min_value=2, max_value=252, value=20, step=1,
            help="Days in the moving average. Higher = smoother & fewer trades; lower = faster but choppier."
        )

        # Z score threshold for entering trades
        # A higher value means only entering during more extreme deviations
        mr_z_entry = st.number_input(
            "Z-entry",
            min_value=0.1, max_value=5.0, value=1.0, step=0.1, format="%.1f",
            help="Enter when price is this many std-devs from the mean. Higher = rarer, more extreme setups."
        )

        # Z score threshold for exiting trades
        # A lower exit value closes sooner, a higher one allows longer holds
        mr_z_exit = st.number_input(
            "Z-exit",
            min_value=0.05, max_value=3.0, value=0.25, step=0.05, format="%.2f",
            help="Close when distance shrinks to this level. Lower = exit sooner; higher = ride longer."
        )

    # Risk Controls
    """
    Adds optional safety tools
    Allows users to limit risk
    All defaults have no stops or targets, generally conservative
    """
    with st.expander("Risk Controls", expanded=False):

        # Percentage of cash used per trade
        # i.e., 100% means full capital investment per trade
        # the lower the value can simulate partial allocation
        pos_pct = st.number_input(
            "Position size (% of cash)",
            min_value=1, max_value=100, value=100, step=5,
            help="How much of available cash to invest on each BUY (100 = all-in)."
        )

        # The stop loss threshold allows automatic exits for losing trades
        stop_loss_pct = st.number_input(
            "Stop-loss (%)",
            min_value=0.0, max_value=50.0, value=0.0, step=0.5,
            help="Exit if price falls this % below the entry. Set 0 to disable."
        )

        # The take profit threshold closes positions once a target gain is reached
        take_profit_pct = st.number_input(
            "Take-profit (%)",
            min_value=0.0, max_value=200.0, value=0.0, step=1.0,
            help="Exit if price rises this % above the entry. Set 0 to disable."
        )

    # Frictions
    """
    Models real world trading cost that affects performance
    Default set to zero for clean initial comparisons
    """
    with st.expander("Frictions", expanded=False):

        # Flat commission fee charged per trade
        commission = st.number_input(
            "Commission per trade ($)",
            min_value=0.0, max_value=50.0, value=0.0, step=0.1,
            help="Fixed fee per BUY/SELL."
        )

        # Slippage accounts for difference between expected and actual trade prices
        # Measured in basis points (bp) where 1 bps = 0.01%
        slippage_bps = st.number_input(
            "Slippage (bps)",
            min_value=0, max_value=200, value=0, step=5,
            help="Execution slippage in basis points (10 bps = 0.10%)."
        )



    # Run
    show_ma = st.checkbox("Show moving average line", value=True)
    run_btn = st.button("Run Backtest", type="primary")

# Shared state-backed vars
start = st.session_state.start_date
end = st.session_state.end_date
ticker = st.session_state.selected_ticker
initial_equity = float(starting_cash)

# Tabs: Backtest | Batch Summary
tab_backtest, tab_batch = st.tabs(["Backtest", "Batch Summary"])

with tab_backtest:
    if run_btn:
        try:

            # Grabs historical price data for chosen ticker and range
            with st.spinner("Fetching data..."):
                df = get_data(ticker, start_date=str(start), end_date=str(end))

            # Handles invalid data
            if df is None or df.empty:
                st.warning("No data returned. Try a different date range or ticker.")
            else:
                # These are shared parameters that apply to both momentum and mean reversion
                common_sim_kwargs = dict(
                    initial_equity=initial_equity,
                    position_size_pct=float(pos_pct) / 100.0,
                    stop_loss_pct=(float(stop_loss_pct) / 100.0) if stop_loss_pct > 0 else None,
                    take_profit_pct=(float(take_profit_pct) / 100.0) if take_profit_pct > 0 else None,
                    commission=float(commission),
                    slippage_bps=int(slippage_bps),
                )

                # Buys when prices show sustained gain over a lookback window
                # The 'min_gap' controls how large that move must be to trigger trades
                trades_mom, stats_mom = run_simulation(
                    df.copy(),
                    lambda d: momentum_strategy(d, lookback=int(mom_lookback), min_gap=float(mom_min_gap)),
                    **common_sim_kwargs
                )

                # Parameters here define how far prices must deviate from mean (Z entry)
                # before entering and how close it must be return to mean (Z exit) to exit
                trades_rev, stats_rev = run_simulation(
                    df.copy(),
                    lambda d: mean_reversion_strategy(
                        d,
                        ma_window=int(mr_ma_window),
                        z_entry=float(mr_z_entry),
                        z_exit=float(mr_z_exit),
                    ),
                    **common_sim_kwargs
                )

                # Buy & hold benchmark
                """ 
                Calculates how much profit would have been generated by buying and holding until the end
                """
                close = _close_series(df)
                start_price = float(close.iloc[0])
                end_price = float(close.iloc[-1])

                # Number of shares that can be bought initially
                shares_bh = int(initial_equity // start_price)

                # Remaining cash after buying shares
                rem_cash = initial_equity - shares_bh * start_price

                # Final value = shares # last price + remaining cash
                final_bh = rem_cash + shares_bh * end_price

                # Benchmark profit and return percentages
                bh_profit = final_bh - initial_equity
                bh_return_pct = (final_bh / initial_equity - 1.0) * 100.0

                # top row: Momentum , Mean Reversion , Equity Curves
                """
                Displays three visual panels side by side
                """
                left, middle, right = st.columns(3)

                # Momentum strategy chart
                with left:

                    # Creates a plot for momentum strategy
                    fig, ax = plt.subplots(figsize=(7, 4))

                    # Title includes profit and total trade count
                    title = f"{ticker} — Momentum\nProfit: ${stats_mom['profit']:.2f} | Trades: {stats_mom['trades']}"

                    # Draws buy\sell markers, price line, movering average, and equity overlay
                    plot_trades_matplotlib(df.copy(), trades_mom, title=title, show_ma=show_ma, ax=ax, cumulative_stats=stats_mom)

                    # Render the figure inside streamlit
                    st.pyplot(fig, clear_figure=True)

                    # Allows user to download chart as a PNG
                    st.download_button(
                        label="Download Momentum chart (PNG)",
                        data=fig_to_png_bytes(fig),
                        file_name=f"{ticker}_momentum_{start}_{end}.png",
                        mime="image/png",
                    )

                # Mean Reversion strategy chart
                with middle:

                    # Creates a plot for the mean reversion strategy
                    fig2, ax2 = plt.subplots(figsize=(7, 4))

                    # Title includes profit and total trade count
                    title2 = f"{ticker} — Mean Reversion\nProfit: ${stats_rev['profit']:.2f} | Trades: {stats_rev['trades']}"

                    # Draws buy\sell markers, price line, movering average, and equity overlay
                    plot_trades_matplotlib(df.copy(), trades_rev, title=title2, show_ma=show_ma, ax=ax2, cumulative_stats=stats_rev)

                    # Render the figure inside streamlit
                    st.pyplot(fig2, clear_figure=True)

                    # Allows user to download chart as a PNG
                    st.download_button(
                        label="Download Mean Reversion chart (PNG)",
                        data=fig_to_png_bytes(fig2),
                        file_name=f"{ticker}_mean_reversion_{start}_{end}.png",
                        mime="image/png",
                    )

                # Equity curves
                with right:
                    st.markdown("**Equity Curves - Normalized**")

                    # Compute each equity curve
                    eq_mom = stats_mom.get("equity_curve")
                    eq_rev = stats_rev.get("equity_curve")
                    eq_bh  = equity_curve_buy_hold(df, initial_equity=initial_equity)

                    # Normalizes any equity series so the first point equals 1.0
                    def _norm(s: pd.Series) -> pd.Series:
                        if s is None or len(s) == 0:
                            return s
                        base = s.iloc[0] if s.iloc[0] != 0 else 1.0
                        return s / base

                    # Plot equity curves
                    fig_ec, ax_ec = plt.subplots(figsize=(7, 4))

                    # Plot momentum strategy performance
                    if isinstance(eq_mom, pd.Series) and not eq_mom.empty:
                        ax_ec.plot(eq_mom.index, _norm(eq_mom), label="Momentum")

                    # Plot mean reversion strategy performance
                    if isinstance(eq_rev, pd.Series) and not eq_rev.empty:
                        ax_ec.plot(eq_rev.index, _norm(eq_rev), label="Mean Reversion")
                    
                    # Plot buy and hold benchmark
                    if isinstance(eq_bh, pd.Series) and not eq_bh.empty:
                        ax_ec.plot(eq_bh.index, _norm(eq_bh), label="Buy & Hold")

                    # Plot formatting
                    ax_ec.set_xlabel("Date"); ax_ec.set_ylabel("Equity (normalized)")
                    ax_ec.legend(loc="best"); ax_ec.grid(True, linestyle="--", alpha=0.3)

                    # Render chat in streamlit
                    st.pyplot(fig_ec, clear_figure=True)

                    # risk metrics for single-ticker view
                    # Compute sharpe ratio for both strategies
                    mom_sharpe = sharpe_ratio(eq_mom) if isinstance(eq_mom, pd.Series) else 0.0
                    rev_sharpe = sharpe_ratio(eq_rev) if isinstance(eq_rev, pd.Series) else 0.0

                    # Compute drawdowns by converting fraction to percentage
                    mom_mdd = max_drawdown(eq_mom) * 100.0 if isinstance(eq_mom, pd.Series) else 0.0
                    rev_mdd = max_drawdown(eq_rev) * 100.0 if isinstance(eq_rev, pd.Series) else 0.0

                # quick metrics
                st.markdown("---")
                st.subheader("Metrics")

                # Summary Metrics Display
                def basic_metrics(stats: dict):
                    
                    # Calculates total return as a percentage of inital capital
                    ret_pct = (stats.get("profit", 0.0) / initial_equity) * 100.0

                    # Use streamlit's widget for summary
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Profit ($)", f"{stats.get('profit', 0.0):,.2f}")
                    c2.metric("Trades", f"{stats.get('trades', 0)}")
                    c3.metric("Return (%)", f"{ret_pct:.2f}")

                # Strategy summaries
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**Momentum**")
                    basic_metrics(stats_mom)
                with colB:
                    st.markdown("**Mean Reversion**")
                    basic_metrics(stats_rev)

                # Risk metrics below main summaries
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

                # Buy and hold benchmark comparison
                st.markdown("&nbsp;")
                c1, c2, c3 = st.columns(3)

                # Shows raw profit and percentage return for buy and hold
                c1.metric("Benchmark Profit ($)", f"{bh_profit:,.2f}")
                c2.metric("Benchmark Return (%)", f"{bh_return_pct:.2f}")

                # Visual check
                beats = "Yes" if (stats_mom["profit"] > bh_profit) or (stats_rev["profit"] > bh_profit) else "No"
                c3.metric("Beats Buy & Hold?", beats)

                # comparison row
                st.markdown("&nbsp;")
                comp1, comp2 = st.columns(2)

                # Compute each strategy's returns as % of starting equity
                mom_ret = (stats_mom.get("profit", 0.0) / initial_equity) * 100.0
                rev_ret = (stats_rev.get("profit", 0.0) / initial_equity) * 100.0

                # Identify which strategy earned higher profit
                better_profit = "Momentum" if stats_mom.get("profit", 0.0) >= stats_rev.get("profit", 0.0) else "Mean Reversion"
                profit_delta = stats_mom.get("profit", 0.0) - stats_rev.get("profit", 0.0)
                comp1.metric("Better Profit", better_profit, delta=f"${profit_delta:,.2f}")

                # Identify which strategy had the higher return %
                better_return = "Momentum" if mom_ret >= rev_ret else "Mean Reversion"
                ret_delta = mom_ret - rev_ret
                comp2.metric("Higher Return (%)", better_return, delta=f"{ret_delta:.2f} pp")

                # Trade tables
                st.markdown("---")
                st.subheader("Top Trades (best 10 by PnL)")

                # Two column layout for momentum and mean reversion summaries
                bt_col1, bt_col2 = st.columns(2)
                with bt_col1:
                    st.markdown("**Momentum — Best 10**")
                    df_best_mom = best_trades_desc(trades_mom, n=10)
                    if df_best_mom.empty:
                        st.info("No sell trades to rank yet for Momentum.")
                    else:
                        st.dataframe(df_best_mom, use_container_width=True)

                with bt_col2:
                    st.markdown("**Mean Reversion — Best 10**")
                    df_best_rev = best_trades_desc(trades_rev, n=10)
                    if df_best_rev.empty:
                        st.info("No sell trades to rank yet for Mean Reversion.")
                    else:
                        st.dataframe(df_best_rev, use_container_width=True)

                st.markdown("---")
                st.subheader("All Trades")

                # Two column layout for momentum and mean reversion summaries
                all_col1, all_col2 = st.columns(2)
                with all_col1:
                    st.markdown("**Momentum — All trades**")
                    df_all_mom = all_trades_table(trades_mom)
                    if df_all_mom.empty:
                        st.info("No trades for Momentum in this period.")
                    else:
                        st.dataframe(df_all_mom, use_container_width=True, height=320)

                with all_col2:
                    st.markdown("**Mean Reversion — All trades**")
                    df_all_rev = all_trades_table(trades_rev)
                    if df_all_rev.empty:
                        st.info("No trades for Mean Reversion in this period.")
                    else:
                        st.dataframe(df_all_rev, use_container_width=True, height=320)

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)
    else:
        st.info("Set your options in the sidebar and click **Run Backtest**.")

with tab_batch:
    st.subheader("Batch Summary (Sharpe & Max DD)")
    st.caption("Enter tickers below and click **Run Batch Summary** to compare strategies.")
    batch_tickers = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, GOOG", key="batch_tickers")
    run_batch = st.button("Run Batch Summary", key="run_batch_btn")

    if run_batch:
        def run_one(tk: str):

            # Normalize and validate the ticker
            tk = tk.strip().upper()
            if not tk:
                return None
            
            # Fetch price data for this ticker
            dfb = get_data(tk, start_date=str(start), end_date=str(end))
            if dfb is None or dfb.empty:

                # Handles missing or invalid data
                return {
                    "Ticker": tk, "Mom Return (%)": "—", "MR Return (%)": "—",
                    "Mom Sharpe": "—", "MR Sharpe": "—",
                    "Mom Max DD (%)": "—", "MR Max DD (%)": "—",
                    "Best": "No data"
                }

            # Shared simulation params
            common_sim_kwargs = dict(
                initial_equity=initial_equity,
                position_size_pct=float(pos_pct) / 100.0,
                stop_loss_pct=(float(stop_loss_pct) / 100.0) if stop_loss_pct > 0 else None,
                take_profit_pct=(float(take_profit_pct) / 100.0) if take_profit_pct > 0 else None,
                commission=float(commission),
                slippage_bps=int(slippage_bps),
            )

            # Each strategy passed as lambda so params can be adjusted without modifying strategy function itself
            t_m, s_m = run_simulation(

                # dfb.copy() ensures that each run works with an isolated DataFrame without changing original
                dfb.copy(),
                lambda d: momentum_strategy(d, lookback=int(mom_lookback), min_gap=float(mom_min_gap)),
                **common_sim_kwargs
            )
            t_r, s_r = run_simulation(
                dfb.copy(),
                lambda d: mean_reversion_strategy(
                    d, ma_window=int(mr_ma_window), z_entry=float(mr_z_entry), z_exit=float(mr_z_exit)
                ),
                **common_sim_kwargs
            )

            # Converts profits into %, returns relative to starting equity
            mom_ret = (s_m.get("profit", 0.0) / initial_equity) * 100.0
            mr_ret  = (s_r.get("profit", 0.0) / initial_equity) * 100.0

            # Pull equity curve data for risk metrics
            mom_eq  = s_m.get("equity_curve"); mr_eq = s_r.get("equity_curve")

            # Calculates sharpe ration and max drawdown
            mom_sharpe = sharpe_ratio(mom_eq)
            mr_sharpe  = sharpe_ratio(mr_eq)
            mom_mdd    = max_drawdown(mom_eq) * 100.0
            mr_mdd     = max_drawdown(mr_eq) * 100.0

            # Determine which strategy performed better by total return
            best_name = "Momentum" if mom_ret >= mr_ret else "MeanRev"
            best_val  = mom_ret if mom_ret >= mr_ret else mr_ret

            # Package results for batch summary table
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

        # Parse user input into list of tickers
        tickers_list = [t for t in (batch_tickers.split(",") if batch_tickers else [])]

        # Run each ticker through helper function, collect results, skip invalid or missing data
        rows = [r for tk in tickers_list if (r := run_one(tk)) is not None]

        if rows:
            
            # Convert collected summaries into DataFrame
            df_out = pd.DataFrame(rows)

            # Render batch results table in Streamlit
            st.dataframe(df_out, use_container_width=True)

            # Add CSV download option 
            st.download_button(
                "Download batch summary (CSV)",
                df_out.to_csv(index=False).encode("utf-8"),
                file_name=f"batch_summary_{start}_{end}.csv",
                mime="text/csv"
            )
        else:

            # Handles emptpy input
            st.info("No tickers to run. Add symbols like: `AAPL, MSFT, GOOG`")

