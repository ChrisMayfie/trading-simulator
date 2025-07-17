def plot_trades_matplotlib(df, trades, title="Trading Strategy", show_ma=False, ax=None, cumulative_stats=None):

    """
    Plot trade signals (buy/sell) and cumulative P&L on a price chart using matplotlib.

    Parameters:
        df (pd.DataFrame): Price data including 'Close' column.
        trades (pd.DataFrame): DataFrame with BUY/SELL actions and prices.
        title (str): Title for the chart.
        show_ma (bool): Whether to overlay a 10-day moving average.
        ax (matplotlib.axes): Optional matplotlib axes object.
        cumulative_stats (dict): Optional stats dict with cumulative P&L for overlay.
    """

    if ax is None:
        ax = plt.gca() # Use current axes if none provided

    # Plot closing price
    ax.plot(df.index, df["Close"], label="Close Price", color="blue", linewidth=1.5, zorder=1)

    # Optionally overlay a moving average
    if show_ma:
        df["MA10"] = df["Close"].rolling(window=10).mean()
        ax.plot(df.index, df["MA10"], label="10-Day MA", color="orange", linewidth=2, zorder=0)

    # Flags to avoid duplicate legend labels
    buy_shown = sell_shown = False
    entry_points = [] # Store open trades (buy) to draw dashed lines to sells

    for row in trades.itertuples():
        date, action, price = row.Index, row.Action, row.Price
        label = None

        # Handle buy markers and track for sell connection
        if action == "BUY":
            if not buy_shown:
                label = "Buy"
                buy_shown = True
            entry_points.append((date, price))
        
        # Handle sell markers and draw dashed P&L lines
        elif action == "SELL":
            if not sell_shown:
                label = "Sell"
                sell_shown = True
            if entry_points:
                entry_date, entry_price = entry_points.pop(0)
                line_color = "green" if price > entry_price else "red"
                ax.plot([entry_date, date], [entry_price, price], linestyle="--", color=line_color, alpha=0.6, linewidth=1.2)

        # Plot individual trade markers
        color = "green" if action == "BUY" else "red"
        marker = "^" if action == "BUY" else "v"
        ax.scatter(date, price, color=color, marker=marker, label=label, zorder=2)

    # Optional: Overlay cumulative P&L on a secondary y-axis
    if cumulative_stats and "cumulative_pnl" in cumulative_stats:
        pnl_df = pd.DataFrame(cumulative_stats["cumulative_pnl"], columns=["Date", "PnL"])
        pnl_df.set_index("Date", inplace=True)
        ax2 = ax.twinx()
        ax2.plot(pnl_df.index, pnl_df["PnL"], color="gray", linestyle="--", label="Cumulative P&L", linewidth=1.2)
        ax2.tick_params(axis="y", labelsize=8)
        ax2.legend(loc="upper right", fontsize=7)

    # Final chart formatting
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Price", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True)

def show_trade_table(ax, df1, df2, title="Trade Table"):

    """
    Render a simple trade summary table in a matplotlib subplot.

    Parameters:
        ax (matplotlib.axes): Axes to render the table into.
        df1 (DataFrame): Trades from strategy 1 (momentum).
        df2 (DataFrame): Trades from strategy 2 (mean reversion).
        title (str): Title for the table subplot.
    """

    ax.axis("off") # Hide axes

    # Tag each strategy's trades
    df1 = df1.copy().assign(Strategy="Momentum")
    df2 = df2.copy().assign(Strategy="MeanRev")
    df_combined = pd.concat([df1, df2]).reset_index()

    # Handle missing PnL column gracefully
    if "PnL" not in df_combined:
        df_combined["PnL"] = "-"
    df_combined["Price"] = df_combined["Price"].round(2)
    df_combined["PnL"] = pd.to_numeric(df_combined["PnL"], errors="coerce").round(2)

    # Clean up and format date
    df_combined["Date"] = pd.to_datetime(df_combined["Date"]).dt.strftime("%Y-%m-%d")
    df_display = df_combined[["Date", "Strategy", "Action", "Price", "PnL"]].head(10)

    # Format table data
    table_data = [df_display.columns.tolist()] + df_display.values.tolist()
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    ax.set_title(title, fontsize=10)

    # Highlight best/worst trades using color
    valid_pnl = df_display["PnL"].apply(lambda x: isinstance(x, (int, float)))
    if valid_pnl.any():
        numeric_pnls = df_display.loc[valid_pnl, "PnL"]
        if not numeric_pnls.empty:
            max_idx = numeric_pnls.idxmax()
            min_idx = numeric_pnls.idxmin()
            for col_idx in range(len(df_display.columns)):
                table[(max_idx + 1, col_idx)].set_facecolor("#d4f7d4")  # green
                table[(min_idx + 1, col_idx)].set_facecolor("#f7d4d4")  # red


import matplotlib.pyplot as plt
import pandas as pd