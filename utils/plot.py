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

import pandas as pd
import matplotlib.pyplot as plt

def show_trade_table(ax, trades_momentum: pd.DataFrame, trades_reversion: pd.DataFrame, title: str = "Trades"):
    """
    Render a compact comparison table on a given Matplotlib axis.
    Robust to object dtypes in Price/PnL/Qty and empty inputs.
    """
    # Defensive copies
    tm = trades_momentum.copy() if trades_momentum is not None else pd.DataFrame()
    tr = trades_reversion.copy() if trades_reversion is not None else pd.DataFrame()

    # Ensure columns exist
    for df in (tm, tr):
        for c in ["Action", "Price", "Qty", "PnL"]:
            if c not in df.columns:
                df[c] = pd.Series([pd.NA] * len(df), index=df.index)

    # Coerce numeric columns (avoids .round on object dtype)
    for df in (tm, tr):
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df["Qty"]   = pd.to_numeric(df["Qty"],   errors="coerce")
        df["PnL"]   = pd.to_numeric(df["PnL"],   errors="coerce")

    # Add labels for concatenation
    tm["_strategy"] = "Momentum"
    tr["_strategy"] = "Mean Reversion"

    # Combine and format
    cols = ["_strategy", "Action", "Price", "Qty", "PnL"]
    df_combined = pd.concat([tm[cols], tr[cols]], axis=0, ignore_index=True)

    # Clean NaNs and format numbers
    if not df_combined.empty:
        # Only round after we coerce to numeric
        df_combined["Price"] = df_combined["Price"].round(2)
        df_combined["PnL"]   = df_combined["PnL"].round(2)
        df_combined["Qty"]   = df_combined["Qty"].astype("Int64")  # pretty ints with NaN support

        # Optional: sort with sells last or by date if you include index
        # df_combined.sort_values(["_strategy", "Action"], inplace=True)

    # Draw table
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)

    if df_combined.empty:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center", fontsize=11)
        return

    # Render a matplotlib table
    table = ax.table(
        cellText=df_combined.values,
        colLabels=["Strategy", "Action", "Price", "Qty", "PnL"],
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)