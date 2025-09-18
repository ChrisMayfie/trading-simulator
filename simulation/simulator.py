import pandas as pd
import numpy as np

def run_simulation(df, strategy_fn, strategy_name="Strategy", show_ma=False):
    """
    Very simple backtest loop:
    - Start with $10,000 cash
    - Step forward one bar at a time
    - Ask the strategy for 'buy' / 'sell' / None
    - Track trades and final profit
    """

    if df is None or df.empty:
        # return empty trade log and zeroed stats
        empty = pd.DataFrame(columns=["Date", "Action", "Price", "PnL"]).set_index("Date")
        return empty, {"profit": 0.0, "trades": 0}

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    cash = 10_000.0
    shares = 0
    entry_price = None
    trades = []

    for i in range(len(df)):
        # use all data up to and including current bar
        window = df.iloc[: i + 1]
        price = float(window["Close"].iloc[-1])

        # simple signal: 'buy', 'sell', or None
        signal = strategy_fn(window)

        # buy only if flat
        if signal == "buy" and shares == 0:
            # whole shares only
            qty = int(cash // price)
            if qty > 0:
                cash -= qty * price
                shares = qty
                entry_price = price
                trades.append((window.index[-1], "buy", price))

        # sell only if long
        elif signal == "sell" and shares > 0:
            cash += shares * price
            pnl = (price - (entry_price if entry_price is not None else price)) * shares
            trades.append((window.index[-1], "sell", price, pnl))
            shares = 0
            entry_price = None

    # close any open position at the very end
    if shares > 0:
        last_price = float(df["Close"].iloc[-1])
        cash += shares * last_price
        pnl = (last_price - (entry_price if entry_price is not None else last_price)) * shares
        trades.append((df.index[-1], "sell", last_price, pnl))
        shares = 0
        entry_price = None

    final_equity = cash
    profit = final_equity - 10_000.0
    stats = {"profit": float(profit), "trades": int(sum(1 for t in trades if t[1] == "sell"))}

    # format trade log
    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        # some rows are buys (3 cols), some are sells (4 cols)
        if df_trades.shape[1] == 4:
            df_trades.columns = ["Date", "Action", "Price", "PnL"]
        else:
            df_trades.columns = ["Date", "Action", "Price"]
        df_trades.set_index("Date", inplace=True)
    else:
        df_trades = pd.DataFrame(columns=["Date", "Action", "Price", "PnL"]).set_index("Date")

    return df_trades, stats
