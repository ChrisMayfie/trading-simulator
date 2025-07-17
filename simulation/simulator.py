import pandas as pd
import numpy as np

def run_simulation(df, strategy_fn, strategy_name="Strategy", show_ma=False):

    """
    
    Rum a simulated trading session using a given strategy.

    Parameters:
        df (pd.DataFrame): Historical price data
        strategy_fn (function): Function that returns a 'buy' or 'sell' signal
        strategy_name (str): Label for the strategy
        show_ma (bool): Option to toggle moving average plotting (UNUSED)

    Returns:
        df_trades (pd.DataFrame): Log of trades made during simulation
        stats (dict): Summary statistics of performance

    """

    #Initial capital setup
    cash = 10000            #start with $10,000 in cash
    position = 0            #number of shares held
    entry_price = 0         #price at which last buy was made
    trades = []             #list to record all trade actions
    returns = []            #store ind. trade returns for metrics
    cumulative_pnl = []     #running total of prof/loss per trade
    running_total = 0       #tracks total PnL

    #Start sim after 20 days, allows for MA or other lookback indicators
    for i in range(20, len(df)):
        window = df.iloc[:i]            #slice data up to current point
        signal = strategy_fn(window)    #get signal from strategy logic
        price = float(df["Close"].iloc[i])

        # Execute Buy
        if signal == "buy" and cash > price:
            position = cash // price        #buy as many shares as possible
            cash -= position * price        #deduct cost from cash
            entry_price = price             #store entry price for later PnL
            trades.append((df.index[i], "BUY", price))

        #Execute Sell
        elif signal == "sell" and position > 0:
            cash += position * price                            #sell entire position
            pnl = (price - entry_price) * position              #gets profit and loss (PnL)
            running_total += pnl                                #update PnL
            cumulative_pnl.append((df.index[i], running_total))
            trades.append((df.index[i], "SELL", price, pnl))
            returns.append((price - entry_price) / entry_price)
            position = 0                                        #reset position

    # Calculate final portfolio value based on last price
    final_price = float(df["Close"].iloc[-1])
    final_value = cash + position * final_price
    profit = final_value - 10000            #Net profit from original capital

    # Reporting Summary
    print(f"\n{strategy_name} Results:")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Profit: ${profit:.2f}")
    print(f"Total Trades: {len(trades)}")

    # Store summary stats
    stats = {
        "profit": profit,
        "trades": len(trades),
        "cumulative_pnl": cumulative_pnl
    }

    # If trades made, compute performance metrics
    if returns:
        total_return = (final_value - 10000) / 10000
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) != 0 else 0
        stats.update({
            "total_return": total_return,
            "win_rate": win_rate,
            "sharpe": sharpe
        })
        print(f"Total Return: {total_return:.2%}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Sharpe Ratio (approx): {sharpe:.2f}")

    # Print individual trade log
    for t in trades:
        if len(t) == 4:
            print(f"{t[0].date()} - {t[1]} at ${t[2]:.2f} | P&L: ${t[3]:.2f}")
        else:
            print(f"{t[0].date()} - {t[1]} at ${t[2]:.2f}")

    # Format trade list into DataFrame for later use
    df_trades = pd.DataFrame(trades)
    if df_trades.shape[1] == 4:
        df_trades.columns = ["Date", "Action", "Price", "PnL"]
    else:
        df_trades.columns = ["Date", "Action", "Price"]
    df_trades.set_index("Date", inplace=True)
    return df_trades, stats