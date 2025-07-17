import matplotlib.pyplot as plt
import pandas as pd
from data.fetch_data import get_data
from simulation.simulator import run_simulation
from strategies.momentum import momentum_strategy
from strategies.mean_reversion import mean_reversion_strategy
from utils.plot import plot_trades_matplotlib, show_trade_table


def main():

    """
    Main function to execute simulations and visualizations for multiple tickers
    using both Momentum and Mean Reversion strategies.
    """

    tickers = ["AAPL", "MSFT", "GOOG"] # Stocks to simulate

    # Create a grid of subplots: 3 rows (one per ticker), 3 columns (momentum, reversion, table)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))

    for row, ticker in enumerate(tickers):
        print(f"\n=== {ticker} ===")

        # Load data
        df = get_data(ticker)

        # Run momentum strat
        print("Running Momentum Strategy...")
        trades_momentum, stats_momentum = run_simulation(df.copy(), momentum_strategy, strategy_name="Momentum")

        # Export trades for review
        trades_momentum.to_csv(f"{ticker}_momentum_trades.csv")

        # Plot price chart with trade markers and P&L line
        ax_momentum = axes[row, 0]
        title_momentum = f"{ticker} - Momentum\nProfit: ${stats_momentum['profit']:.2f}, Trades: {stats_momentum['trades']}"
        plot_trades_matplotlib(df.copy(), trades_momentum, title=title_momentum, show_ma=True, ax=ax_momentum, cumulative_stats=stats_momentum)

        # Run mean reversion strat
        print("\nRunning Mean Reversion Strategy...")
        trades_reversion, stats_reversion = run_simulation(df.copy(), mean_reversion_strategy, strategy_name="Mean Reversion")

        # Export trades for review
        trades_reversion.to_csv(f"{ticker}_mean_reversion_trades.csv")

        # Plot reversion strategy results
        ax_reversion = axes[row, 1]
        title_reversion = f"{ticker} - Mean Reversion\nProfit: ${stats_reversion['profit']:.2f}, Trades: {stats_reversion['trades']}"
        plot_trades_matplotlib(df.copy(), trades_reversion, title=title_reversion, show_ma=True, ax=ax_reversion, cumulative_stats=stats_reversion)

        # Display trade comparison table
        ax_table = axes[row, 2]
        show_trade_table(ax_table, trades_momentum, trades_reversion, title=f"{ticker} Trades")

    # Adjust layout and display plots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()