import matplotlib.pyplot as plt
import pandas as pd

from data.fetch_data import get_data
from simulation.simulator import run_simulation
from strategies.momentum import momentum_strategy
from strategies.mean_reversion import mean_reversion_strategy
from utils.plot import plot_trades_matplotlib, show_trade_table


def main():
    """
    Run momentum and mean-reversion backtests for a set of tickers and
    visualize results: Momentum chart | Mean Reversion chart | Trade table.
    """

    # --- configurable params for CLI run (kept simple / aligned with simulator defaults)
    tickers = ["AAPL", "MSFT", "GOOG"]
    initial_equity = 10_000.0
    position_size_pct = 1.0      # 1.0 = 100% of cash per entry
    stop_loss_pct = None         # e.g. 0.05 for 5% stop; None disables
    take_profit_pct = None       # e.g. 0.10 for 10% TP; None disables
    commission = 0.0             # $ per trade
    slippage_bps = 0             # 10 = 0.10%

    n = len(tickers)
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(18, 4 * n))

    # normalize axes indexing even if n == 1
    if n == 1:
        axes = axes.reshape(1, 3)

    for row, ticker in enumerate(tickers):
        print(f"\n=== {ticker} ===")

        # ---- fetch data (default range inside get_data)
        df = get_data(ticker)
        if df is None or df.empty:
            print(f"[WARN] No data for {ticker}; skipping.")
            # Clear that row with a message
            for col in range(3):
                ax = axes[row, col]
                ax.axis("off")
                ax.text(0.5, 0.5, f"No data for {ticker}", ha="center", va="center")
            continue

        # ---- run both strategies
        trades_momentum, stats_momentum = run_simulation(
            df.copy(),
            lambda d: momentum_strategy(d, lookback=20, min_gap=0.0),
            initial_equity=initial_equity,
            position_size_pct=position_size_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            commission=commission,
            slippage_bps=slippage_bps,
        )

        trades_reversion, stats_reversion = run_simulation(
            df.copy(),
            lambda d: mean_reversion_strategy(d, ma_window=20, z_entry=1.0, z_exit=0.25),
            initial_equity=initial_equity,
            position_size_pct=position_size_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            commission=commission,
            slippage_bps=slippage_bps,
        )

        # ---- Momentum chart
        ax_mom = axes[row, 0]
        title_mom = f"{ticker} — Momentum\nProfit: ${stats_momentum['profit']:.2f}, Trades: {stats_momentum['trades']}"
        plot_trades_matplotlib(
            df.copy(), trades_momentum,
            title=title_mom, show_ma=True, ax=ax_mom, cumulative_stats=stats_momentum
        )

        # ---- Mean Reversion chart
        ax_rev = axes[row, 1]
        title_rev = f"{ticker} — Mean Reversion\nProfit: ${stats_reversion['profit']:.2f}, Trades: {stats_reversion['trades']}"
        plot_trades_matplotlib(
            df.copy(), trades_reversion,
            title=title_rev, show_ma=True, ax=ax_rev, cumulative_stats=stats_reversion
        )

        # ---- Trade comparison table
        ax_tbl = axes[row, 2]
        show_trade_table(ax_tbl, trades_momentum, trades_reversion, title=f"{ticker} Trades")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
