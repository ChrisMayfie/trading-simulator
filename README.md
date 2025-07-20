# Trading Simulator: Momentum and Mean Reversion Strategies

This project is a Python-based backtesting simulator that runs and compares two classic algorithmic trading strategies: momentum and mean reversion. It uses real historical stock data (AAPL, MSFT, GOOG) and visualizes both performance metrics and individual trades.

## Features

- Backtest two strategies across multiple stocks
- Real equity data from Yahoo Finance (`yfinance`)
- Cumulative profit & loss tracking
- Clear matplotlib visualizations with annotated trades
- Side-by-side trade table comparison
- Modular structure for strategy experimentation

## Strategies

### Momentum
Buys if the price has increased over a recent window. Sells if it has declined.

### Mean Reversion
Buys if the price drops significantly below its moving average. Sells if it rises above.

## Requirements

- Python 3.10+
- pandas
- matplotlib
- yfinance

## Adding Tickers in main.py

Original:

tickers = ["AAPL", "MSFT", "GOOG"] # Stocks to simulate

Create a grid of subplots: 3 rows (one per ticker), 3 columns (momentum, reversion, table)
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))

Modified: 

tickers = ["AAPL", "MSFT", "GOOG", "TSLA"] # Stocks to simulate

Create a grid of subplots: 3 rows (one per ticker), 3 columns (momentum, reversion, table)
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 12))

## Installation & Usage

```bash
# Clone the repo
git clone https://github.com/your-username/trading-simulator.git
cd trading-simulator

# (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas matplotlib yfinance

# Run the simulator
python main.py
