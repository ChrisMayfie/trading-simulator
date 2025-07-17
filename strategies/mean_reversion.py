def mean_reversion_strategy(df, ma_window=10, threshold=0.02):

    """
    A basic mean reversion strategy:
    - Buy when the current price is significantly below its moving average.
    - Sell when it's significantly above.

    Parameters:
        df (pd.DataFrame): Historical stock data.
        ma_window (int): Window size for the moving average.
        threshold (float): % deviation from the mean that triggers buy/sell.

    Returns:
        str or None: 'buy', 'sell', or None (no action).
    """
        
    closes = df["Close"] # Get the series of closing prices

    # Ensure there’s enough data to compute the moving average
    if len(closes) < ma_window:
        return None
    
    # Calculate the moving average of the last N closing prices
    ma = float(closes[-ma_window:].mean())
    current_price = float(closes.iloc[-1]) # Get the most recent closing price

    # Buy signal: if price is significantly below the moving average
    if current_price < ma * (1 - threshold):
        return "buy"
    
    # Sell signal: if price is significantly above the moving average
    elif current_price > ma * (1 + threshold):
        return "sell"
    
    # Otherwise, no action
    return None