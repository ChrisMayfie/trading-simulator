def momentum_strategy(df, window_size=5):

    """
    A simple momentum strategy:
    - Buy if the price has increased over the past window.
    - Sell if it has decreased.

    Parameters:
        df (pd.DataFrame): Historical stock data.
        window_size (int): Lookback window to measure price momentum.

    Returns:
        str or None: 'buy', 'sell', or None (no action).
    """

    closes = df["Close"] # Extract closing prices

    # Make sure there's enough data to compare across the window
    if len(closes) < window_size + 1:
        return None

    # Get the current and lookback prices
    current = float(closes.iloc[-1])
    previous = float(closes.iloc[-window_size])

    # If the price has risen over the window, assume momentum is upward
    if current > previous:
        return "buy"
    
    # If the price has fallen, assume momentum is downward
    elif current < previous:
        return "sell"
    
    # No signal if the price hasn't moved significantly
    return None