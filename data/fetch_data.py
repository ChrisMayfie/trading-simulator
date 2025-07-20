import yfinance as yf   #import yahoo finance for historical stock prices
import pandas as pd     #handling data structures and analysis

def get_data(ticker: str, start_date="2022-01-01", end_date="2023-01-01") -> pd.DataFrame:

    """
    This fetches historical market data for specified stock ticker.

    Parameters:
        ticker (str): Stock symbol (AAPL - 'Apple' , GOOG - 'Google')
        start_date (str): Start date for data (YYYY-MM-DD)
        end_date (str): end date for data (YYYY-MM-DD)

    Returns:
        pd.DataFrame: Cleaned dataframe containing Open, High, Low, Close, & Volume Columns
    """

    #downloads from yf
    df = yf.download(ticker, start=start_date, end=end_date)

    #keep only essential columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    #drops rows with missing values
    df.dropna(inplace=True)

    #returns cleaned dataframe
    return df
