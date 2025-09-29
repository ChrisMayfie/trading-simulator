# strategies/mean_reversion.py
from __future__ import annotations
import pandas as pd

def _close_series(df: pd.DataFrame) -> pd.Series:
    s = None
    if isinstance(df, pd.DataFrame):
        if "Close" in df.columns:
            s = df["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
        elif df.shape[1] >= 1:
            s = df.iloc[:, 0]
    if s is None:
        s = pd.Series(getattr(df, "values", df).ravel(), index=getattr(df, "index", None))
    s = pd.to_numeric(s.squeeze(), errors="coerce")
    s.name = "Close"
    return s

def mean_reversion_strategy(df: pd.DataFrame, ma_window: int = 20, z_entry: float = 1.0, z_exit: float = 0.25) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=int)
    close = _close_series(df)

    sma = close.rolling(ma_window, min_periods=ma_window//2).mean()
    std = close.rolling(ma_window, min_periods=ma_window//2).std(ddof=0)
    z = (close - sma) / std.replace(0, pd.NA)

    sig = pd.Series(0, index=df.index, dtype=int)
    sig.loc[z <= -z_entry] = 1
    sig.loc[z >=  z_entry] = -1
    sig.loc[z.abs() < z_exit] = 0
    return sig
