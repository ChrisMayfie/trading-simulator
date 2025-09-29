# strategies/momentum.py
from __future__ import annotations
import pandas as pd

def _close_series(df: pd.DataFrame) -> pd.Series:
    # Always return a 1-D float Series named "Close"
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

def momentum_strategy(df: pd.DataFrame, lookback: int = 20, min_gap: float = 0.0) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=int)
    close = _close_series(df)
    ref = close.shift(lookback)
    pct_change_vs_ref = (close - ref) / ref

    sig = pd.Series(0, index=df.index, dtype=int)
    if min_gap <= 0:
        sig.loc[pct_change_vs_ref > 0] = 1
        sig.loc[pct_change_vs_ref < 0] = -1
    else:
        sig.loc[pct_change_vs_ref >= min_gap] = 1
        sig.loc[pct_change_vs_ref <= -min_gap] = -1
    return sig
