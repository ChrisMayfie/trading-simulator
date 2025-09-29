# simulation/simulator.py

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Callable, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------
# Helpers to normalize price series and trades
# ---------------------------------------------------------------------

def _close_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a clean 1-D float Series of Close prices with a datetime index.
    """
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
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = "Close"
    return s


def _ensure_trade_columns(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure trades has Date index + Action, Price, Qty, PnL columns.
    If Qty/PnL are missing, compute them under a full-position approach:
      - BUY: invest all cash
      - SELL: liquidate the position
    """
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["Action", "Price", "Qty", "PnL"])

    t = trades.copy()

    # normalize index to datetime if possible
    if not isinstance(t.index, pd.DatetimeIndex):
        try:
            t.index = pd.to_datetime(t.index)
        except Exception:
            pass

    # map common alternative column names
    if "Action" not in t.columns:
        for alt in ("action", "Signal", "signal", "Side", "side"):
            if alt in t.columns:
                t["Action"] = t[alt]
                break
    if "Price" not in t.columns:
        for alt in ("price", "FillPrice", "fill_price"):
            if alt in t.columns:
                t["Price"] = t[alt]
                break

    # create missing numeric columns
    for c in ("Qty", "PnL"):
        if c not in t.columns:
            t[c] = np.nan

    # compute Qty/PnL if missing (whole-position assumption)
    cash = 10_000.0
    qty = 0
    last_buy_px = None

    t = t.sort_index()
    for i, row in t.iterrows():
        action = str(row.get("Action", "")).lower()
        px = row.get("Price", np.nan)
        if pd.isna(px):
            continue
        px = float(px)

        if action == "buy" and qty == 0:
            qty = int(cash // px)
            if qty > 0:
                cash -= qty * px
                t.at[i, "Qty"] = qty
            last_buy_px = px

        elif action == "sell" and qty > 0:
            cash += qty * px
            pnl = (px - (last_buy_px if last_buy_px is not None else px)) * qty
            t.at[i, "Qty"] = qty
            t.at[i, "PnL"] = pnl
            qty = 0
            last_buy_px = None

    return t


def compute_equity_curve(
    df_prices: pd.DataFrame,
    trades: pd.DataFrame,
    initial_equity: float = 10_000.0
) -> pd.Series:
    """
    Replay trades over the Close price series and return an equity curve Series
    (cash + market value of open position) indexed by date.
    Full-position sizing: invest all cash on BUY; liquidate on SELL.
    """
    close = _close_series(df_prices)
    tr = _ensure_trade_columns(trades).sort_index()

    cash = float(initial_equity)
    shares = 0
    eq = []

    # iterate trades alongside price dates
    t_iter = iter(tr.itertuples(index=True, name="T"))
    current = next(t_iter, None)

    for dt, px in close.items():
        # apply any trades at this timestamp
        while current is not None and current.Index == dt:
            action = str(getattr(current, "Action", "")).lower()
            price = float(getattr(current, "Price", float(px)))
            if action == "buy" and shares == 0:
                qty = int(cash // price)
                if qty > 0:
                    cash -= qty * price
                    shares = qty
            elif action == "sell" and shares > 0:
                cash += shares * price
                shares = 0
            current = next(t_iter, None)

        eq.append(cash + shares * float(px))

    return pd.Series(eq, index=close.index, name="equity")


# ---------------------------------------------------------------------
# Strategy output normalization
# ---------------------------------------------------------------------

def _normalize_strategy_output(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame], Any]
) -> pd.Series:
    """
    Call the strategy and normalize its output to a Series of string actions:
    values in {'buy', 'sell', 'hold'} aligned to df.index.

    Accepted strategy outputs:
      - pd.Series of {1, -1, 0} or {'buy','sell','hold'}
      - pd.DataFrame with 'signal' or 'Action' column (numeric or text)
      - tuple(..., series_or_df) -> we pick the last element if it looks like signals
    """
    out = strategy_fn(df)

    # if tuple, try to use the last Series/DataFrame-looking element
    if isinstance(out, tuple) and len(out) > 0:
        for cand in reversed(out):
            if isinstance(cand, (pd.Series, pd.DataFrame)):
                out = cand
                break

    # DataFrame: try a known column
    if isinstance(out, pd.DataFrame):
        if "signal" in out.columns:
            sig = out["signal"]
        elif "Action" in out.columns:
            sig = out["Action"]
        else:
            sig = out.iloc[:, 0]
    elif isinstance(out, pd.Series):
        sig = out
    else:
        sig = pd.Series(index=df.index, data=0)

    sig = sig.reindex(df.index)

    # Coerce to text actions
    if pd.api.types.is_numeric_dtype(sig):
        m = sig.fillna(0).astype(int)
        return m.map({1: "buy", -1: "sell"}).fillna("hold")

    s = sig.astype(str).str.lower().str.strip()
    return s.where(s.isin(["buy", "sell", "hold"]), "hold")


# ---------------------------------------------------------------------
# Simulation based on normalized signals
# ---------------------------------------------------------------------

def _simulate_from_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_equity: float = 10_000.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simple long-only simulator:
      - On 'buy' when flat: invest all cash at Close
      - On 'sell' when long: liquidate at Close
      - Ignore repeated buys while long or sells while flat
    Returns a (trades_df, stats_dict).
    """
    px = _close_series(df)
    signals = signals.reindex(px.index).fillna("hold")

    cash = float(initial_equity)
    shares = 0
    last_buy_px = None

    records = []

    for dt, action in signals.items():
        a = str(action).lower()
        price = float(px.loc[dt])

        if a == "buy" and shares == 0:
            qty = int(cash // price)
            if qty > 0:
                cash -= qty * price
                shares = qty
                last_buy_px = price
                records.append({"Action": "buy", "Price": price, "Qty": qty, "PnL": np.nan, "Date": dt})

        elif a == "sell" and shares > 0:
            cash += shares * price
            pnl = (price - (last_buy_px if last_buy_px is not None else price)) * shares
            records.append({"Action": "sell", "Price": price, "Qty": shares, "PnL": pnl, "Date": dt})
            shares = 0
            last_buy_px = None

    trades = pd.DataFrame.from_records(records)
    if not trades.empty:
        trades.set_index("Date", inplace=True)
        trades.index = pd.to_datetime(trades.index).tz_localize(None)

    # Profit = sum of PnL on sell rows
    total_profit = float(pd.to_numeric(trades.get("PnL", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    num_round_trips = int((trades.get("Action", pd.Series(dtype=str)).str.lower() == "sell").sum())

    # Equity curve (cash + MV replay)
    eq_curve = compute_equity_curve(df, trades, initial_equity=initial_equity)

    stats: Dict[str, Any] = {
        "profit": total_profit,
        "trades": num_round_trips,
        "equity_curve": eq_curve,
    }

    # optional extras
    if num_round_trips > 0:
        avg_trade_pnl = total_profit / num_round_trips
        stats["avg_trade_pnl"] = avg_trade_pnl

        sells = trades[trades["Action"].str.lower() == "sell"]
        if not sells.empty and "PnL" in sells:
            wins = (sells["PnL"] > 0).sum()
            stats["win_rate"] = wins / len(sells)

    return _ensure_trade_columns(trades), stats


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run_simulation(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame], Union[pd.Series, pd.DataFrame, Tuple[Any, ...]]],
    initial_equity: float = 10_000.0,
    **kwargs,  # safely ignore extra args like strategy_name from older callers
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run a strategy on price data and produce (trades, stats).

    Returns
    -------
    trades : pd.DataFrame
        Index = trade timestamps; columns include Action, Price, Qty, PnL (PnL on sells).
    stats : dict
        Contains at least: 'profit', 'trades', and 'equity_curve' (pd.Series).
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["Action", "Price", "Qty", "PnL"]), {
            "profit": 0.0, "trades": 0, "equity_curve": pd.Series(dtype=float)
        }

    signals = _normalize_strategy_output(df, strategy_fn)
    trades, stats = _simulate_from_signals(df, signals, initial_equity=initial_equity)

    # safety: ensure trades standardized & equity curve present
    trades = _ensure_trade_columns(trades)
    if "equity_curve" not in stats or not isinstance(stats["equity_curve"], pd.Series):
        try:
            stats["equity_curve"] = compute_equity_curve(df, trades, initial_equity=initial_equity)
        except Exception:
            stats["equity_curve"] = pd.Series(dtype=float)

    stats.setdefault("profit", float(pd.to_numeric(trades.get("PnL", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()))
    stats.setdefault("trades", int((trades.get("Action", pd.Series(dtype=str)).str.lower() == "sell").sum()))

    return trades, stats
