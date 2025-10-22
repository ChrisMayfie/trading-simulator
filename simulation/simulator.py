# simulation/simulator.py
from __future__ import annotations
import pandas as pd
import numpy as np

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
    try:
        s.index = pd.to_datetime(s.index).tz_localize(None)
    except Exception:
        try:
            s.index = pd.to_datetime(s.index).tz_convert(None)
        except Exception:
            s.index = pd.to_datetime(s.index, errors="coerce")
    s.name = "Close"
    return s

def _to_signals(df: pd.DataFrame, strategy_fn) -> pd.Series:
    """
    Normalizes strategy output to a 1D signal Series on df's index.
    Expected: long-only signals in {-1, 0, +1} or in {0, 1}.
    """
    out = strategy_fn(df)
    if isinstance(out, (pd.DataFrame, pd.Series)):
        sig = out.squeeze()
    else:
        sig = pd.Series(out, index=df.index)
    sig = pd.to_numeric(sig, errors="coerce").fillna(0.0)
    # clamp to {-1, 0, 1}
    sig = sig.clip(-1, 1)
    sig.index = pd.to_datetime(sig.index).tz_localize(None) if getattr(sig.index, "tz", None) else pd.to_datetime(sig.index)
    return sig

def run_simulation(
    df: pd.DataFrame,
    strategy_fn,
    *,
    initial_equity: float = 10_000.0,
    position_size_pct: float = 1.0,        # 1.0 = 100% of available cash
    stop_loss_pct: float | None = None,    # e.g., 0.05 for 5% below entry -> exit
    take_profit_pct: float | None = None,  # e.g., 0.1 for 10% above entry -> exit
    commission: float = 0.0,               # fixed $ per trade (applied on buy and sell)
    slippage_bps: float = 0.0              # slippage in basis points (e.g., 10 = 0.10%)
):
    """
    Long-only, single-position backtester.
    - Buys when signal > 0 and flat.
    - Sells when signal <= 0 OR stop-loss / take-profit triggers while in position.
    - Buys/Sells at close with slippage applied: price * (1 +/- bps)
    """
    px = _close_series(df).sort_index()
    signals = _to_signals(df, strategy_fn).reindex(px.index).fillna(0.0)

    cash = float(initial_equity)
    shares = 0
    entry_price = None

    records = []  # rows: index(date), Action, Price, PnL
    equity_vals = []

    slip_buy = lambda p: p * (1.0 + slippage_bps / 10_000.0)
    slip_sell = lambda p: p * (1.0 - slippage_bps / 10_000.0)

    for dt, price_raw in px.items():
        price = float(price_raw)

        # While in position, evaluate risk exits vs *current* price
        if shares > 0 and entry_price is not None:
            if stop_loss_pct and stop_loss_pct > 0:
                stop_trigger = price <= entry_price * (1.0 - float(stop_loss_pct))
            else:
                stop_trigger = False

            if take_profit_pct and take_profit_pct > 0:
                tp_trigger = price >= entry_price * (1.0 + float(take_profit_pct))
            else:
                tp_trigger = False

            if stop_trigger or tp_trigger:
                exit_px = slip_sell(price)
                cash += shares * exit_px
                cash -= commission if commission > 0 else 0.0
                pnl = (exit_px - entry_price) * shares - (commission if commission > 0 else 0.0)
                # NOTE: store the date for this action
                records.append({"Date": dt, "Action": "Sell", "Price": exit_px, "PnL": pnl})
                shares = 0
                entry_price = None

        sig = signals.loc[dt]

        # Entry/Exit from signal
        if shares == 0:
            if sig > 0:
                buy_cash = cash * float(position_size_pct)
                if buy_cash > 0:
                    buy_px = slip_buy(price)
                    qty = int(buy_cash // buy_px)
                    if qty > 0:
                        cost = qty * buy_px
                        cash -= cost
                        cash -= commission if commission > 0 else 0.0
                        shares = qty
                        entry_price = buy_px
                        # NOTE: store the date for this action
                        records.append({"Date": dt, "Action": "Buy", "Price": buy_px, "PnL": np.nan})
        else:
            if sig <= 0:
                sell_px = slip_sell(price)
                cash += shares * sell_px
                cash -= commission if commission > 0 else 0.0
                pnl = (sell_px - entry_price) * shares - (commission if commission > 0 else 0.0)
                # NOTE: store the date for this action
                records.append({"Date": dt, "Action": "Sell", "Price": sell_px, "PnL": pnl})
                shares = 0
                entry_price = None

        equity_vals.append(cash + shares * price)

    # Equity curve
    equity_curve = pd.Series(equity_vals, index=px.index, name="equity")

    # Build trades df SAFELY (no zero-length index)
    trades = pd.DataFrame(records)
    if not trades.empty:
        trades["Date"] = pd.to_datetime(trades["Date"])
        trades = trades.set_index("Date").sort_index()

    total_profit = float(equity_curve.iloc[-1] - initial_equity)
    stats = {
        "profit": total_profit,
        "trades": int((trades["Action"] == "Sell").sum() + (trades["Action"] == "Buy").sum()) if not trades.empty else 0,
        "equity_curve": equity_curve,
        "position_size_pct": float(position_size_pct),
        "stop_loss_pct": float(stop_loss_pct) if stop_loss_pct else None,
        "take_profit_pct": float(take_profit_pct) if take_profit_pct else None,
        "commission": float(commission),
        "slippage_bps": float(slippage_bps),
    }
    return trades, stats
