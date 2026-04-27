from __future__ import annotations

import math

import pandas as pd


def summarize_backtest(
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
) -> dict[str, float]:
    pnl = equity_curve["total_pnl"].fillna(0.0)
    cumulative = equity_curve["cum_pnl"].fillna(0.0)
    drawdown = cumulative - cumulative.cummax()
    sharpe_like = 0.0
    pnl_std = float(pnl.std())
    if pnl_std > 0.0:
        sharpe_like = float(pnl.mean() / pnl_std * math.sqrt(24.0 * 365.0))

    return {
        "total_return": float(ledger["total_pnl"].sum()),
        "sharpe_like": sharpe_like,
        "max_drawdown": float(drawdown.min()),
        "trade_count": float(len(trades)),
        "average_holding_period": (
            float(trades["holding_minutes"].mean()) if not trades.empty else 0.0
        ),
    }
