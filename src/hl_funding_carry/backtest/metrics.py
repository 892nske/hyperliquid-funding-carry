from __future__ import annotations

import math

import pandas as pd


def summarize_backtest(df: pd.DataFrame) -> dict[str, float]:
    pnl = df["total_pnl"].fillna(0.0)
    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax()
    trade_count = float(df["position_change"].ne(0.0).sum() / 2.0)

    holding_periods = (
        df.dropna(subset=["trade_id"])
        .groupby(["symbol", "trade_id"], sort=False)["holding_hours"]
        .max()
    )
    sharpe_like = 0.0
    pnl_std = float(pnl.std())
    if pnl_std > 0.0:
        sharpe_like = float(pnl.mean() / pnl_std * math.sqrt(24.0 * 365.0))

    return {
        "total_return": float(pnl.sum()),
        "sharpe_like": sharpe_like,
        "max_drawdown": float(drawdown.min()),
        "trade_count": trade_count,
        "average_holding_period": (
            float(holding_periods.mean()) if not holding_periods.empty else 0.0
        ),
    }
