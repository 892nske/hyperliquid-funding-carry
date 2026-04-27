from __future__ import annotations

import pandas as pd

from hl_funding_carry.types import BacktestResult


def build_attribution_tables(
    run_id: str,
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pnl_attribution = (
        ledger.groupby(["run_id", "symbol", "execution_model"], as_index=False)[
            [
                "price_pnl_spot",
                "price_pnl_perp",
                "funding_pnl",
                "fee",
                "slippage",
                "timing_drag",
                "total_pnl",
            ]
        ]
        .sum()
        .sort_values(["run_id", "symbol", "execution_model"])
        .reset_index(drop=True)
    )

    execution_summary = (
        ledger.groupby(["run_id", "execution_model"], as_index=False)
        .agg(
            trade_rows=("position_pair", "size"),
            avg_fee=("fee", "mean"),
            avg_slippage=("slippage", "mean"),
            avg_timing_drag=("timing_drag", "mean"),
            fallback_count=("execution_fallback", lambda series: int(series.ne("none").sum())),
        )
        .sort_values(["run_id", "execution_model"])
        .reset_index(drop=True)
    )

    trade_attribution = trades.copy()
    if not trade_attribution.empty:
        trade_attribution["run_id"] = run_id
        trade_attribution = trade_attribution.sort_values(["run_id", "symbol", "entry_time"])
        trade_attribution = trade_attribution.reset_index(drop=True)

    return pnl_attribution, execution_summary, trade_attribution


def build_attribution_from_result(result: BacktestResult) -> BacktestResult:
    pnl_attribution, execution_summary, trade_attribution = build_attribution_tables(
        result.run_id,
        result.ledger,
        result.trades,
    )
    result.pnl_attribution = pnl_attribution
    result.execution_summary = execution_summary
    result.trade_attribution = trade_attribution
    return result
