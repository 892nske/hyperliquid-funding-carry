from __future__ import annotations

import pandas as pd

from hl_funding_carry.types import BacktestResult


def build_attribution_tables(
    run_id: str,
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    summary: dict[str, float | str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timestamp_exposure = (
        ledger.groupby("timestamp", as_index=False)
        .agg(
            gross_exposure=("gross_exposure", "max"),
            active_symbol_count=("active_symbol_count", "max"),
        )
    )
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
            avg_gross_exposure=("gross_exposure", "mean"),
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

    portfolio_summary = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "execution_model": summary.get("execution_model", ""),
                "total_return": summary.get("total_return", 0.0),
                "sharpe_like": summary.get("sharpe_like", 0.0),
                "max_drawdown": summary.get("max_drawdown", 0.0),
                "trade_count": summary.get("trade_count", 0.0),
                "average_holding_period": summary.get("average_holding_period", 0.0),
                "mean_gross_exposure": float(timestamp_exposure["gross_exposure"].mean())
                if "gross_exposure" in ledger.columns
                else 0.0,
                "max_active_symbols": float(timestamp_exposure["active_symbol_count"].max())
                if "active_symbol_count" in ledger.columns
                else 0.0,
            },
        ],
    )

    symbol_summary = (
        ledger.groupby(["run_id", "symbol"], as_index=False)[
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
        .sort_values(["run_id", "symbol"])
        .reset_index(drop=True)
    )
    trade_counts = trades.groupby("symbol", as_index=False).size().rename(
        columns={"size": "trade_count"},
    )
    symbol_summary = symbol_summary.merge(trade_counts, on="symbol", how="left")
    symbol_summary["trade_count"] = symbol_summary["trade_count"].fillna(0.0)
    symbol_summary["execution_model"] = summary.get("execution_model", "")
    return (
        pnl_attribution,
        execution_summary,
        trade_attribution,
        portfolio_summary,
        symbol_summary,
    )


def build_attribution_from_result(result: BacktestResult) -> BacktestResult:
    (
        result.pnl_attribution,
        result.execution_summary,
        result.trade_attribution,
        result.portfolio_summary,
        result.symbol_summary,
    ) = build_attribution_tables(
        result.run_id,
        result.ledger,
        result.trades,
        result.summary,
    )
    return result
