from __future__ import annotations

from typing import Any

import pandas as pd

from hl_funding_carry.backtest.attribution import build_attribution_tables
from hl_funding_carry.backtest.events import as_timestamp
from hl_funding_carry.backtest.execution import resolve_execution_fill
from hl_funding_carry.backtest.metrics import summarize_backtest
from hl_funding_carry.settings import FundingCarryConfig
from hl_funding_carry.types import BacktestResult


def _first_nonzero(series: pd.Series) -> float:
    nonzero = series.replace(0.0, pd.NA).dropna()
    if nonzero.empty:
        return 0.0
    return float(nonzero.iloc[0])


def _build_trade_log(ledger: pd.DataFrame) -> pd.DataFrame:
    trades: list[dict[str, Any]] = []
    for symbol, symbol_df in ledger.groupby("symbol", sort=True):
        open_trade: dict[str, Any] | None = None
        trade_rows: list[dict[str, Any]] = []
        for _, row in symbol_df.iterrows():
            timestamp = as_timestamp(row["timestamp"])
            position_pair = float(row["position_pair"])
            if open_trade is None and position_pair != 0.0:
                open_trade = {
                    "symbol": symbol,
                    "entry_time": timestamp,
                    "entry_side": row["entry_side"],
                    "direction": float(row["position_pair"]),
                    "funding_event_time": row["active_funding_event_time"],
                    "execution_model": row["execution_model"],
                }
                trade_rows = []
            if open_trade is not None:
                trade_rows.append({str(key): value for key, value in row.to_dict().items()})
                if position_pair == 0.0:
                    trade_frame = pd.DataFrame(trade_rows)
                    entry_time = as_timestamp(open_trade["entry_time"])
                    trades.append(
                        {
                            **open_trade,
                            "exit_time": timestamp,
                            "holding_minutes": (timestamp - entry_time).total_seconds() / 60.0,
                            "spot_fill_price": _first_nonzero(trade_frame["spot_fill_price"]),
                            "perp_fill_price": _first_nonzero(trade_frame["perp_fill_price"]),
                            "price_pnl_spot": float(trade_frame["price_pnl_spot"].sum()),
                            "price_pnl_perp": float(trade_frame["price_pnl_perp"].sum()),
                            "funding_pnl": float(trade_frame["funding_pnl"].sum()),
                            "fee": float(trade_frame["fee"].sum()),
                            "slippage": float(trade_frame["slippage"].sum()),
                            "timing_drag": float(trade_frame["timing_drag"].sum()),
                            "total_pnl": float(trade_frame["total_pnl"].sum()),
                        },
                    )
                    open_trade = None
                    trade_rows = []
        if open_trade is not None:
            trade_frame = pd.DataFrame(trade_rows)
            exit_time = as_timestamp(symbol_df.iloc[-1]["timestamp"])
            entry_time = as_timestamp(open_trade["entry_time"])
            trades.append(
                {
                    **open_trade,
                    "exit_time": exit_time,
                    "holding_minutes": (exit_time - entry_time).total_seconds() / 60.0,
                    "spot_fill_price": _first_nonzero(trade_frame["spot_fill_price"]),
                    "perp_fill_price": _first_nonzero(trade_frame["perp_fill_price"]),
                    "price_pnl_spot": float(trade_frame["price_pnl_spot"].sum()),
                    "price_pnl_perp": float(trade_frame["price_pnl_perp"].sum()),
                    "funding_pnl": float(trade_frame["funding_pnl"].sum()),
                    "fee": float(trade_frame["fee"].sum()),
                    "slippage": float(trade_frame["slippage"].sum()),
                    "timing_drag": float(trade_frame["timing_drag"].sum()),
                    "total_pnl": float(trade_frame["total_pnl"].sum()),
                },
            )
    return pd.DataFrame(trades)


def _build_equity_curve(ledger: pd.DataFrame) -> pd.DataFrame:
    equity_curve = (
        ledger.groupby("timestamp", as_index=False)
        .agg(
            price_pnl_spot=("price_pnl_spot", "sum"),
            price_pnl_perp=("price_pnl_perp", "sum"),
            funding_pnl=("funding_pnl", "sum"),
            fee=("fee", "sum"),
            slippage=("slippage", "sum"),
            timing_drag=("timing_drag", "sum"),
            total_pnl=("total_pnl", "sum"),
            gross_exposure=("gross_exposure", "max"),
            active_symbol_count=("active_symbol_count", "max"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    equity_curve["cum_pnl"] = equity_curve["total_pnl"].cumsum()
    return equity_curve


def _cost_from_fill_diff(
    position_change: float,
    fill_price: float,
    benchmark_price: float,
) -> float:
    if benchmark_price == 0.0 or position_change == 0.0:
        return 0.0
    trade_sign = 1.0 if position_change > 0.0 else -1.0
    return trade_sign * ((fill_price - benchmark_price) / benchmark_price) * abs(position_change)


def simulate_backtest(
    target_df: pd.DataFrame,
    config: FundingCarryConfig,
    execution_inputs: dict[str, pd.DataFrame] | None = None,
    run_id: str = "single",
) -> BacktestResult:
    records: list[pd.DataFrame] = []
    execution_inputs = execution_inputs or {}
    fee_rate = config.execution.fee_bps / 10000.0
    slippage_rate = config.execution.slippage_bps / 10000.0

    for _symbol, symbol_df in target_df.groupby("symbol", sort=True):
        frame = symbol_df.copy().sort_values("timestamp").reset_index(drop=True)
        frame["position_pair"] = frame["target_position"].shift(1).fillna(0.0)
        frame["position_spot"] = 0.0
        frame["position_perp"] = frame["position_pair"]
        if config.strategy.mode == "spot_perp":
            frame["position_spot"] = frame["position_pair"]
            frame["position_perp"] = -frame["position_pair"]

        frame["spot_position_change"] = (
            frame["position_spot"].diff().fillna(frame["position_spot"])
        )
        frame["perp_position_change"] = (
            frame["position_perp"].diff().fillna(frame["position_perp"])
        )
        frame["execution_model"] = config.execution.model
        frame["execution_fallback"] = "none"
        frame["spot_fill_price"] = 0.0
        frame["perp_fill_price"] = 0.0
        frame["spot_fee"] = 0.0
        frame["perp_fee"] = 0.0
        frame["spot_slippage"] = 0.0
        frame["perp_slippage"] = 0.0
        frame["timing_drag_spot"] = 0.0
        frame["timing_drag_perp"] = 0.0

        for index, row in frame.iterrows():
            execution_timestamp = row["timestamp"]
            if float(row["spot_position_change"]) != 0.0:
                spot_fill = resolve_execution_fill(
                    execution_model=config.execution.model,
                    execution_config=config.execution,
                    execution_inputs=execution_inputs,
                    symbol=str(row["symbol"]),
                    execution_timestamp=execution_timestamp,
                    benchmark_price=float(row["spot_price"]),
                )
                frame.loc[index, "spot_fill_price"] = spot_fill.fill_price
                frame.loc[index, "spot_fee"] = abs(float(row["spot_position_change"])) * fee_rate
                frame.loc[index, "spot_slippage"] = (
                    abs(float(row["spot_position_change"])) * slippage_rate
                )
                frame.loc[index, "timing_drag_spot"] = _cost_from_fill_diff(
                    float(row["spot_position_change"]),
                    spot_fill.fill_price,
                    spot_fill.benchmark_price,
                )
                frame.loc[index, "execution_fallback"] = spot_fill.fallback

            if float(row["perp_position_change"]) != 0.0:
                perp_fill = resolve_execution_fill(
                    execution_model=config.execution.model,
                    execution_config=config.execution,
                    execution_inputs=execution_inputs,
                    symbol=str(row["symbol"]),
                    execution_timestamp=execution_timestamp,
                    benchmark_price=float(row["mark_price"]),
                )
                frame.loc[index, "perp_fill_price"] = perp_fill.fill_price
                frame.loc[index, "perp_fee"] = abs(float(row["perp_position_change"])) * fee_rate
                frame.loc[index, "perp_slippage"] = (
                    abs(float(row["perp_position_change"])) * slippage_rate
                )
                frame.loc[index, "timing_drag_perp"] = _cost_from_fill_diff(
                    float(row["perp_position_change"]),
                    perp_fill.fill_price,
                    perp_fill.benchmark_price,
                )
                if str(frame.at[index, "execution_fallback"]) == "none":
                    frame.at[index, "execution_fallback"] = perp_fill.fallback

        frame["spot_return"] = frame["spot_price"].pct_change().fillna(0.0)
        frame["perp_return"] = frame["mark_price"].pct_change().fillna(0.0)
        frame["price_pnl_spot"] = frame["position_spot"] * frame["spot_return"]
        frame["price_pnl_perp"] = frame["position_perp"] * frame["perp_return"]
        frame["price_pnl"] = frame["price_pnl_spot"] + frame["price_pnl_perp"]
        frame["funding_pnl"] = 0.0
        event_mask = frame["is_funding_event"] == 1
        frame.loc[event_mask, "funding_pnl"] = (
            -frame.loc[event_mask, "position_perp"] * frame.loc[event_mask, "current_funding"]
        )
        frame["fee"] = frame["spot_fee"] + frame["perp_fee"]
        frame["slippage"] = frame["spot_slippage"] + frame["perp_slippage"]
        frame["timing_drag"] = frame["timing_drag_spot"] + frame["timing_drag_perp"]
        frame["total_pnl"] = (
            frame["price_pnl_spot"]
            + frame["price_pnl_perp"]
            + frame["funding_pnl"]
            - frame["fee"]
            - frame["slippage"]
            - frame["timing_drag"]
        )
        frame["run_id"] = run_id
        records.append(frame)

    ledger = pd.concat(records, ignore_index=True).sort_values(["timestamp", "symbol"])
    ledger = ledger.reset_index(drop=True)
    trades = _build_trade_log(ledger)
    if not trades.empty:
        trades["run_id"] = run_id
    equity_curve = _build_equity_curve(ledger)
    summary_values = summarize_backtest(ledger, trades, equity_curve)
    summary: dict[str, float | str] = {key: value for key, value in summary_values.items()}
    summary["run_id"] = run_id
    summary["execution_model"] = config.execution.model
    summary["symbol_count"] = float(ledger["symbol"].nunique())
    summary["mean_gross_exposure"] = float(ledger["gross_exposure"].mean())
    (
        pnl_attribution,
        execution_summary,
        trade_attribution,
        portfolio_summary,
        symbol_summary,
    ) = build_attribution_tables(
        run_id,
        ledger,
        trades,
        summary,
    )
    return BacktestResult(
        run_id=run_id,
        summary=summary,
        ledger=ledger,
        trades=trades,
        equity_curve=equity_curve,
        pnl_attribution=pnl_attribution,
        execution_summary=execution_summary,
        trade_attribution=trade_attribution,
        portfolio_summary=portfolio_summary,
        symbol_summary=symbol_summary,
    )
