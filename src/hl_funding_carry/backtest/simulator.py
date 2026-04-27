from __future__ import annotations

from typing import Any

import pandas as pd

from hl_funding_carry.backtest.events import as_timestamp
from hl_funding_carry.backtest.metrics import summarize_backtest
from hl_funding_carry.settings import FundingCarryConfig
from hl_funding_carry.types import BacktestResult


def _build_trade_log(ledger: pd.DataFrame) -> pd.DataFrame:
    trades: list[dict[str, Any]] = []
    for symbol, symbol_df in ledger.groupby("symbol", sort=True):
        open_trade: dict[str, Any] | None = None
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
                }
            if open_trade is not None:
                open_trade.setdefault("trade_rows", [])
                open_trade["trade_rows"].append(row)
                if position_pair == 0.0:
                    trade_rows = pd.DataFrame(open_trade.pop("trade_rows"))
                    entry_time = as_timestamp(open_trade["entry_time"])
                    exit_time = timestamp
                    trades.append(
                        {
                            **open_trade,
                            "exit_time": exit_time,
                            "holding_minutes": (
                                exit_time - entry_time
                            ).total_seconds()
                            / 60.0,
                            "price_pnl_spot": float(trade_rows["price_pnl_spot"].sum()),
                            "price_pnl_perp": float(trade_rows["price_pnl_perp"].sum()),
                            "funding_pnl": float(trade_rows["funding_pnl"].sum()),
                            "fee": float(trade_rows["fee"].sum()),
                            "slippage": float(trade_rows["slippage"].sum()),
                            "total_pnl": float(trade_rows["total_pnl"].sum()),
                        },
                    )
                    open_trade = None
        if open_trade is not None:
            trade_rows = pd.DataFrame(open_trade.pop("trade_rows"))
            exit_time = as_timestamp(symbol_df.iloc[-1]["timestamp"])
            entry_time = as_timestamp(open_trade["entry_time"])
            trades.append(
                {
                    **open_trade,
                    "exit_time": exit_time,
                    "holding_minutes": (exit_time - entry_time).total_seconds() / 60.0,
                    "price_pnl_spot": float(trade_rows["price_pnl_spot"].sum()),
                    "price_pnl_perp": float(trade_rows["price_pnl_perp"].sum()),
                    "funding_pnl": float(trade_rows["funding_pnl"].sum()),
                    "fee": float(trade_rows["fee"].sum()),
                    "slippage": float(trade_rows["slippage"].sum()),
                    "total_pnl": float(trade_rows["total_pnl"].sum()),
                },
            )
    return pd.DataFrame(trades)


def _build_equity_curve(ledger: pd.DataFrame) -> pd.DataFrame:
    equity_curve = (
        ledger.groupby("timestamp", as_index=False)[
            ["price_pnl_spot", "price_pnl_perp", "funding_pnl", "fee", "slippage", "total_pnl"]
        ]
        .sum()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    equity_curve["cum_pnl"] = equity_curve["total_pnl"].cumsum()
    return equity_curve


def simulate_backtest(
    target_df: pd.DataFrame,
    config: FundingCarryConfig,
    run_id: str = "single",
) -> BacktestResult:
    records: list[pd.DataFrame] = []
    fee_rate = config.execution.fee_bps_taker / 10000.0
    slippage_rate = config.execution.slippage_bps / 10000.0

    for _symbol, symbol_df in target_df.groupby("symbol", sort=True):
        frame = symbol_df.copy().sort_values("timestamp").reset_index(drop=True)
        frame["position_pair"] = frame["target_position"].shift(1).fillna(0.0)
        frame["position_spot"] = 0.0
        frame["position_perp"] = frame["position_pair"]
        if config.strategy.mode == "spot_perp":
            frame["position_spot"] = frame["position_pair"]
            frame["position_perp"] = -frame["position_pair"]

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

        frame["spot_turnover"] = (
            frame["position_spot"].diff().abs().fillna(frame["position_spot"].abs())
        )
        frame["perp_turnover"] = (
            frame["position_perp"].diff().abs().fillna(frame["position_perp"].abs())
        )
        frame["fee"] = (frame["spot_turnover"] + frame["perp_turnover"]) * fee_rate
        frame["slippage"] = (frame["spot_turnover"] + frame["perp_turnover"]) * slippage_rate
        frame["total_pnl"] = (
            frame["price_pnl_spot"]
            + frame["price_pnl_perp"]
            + frame["funding_pnl"]
            - frame["fee"]
            - frame["slippage"]
        )
        records.append(frame)

    ledger = pd.concat(records, ignore_index=True).sort_values(["timestamp", "symbol"])
    ledger = ledger.reset_index(drop=True)
    trades = _build_trade_log(ledger)
    equity_curve = _build_equity_curve(ledger)
    summary_values = summarize_backtest(ledger, trades, equity_curve)
    summary: dict[str, float | str] = {key: value for key, value in summary_values.items()}
    summary["run_id"] = run_id
    return BacktestResult(
        run_id=run_id,
        summary=summary,
        ledger=ledger,
        trades=trades,
        equity_curve=equity_curve,
    )
