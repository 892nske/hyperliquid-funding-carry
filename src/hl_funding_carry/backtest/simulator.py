from __future__ import annotations

import pandas as pd

from hl_funding_carry.backtest.metrics import summarize_backtest
from hl_funding_carry.settings import FundingCarryConfig


def simulate_backtest(
    target_df: pd.DataFrame,
    config: FundingCarryConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
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
        frame["funding_pnl"] = -frame["position_perp"] * frame["current_funding"]

        frame["spot_turnover"] = (
            frame["position_spot"].diff().abs().fillna(frame["position_spot"].abs())
        )
        frame["perp_turnover"] = (
            frame["position_perp"].diff().abs().fillna(frame["position_perp"].abs())
        )
        frame["fee"] = (frame["spot_turnover"] + frame["perp_turnover"]) * fee_rate
        frame["slippage_cost"] = (
            (frame["spot_turnover"] + frame["perp_turnover"]) * slippage_rate
        )
        frame["total_pnl"] = (
            frame["price_pnl"] + frame["funding_pnl"] - frame["fee"] - frame["slippage_cost"]
        )
        frame["cum_pnl"] = frame["total_pnl"].cumsum()
        frame["position_change"] = frame["position_pair"].diff().fillna(frame["position_pair"])

        trade_ids: list[int | None] = []
        holding_hours: list[float] = []
        trade_id = 0
        open_trade_id: int | None = None
        entry_time: pd.Timestamp | None = None

        for _, row in frame.iterrows():
            position = float(row["position_pair"])
            timestamp = pd.Timestamp(row["timestamp"])
            if position != 0.0 and open_trade_id is None:
                trade_id += 1
                open_trade_id = trade_id
                entry_time = timestamp
            elif position == 0.0 and open_trade_id is not None:
                open_trade_id = None
                entry_time = None

            trade_ids.append(open_trade_id)
            if open_trade_id is None or entry_time is None:
                holding_hours.append(0.0)
            else:
                holding_hours.append((timestamp - entry_time).total_seconds() / 3600.0)

        frame["trade_id"] = trade_ids
        frame["holding_hours"] = holding_hours
        records.append(frame)

    backtest_df = pd.concat(records, ignore_index=True).sort_values(["timestamp", "symbol"])
    backtest_df = backtest_df.reset_index(drop=True)
    summary = summarize_backtest(backtest_df)
    return backtest_df, summary
