from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from hl_funding_carry.features.funding import build_funding_features
from hl_funding_carry.settings import FundingCarryConfig, StrategyConfig

SYMBOL_SIZE_WEIGHTS = {"BTC": 1.0, "ETH": 0.8, "HYPE": 0.5}


@dataclass
class PositionState:
    direction: int = 0
    size: float = 0.0
    entry_basis: float = 0.0
    entry_pred_funding: float = 0.0
    entry_time: pd.Timestamp | None = None
    expected_hold_until: pd.Timestamp | None = None


def _as_timestamp(value: Any) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.to_datetime(value, utc=True))


class FundingCarryStrategy:
    def __init__(self, config: FundingCarryConfig | StrategyConfig) -> None:
        self.config = config.strategy if isinstance(config, FundingCarryConfig) else config

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_funding_features(df)

    def generate_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        entry = self.config.entry
        signals = df.copy()
        signals["entry_long_spot_short_perp"] = (
            (signals["pred_funding_1h"] > entry.predicted_funding_min)
            & (signals["current_funding"] > entry.current_funding_min)
            & (signals["basis"] > entry.basis_min)
            & (signals["oi_change_1h"].fillna(0.0) >= entry.oi_change_1h_min)
            & (signals["spread_bps"] < entry.spread_bps_max)
        ).astype(int)
        signals["entry_short_spot_long_perp"] = (
            (signals["pred_funding_1h"] < -entry.predicted_funding_min)
            & (signals["current_funding"] < -entry.current_funding_min)
            & (signals["basis"] < -entry.basis_min)
            & (signals["oi_change_1h"].fillna(0.0) >= entry.oi_change_1h_min)
            & (signals["spread_bps"] < entry.spread_bps_max)
        ).astype(int)
        signals["entry_side"] = "flat"
        signals.loc[signals["entry_long_spot_short_perp"] == 1, "entry_side"] = (
            "long_spot_short_perp"
        )
        signals.loc[signals["entry_short_spot_long_perp"] == 1, "entry_side"] = (
            "short_spot_long_perp"
        )
        signals["signal"] = signals["entry_side"]
        signals["reason_code"] = "flat"
        signals.loc[signals["entry_side"] != "flat", "reason_code"] = "entry_signal"
        return signals

    def apply_exit_rules(self, row: pd.Series, state: PositionState) -> str | None:
        if state.direction == 0 or state.entry_time is None or state.expected_hold_until is None:
            return None

        exit_config = self.config.exit
        current_timestamp = _as_timestamp(row["timestamp"])
        holding_hours = (current_timestamp - state.entry_time).total_seconds() / 3600.0
        if state.direction > 0 and row["basis"] < exit_config.basis_exit:
            return "exit_basis_normalized"
        if state.direction < 0 and row["basis"] > -exit_config.basis_exit:
            return "exit_basis_normalized"
        if state.direction > 0 and row["basis"] > state.entry_basis + exit_config.basis_stop:
            return "exit_basis_stop"
        if state.direction < 0 and row["basis"] < state.entry_basis - exit_config.basis_stop:
            return "exit_basis_stop"
        if (
            abs(float(row["pred_funding_1h"]))
            < abs(state.entry_pred_funding) * exit_config.predicted_funding_decay_ratio
        ):
            return "exit_funding_decay"
        if holding_hours >= float(exit_config.max_hold_hours):
            return "exit_time_stop"
        return None

    def position_sizing(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        sized = signal_df.copy()
        sized["symbol_weight"] = sized["symbol"].map(SYMBOL_SIZE_WEIGHTS).fillna(0.0)
        sized["target_size"] = sized["symbol_weight"] * self.config.risk.max_notional_pct
        return sized

    def generate_target_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        signal_df = self.position_sizing(self.generate_signal(df))
        results: list[dict[str, Any]] = []
        states = {
            symbol: PositionState()
            for symbol in signal_df["symbol"].drop_duplicates().sort_values().tolist()
        }

        for timestamp, timestamp_df in signal_df.groupby("timestamp", sort=True):
            bar_timestamp = _as_timestamp(timestamp)
            active_count = sum(1 for state in states.values() if state.direction != 0)
            candidates = timestamp_df.copy()
            candidates["entry_priority"] = candidates["carry_score"].abs()
            candidates = candidates.sort_values("entry_priority", ascending=False)

            for _, row in candidates.iterrows():
                symbol = str(row["symbol"])
                state = states[symbol]
                exit_reason = self.apply_exit_rules(row, state)
                signal = "flat"
                reason_code = "flat"
                expected_hold_until: pd.Timestamp | None = state.expected_hold_until

                if exit_reason is not None:
                    state = PositionState()
                    states[symbol] = state
                    active_count = sum(1 for item in states.values() if item.direction != 0)
                    signal = exit_reason
                    reason_code = exit_reason
                    expected_hold_until = None
                elif state.direction == 0 and active_count < self.config.risk.max_positions:
                    direction = 0
                    entry_side = "flat"
                    if int(row["entry_long_spot_short_perp"]) == 1:
                        direction = 1
                        entry_side = "long_spot_short_perp"
                    elif int(row["entry_short_spot_long_perp"]) == 1:
                        direction = -1
                        entry_side = "short_spot_long_perp"

                    if direction != 0:
                        state = PositionState(
                            direction=direction,
                            size=float(row["target_size"]),
                            entry_basis=float(row["basis"]),
                            entry_pred_funding=float(row["pred_funding_1h"]),
                            entry_time=bar_timestamp,
                            expected_hold_until=bar_timestamp
                            + pd.Timedelta(hours=self.config.exit.max_hold_hours),
                        )
                        states[symbol] = state
                        active_count += 1
                        signal = entry_side
                        reason_code = "entry_signal"
                        expected_hold_until = state.expected_hold_until

                state = states[symbol]
                target_position = float(state.direction) * float(state.size)
                entry_side = "flat"
                if state.direction > 0:
                    entry_side = "long_spot_short_perp"
                elif state.direction < 0:
                    entry_side = "short_spot_long_perp"

                payload: dict[str, Any] = {
                    str(key): value for key, value in row.to_dict().items()
                }
                payload.update(
                    {
                        "signal": signal,
                        "target_position": target_position,
                        "entry_side": entry_side,
                        "expected_hold_until": expected_hold_until,
                        "reason_code": reason_code,
                    },
                )
                results.append(
                    payload,
                )

        out = pd.DataFrame(results)
        return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
