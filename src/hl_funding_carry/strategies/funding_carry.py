from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from hl_funding_carry.backtest.events import add_funding_event_calendar, as_timestamp
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
    funding_event_time: pd.Timestamp | None = None
    min_exit_time: pd.Timestamp | None = None
    max_exit_time: pd.Timestamp | None = None


class FundingCarryStrategy:
    def __init__(self, config: FundingCarryConfig | StrategyConfig) -> None:
        self.config = config.strategy if isinstance(config, FundingCarryConfig) else config

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_funding_features(df)

    def add_timing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_funding_event_calendar(df, self.config.timing.funding_interval_minutes)

    def generate_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        entry = self.config.entry
        timing = self.config.timing
        timed = self.add_timing_columns(df)
        signals = timed.copy()
        entry_window = (
            signals["execution_timestamp"].notna()
            & signals["execution_minutes_to_funding"].ge(0.0)
            & signals["execution_minutes_to_funding"].le(float(timing.entry_lead_minutes))
        )
        signals["entry_long_spot_short_perp"] = (
            entry_window
            & (signals["pred_funding_1h"] > entry.predicted_funding_min)
            & (signals["current_funding"] > entry.current_funding_min)
            & (signals["basis"] > entry.basis_min)
            & (signals["oi_change_1h"].fillna(0.0) >= entry.oi_change_1h_min)
            & (signals["spread_bps"] < entry.spread_bps_max)
        ).astype(int)
        signals["entry_short_spot_long_perp"] = (
            entry_window
            & (signals["pred_funding_1h"] < -entry.predicted_funding_min)
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
        signals["entry_window_open"] = entry_window.astype(int)
        return signals

    def apply_exit_rules(self, row: pd.Series, state: PositionState) -> str | None:
        if (
            state.direction == 0
            or state.entry_time is None
            or state.funding_event_time is None
            or state.min_exit_time is None
            or state.max_exit_time is None
        ):
            return None

        exit_config = self.config.exit
        current_timestamp = as_timestamp(row["timestamp"])
        if state.direction > 0 and float(row["basis"]) > state.entry_basis + exit_config.basis_stop:
            return "exit_basis_stop"
        if state.direction < 0 and float(row["basis"]) < state.entry_basis - exit_config.basis_stop:
            return "exit_basis_stop"
        if current_timestamp >= state.max_exit_time:
            return "exit_time_stop"
        if current_timestamp < state.min_exit_time:
            return None
        if state.direction > 0 and float(row["basis"]) < exit_config.basis_exit:
            return "exit_basis_normalized"
        if state.direction < 0 and float(row["basis"]) > -exit_config.basis_exit:
            return "exit_basis_normalized"
        if (
            abs(float(row["pred_funding_1h"]))
            < abs(state.entry_pred_funding) * exit_config.predicted_funding_decay_ratio
        ):
            return "exit_funding_decay"
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
        last_entry_times: dict[str, pd.Timestamp | None] = {symbol: None for symbol in states}

        for timestamp, timestamp_df in signal_df.groupby("timestamp", sort=True):
            bar_timestamp = as_timestamp(timestamp)
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
                expected_hold_until = state.max_exit_time
                state_for_output = state

                if exit_reason is not None:
                    state_for_output = state
                    states[symbol] = PositionState()
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

                    last_entry_time = last_entry_times[symbol]
                    cooldown_ok = True
                    if last_entry_time is not None:
                        cooldown_ok = (
                            (bar_timestamp - last_entry_time).total_seconds() / 3600.0
                        ) >= float(self.config.min_signal_interval_hours)

                    funding_event_time = as_timestamp(row["execution_funding_time"])
                    if direction != 0 and cooldown_ok:
                        state = PositionState(
                            direction=direction,
                            size=float(row["target_size"]),
                            entry_basis=float(row["basis"]),
                            entry_pred_funding=float(row["pred_funding_1h"]),
                            entry_time=as_timestamp(row["execution_timestamp"]),
                            funding_event_time=funding_event_time,
                            min_exit_time=funding_event_time
                            + pd.Timedelta(
                                minutes=self.config.timing.min_hold_minutes_after_funding,
                            ),
                            max_exit_time=as_timestamp(row["execution_timestamp"])
                            + pd.Timedelta(minutes=self.config.timing.max_hold_minutes),
                        )
                        states[symbol] = state
                        last_entry_times[symbol] = bar_timestamp
                        active_count += 1
                        signal = entry_side
                        reason_code = "entry_signal"
                        expected_hold_until = state.max_exit_time
                        state_for_output = state

                state = states[symbol]
                target_position = float(state.direction) * float(state.size)
                entry_side = "flat"
                if state.direction > 0:
                    entry_side = "long_spot_short_perp"
                elif state.direction < 0:
                    entry_side = "short_spot_long_perp"

                payload: dict[str, Any] = {str(key): value for key, value in row.to_dict().items()}
                payload.update(
                    {
                        "signal": signal,
                        "target_position": target_position,
                        "entry_side": entry_side,
                        "expected_hold_until": expected_hold_until,
                        "reason_code": reason_code,
                        "active_funding_event_time": state_for_output.funding_event_time,
                        "min_exit_time": state_for_output.min_exit_time,
                        "max_exit_time": state_for_output.max_exit_time,
                    },
                )
                results.append(payload)

        out = pd.DataFrame(results)
        return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
