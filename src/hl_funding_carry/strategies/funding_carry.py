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

    def _base_symbol_budget(self, symbol: str) -> float:
        risk = self.config.risk
        max_active = max(1, risk.max_active_symbols)
        per_symbol_cap = risk.max_notional_per_symbol
        if per_symbol_cap is None:
            per_symbol_cap = risk.max_notional_pct
        if risk.allocation_mode == "fixed_notional":
            fixed = risk.fixed_notional_per_symbol or min(
                risk.max_notional_pct,
                risk.max_gross_exposure / max_active,
            )
            return min(fixed, per_symbol_cap)
        equal_weight_budget = risk.max_gross_exposure / max_active
        weighted_budget = equal_weight_budget * SYMBOL_SIZE_WEIGHTS.get(symbol, 1.0)
        return min(weighted_budget, risk.max_notional_pct, per_symbol_cap)

    def position_sizing(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        sized = signal_df.copy()
        sized["symbol_weight"] = sized["symbol"].map(SYMBOL_SIZE_WEIGHTS).fillna(1.0)
        sized["target_size"] = sized["symbol"].map(
            lambda symbol: self._base_symbol_budget(str(symbol)),
        )
        return sized

    def _build_entry_state(self, row: pd.Series, direction: int) -> PositionState:
        funding_event_time = as_timestamp(row["execution_funding_time"])
        entry_time = as_timestamp(row["execution_timestamp"])
        return PositionState(
            direction=direction,
            size=float(row["target_size"]),
            entry_basis=float(row["basis"]),
            entry_pred_funding=float(row["pred_funding_1h"]),
            entry_time=entry_time,
            funding_event_time=funding_event_time,
            min_exit_time=funding_event_time
            + pd.Timedelta(minutes=self.config.timing.min_hold_minutes_after_funding),
            max_exit_time=entry_time + pd.Timedelta(minutes=self.config.timing.max_hold_minutes),
        )

    @staticmethod
    def _current_gross_exposure(states: dict[str, PositionState]) -> float:
        return float(sum(abs(state.direction * state.size) for state in states.values()))

    @staticmethod
    def _current_active_count(states: dict[str, PositionState]) -> int:
        return sum(1 for state in states.values() if state.direction != 0)

    def generate_target_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        signal_df = self.position_sizing(self.generate_signal(df))
        results: list[dict[str, Any]] = []
        symbols = signal_df["symbol"].drop_duplicates().sort_values().tolist()
        states = {str(symbol): PositionState() for symbol in symbols}
        last_entry_times: dict[str, pd.Timestamp | None] = {str(symbol): None for symbol in symbols}

        for timestamp, timestamp_df in signal_df.groupby("timestamp", sort=True):
            bar_timestamp = as_timestamp(timestamp)
            output_meta: dict[str, dict[str, Any]] = {
                str(row["symbol"]): {
                    "signal": "flat",
                    "reason_code": "flat",
                    "expected_hold_until": states[str(row["symbol"])].max_exit_time,
                    "state_for_output": states[str(row["symbol"])],
                }
                for _, row in timestamp_df.iterrows()
            }

            for _, row in timestamp_df.iterrows():
                symbol = str(row["symbol"])
                exit_reason = self.apply_exit_rules(row, states[symbol])
                if exit_reason is not None:
                    output_meta[symbol] = {
                        "signal": exit_reason,
                        "reason_code": exit_reason,
                        "expected_hold_until": None,
                        "state_for_output": states[symbol],
                    }
                    states[symbol] = PositionState()

            candidates = timestamp_df.copy()
            candidates["entry_priority"] = candidates["carry_score"].abs()
            candidates = candidates.loc[candidates["entry_side"] != "flat"].sort_values(
                "entry_priority",
                ascending=False,
            )
            top_n = self.config.risk.top_n_signals
            if top_n is not None:
                candidates = candidates.head(top_n)

            for _, row in candidates.iterrows():
                symbol = str(row["symbol"])
                if states[symbol].direction != 0:
                    continue
                if self._current_active_count(states) >= self.config.risk.max_active_symbols:
                    continue
                last_entry_time = last_entry_times[symbol]
                cooldown_ok = True
                if last_entry_time is not None:
                    cooldown_ok = (
                        (bar_timestamp - last_entry_time).total_seconds() / 3600.0
                    ) >= float(self.config.min_signal_interval_hours)
                if not cooldown_ok:
                    continue

                direction = 0
                if int(row["entry_long_spot_short_perp"]) == 1:
                    direction = 1
                elif int(row["entry_short_spot_long_perp"]) == 1:
                    direction = -1
                if direction == 0:
                    continue

                proposed_state = self._build_entry_state(row, direction)
                if (
                    self._current_gross_exposure(states) + abs(proposed_state.size)
                    > self.config.risk.max_gross_exposure
                ):
                    continue
                states[symbol] = proposed_state
                last_entry_times[symbol] = bar_timestamp
                output_meta[symbol] = {
                    "signal": str(row["entry_side"]),
                    "reason_code": "entry_signal",
                    "expected_hold_until": proposed_state.max_exit_time,
                    "state_for_output": proposed_state,
                }

            gross_exposure = self._current_gross_exposure(states)
            active_count = self._current_active_count(states)
            for _, row in timestamp_df.iterrows():
                symbol = str(row["symbol"])
                state = states[symbol]
                meta = output_meta[symbol]
                target_position = float(state.direction) * float(state.size)
                entry_side = "flat"
                if state.direction > 0:
                    entry_side = "long_spot_short_perp"
                elif state.direction < 0:
                    entry_side = "short_spot_long_perp"
                payload: dict[str, Any] = {str(key): value for key, value in row.to_dict().items()}
                payload.update(
                    {
                        "signal": meta["signal"],
                        "target_position": target_position,
                        "entry_side": entry_side,
                        "expected_hold_until": meta["expected_hold_until"],
                        "reason_code": meta["reason_code"],
                        "active_funding_event_time": meta["state_for_output"].funding_event_time,
                        "min_exit_time": meta["state_for_output"].min_exit_time,
                        "max_exit_time": meta["state_for_output"].max_exit_time,
                        "gross_exposure": gross_exposure,
                        "active_symbol_count": active_count,
                        "allocation_mode": self.config.risk.allocation_mode,
                        "symbol_target_weight": state.size,
                    },
                )
                results.append(payload)

        return pd.DataFrame(results).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
