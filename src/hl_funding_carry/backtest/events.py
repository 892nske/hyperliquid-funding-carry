from __future__ import annotations

from typing import Any, cast

import pandas as pd


def infer_bar_interval_minutes(df: pd.DataFrame) -> int:
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    diffs = timestamps.sort_values().drop_duplicates().diff().dropna()
    if diffs.empty:
        return 60
    seconds = diffs.dt.total_seconds().median()
    if pd.isna(seconds) or float(seconds) <= 0.0:
        return 60
    return max(1, int(round(float(seconds) / 60.0)))


def add_funding_event_calendar(
    df: pd.DataFrame,
    funding_interval_minutes: int,
) -> pd.DataFrame:
    calendar = df.copy()
    calendar["timestamp"] = pd.to_datetime(calendar["timestamp"], utc=True)
    frequency = f"{funding_interval_minutes}min"
    calendar["prev_funding_time"] = calendar["timestamp"].dt.floor(frequency)
    calendar["next_funding_time"] = calendar["timestamp"].dt.ceil(frequency)
    is_event = calendar["timestamp"] == calendar["prev_funding_time"]
    calendar["current_funding_event_time"] = pd.to_datetime(
        calendar["timestamp"].where(is_event),
        utc=True,
    )
    calendar.loc[is_event, "next_funding_time"] = (
        calendar.loc[is_event, "timestamp"] + pd.Timedelta(minutes=funding_interval_minutes)
    )
    calendar["is_funding_event"] = is_event.astype(int)
    calendar["minutes_since_prev_funding"] = (
        (calendar["timestamp"] - calendar["prev_funding_time"]).dt.total_seconds() / 60.0
    )
    calendar["minutes_to_next_funding"] = (
        (calendar["next_funding_time"] - calendar["timestamp"]).dt.total_seconds() / 60.0
    )
    next_bar_timestamp = calendar.groupby("symbol")["timestamp"].shift(-1)
    calendar["next_bar_timestamp"] = next_bar_timestamp
    calendar["execution_timestamp"] = next_bar_timestamp
    execution_timestamp = pd.to_datetime(calendar["execution_timestamp"], utc=True)
    calendar["execution_funding_time"] = execution_timestamp.dt.ceil(frequency)
    execution_is_event = execution_timestamp == execution_timestamp.dt.floor(frequency)
    calendar.loc[execution_is_event, "execution_funding_time"] = execution_timestamp.loc[
        execution_is_event
    ]
    calendar["execution_minutes_to_funding"] = (
        (
            pd.to_datetime(calendar["execution_funding_time"], utc=True)
            - execution_timestamp
        ).dt.total_seconds()
        / 60.0
    )
    calendar["execution_minutes_since_prev_funding"] = (
        (
            pd.to_datetime(calendar["execution_timestamp"], utc=True)
            - pd.to_datetime(calendar["prev_funding_time"], utc=True)
        ).dt.total_seconds()
        / 60.0
    )
    return calendar


def as_timestamp(value: Any) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.to_datetime(str(value), utc=True))
