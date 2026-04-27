from __future__ import annotations

from typing import Literal

import pandas as pd

from hl_funding_carry.backtest.events import as_timestamp
from hl_funding_carry.settings import ExecutionConfig
from hl_funding_carry.types import FillResult


def _typical_price(frame: pd.DataFrame) -> pd.Series:
    return (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0


def _window(
    frame: pd.DataFrame,
    symbol: str,
    start: pd.Timestamp,
    minutes: int,
) -> pd.DataFrame:
    end = start + pd.Timedelta(minutes=minutes)
    mask = (
        (frame["symbol"] == symbol)
        & (frame["timestamp"] >= start)
        & (frame["timestamp"] < end)
    )
    return frame.loc[mask].sort_values("timestamp").reset_index(drop=True)


def _next_open_fill(
    benchmark_price: float,
) -> FillResult:
    return FillResult(
        fill_price=benchmark_price,
        benchmark_price=benchmark_price,
        fallback="none",
    )


def _apply_ratio(
    benchmark_price: float,
    proxy_fill_price: float,
    proxy_open_price: float,
    fallback: str,
) -> FillResult:
    if proxy_open_price <= 0.0:
        return _next_open_fill(benchmark_price)
    ratio = proxy_fill_price / proxy_open_price
    return FillResult(
        fill_price=benchmark_price * ratio,
        benchmark_price=benchmark_price,
        fallback=fallback,
    )


def resolve_execution_fill(
    execution_model: Literal["next_open", "twap_5m", "vwap_1m"],
    execution_config: ExecutionConfig,
    execution_inputs: dict[str, pd.DataFrame],
    symbol: str,
    execution_timestamp: object,
    benchmark_price: float,
) -> FillResult:
    timestamp = as_timestamp(execution_timestamp)
    if execution_model == "next_open":
        return _next_open_fill(benchmark_price)

    if execution_model == "twap_5m":
        frame_5m = execution_inputs.get("5m")
        if frame_5m is not None:
            window = _window(frame_5m, symbol, timestamp, execution_config.twap_window_minutes)
            if not window.empty:
                return _apply_ratio(
                    benchmark_price=benchmark_price,
                    proxy_fill_price=float(_typical_price(window).mean()),
                    proxy_open_price=float(window.iloc[0]["open"]),
                    fallback="none",
                )
        frame_1m = execution_inputs.get("1m")
        if frame_1m is not None:
            window = _window(frame_1m, symbol, timestamp, execution_config.twap_window_minutes)
            if not window.empty:
                return _apply_ratio(
                    benchmark_price=benchmark_price,
                    proxy_fill_price=float(_typical_price(window).mean()),
                    proxy_open_price=float(window.iloc[0]["open"]),
                    fallback="proxy_1m_twap",
                )
        return FillResult(
            fill_price=benchmark_price,
            benchmark_price=benchmark_price,
            fallback="next_open",
        )

    frame_1m = execution_inputs.get("1m")
    if frame_1m is not None:
        window = _window(frame_1m, symbol, timestamp, execution_config.vwap_window_minutes)
        if not window.empty:
            total_volume = float(window["volume"].sum())
            if total_volume > 0.0:
                proxy_fill_price = float((window["close"] * window["volume"]).sum() / total_volume)
                return _apply_ratio(
                    benchmark_price=benchmark_price,
                    proxy_fill_price=proxy_fill_price,
                    proxy_open_price=float(window.iloc[0]["open"]),
                    fallback="none",
                )
            return _apply_ratio(
                benchmark_price=benchmark_price,
                proxy_fill_price=float(_typical_price(window).mean()),
                proxy_open_price=float(window.iloc[0]["open"]),
                fallback="proxy_twap_no_volume",
            )
    frame_5m = execution_inputs.get("5m")
    if frame_5m is not None:
        window = _window(frame_5m, symbol, timestamp, execution_config.vwap_window_minutes)
        if not window.empty:
            return _apply_ratio(
                benchmark_price=benchmark_price,
                proxy_fill_price=float(_typical_price(window).mean()),
                proxy_open_price=float(window.iloc[0]["open"]),
                fallback="proxy_5m_twap",
            )
    return FillResult(
        fill_price=benchmark_price,
        benchmark_price=benchmark_price,
        fallback="next_open",
    )
