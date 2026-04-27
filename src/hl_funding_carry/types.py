from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

CANDLE_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
)

ASSET_CONTEXT_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "symbol",
    "mark_price",
    "oracle_price",
    "current_funding",
    "open_interest",
    "spread_bps",
    "spot_price",
)

FUNDING_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "symbol",
    "pred_funding_1h",
)

INTRABAR_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

FUNDING_HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "symbol",
    "funding_rate",
)

RESEARCH_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "mark_price",
    "oracle_price",
    "current_funding",
    "pred_funding_1h",
    "open_interest",
    "spread_bps",
    "spot_price",
)


@dataclass
class BacktestResult:
    run_id: str
    summary: dict[str, float | str]
    ledger: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    pnl_attribution: pd.DataFrame
    execution_summary: pd.DataFrame
    trade_attribution: pd.DataFrame
    artifact_dir: Path | None = None


@dataclass(frozen=True)
class FillResult:
    fill_price: float
    benchmark_price: float
    fallback: str


@dataclass(frozen=True)
class ValidationReport:
    dataset: str
    row_count: int
    missing_ratio: float
    duplicate_count: int
    non_monotonic_count: int
    gap_count: int
    max_gap_minutes: float
    min_timestamp: pd.Timestamp | None
    max_timestamp: pd.Timestamp | None
