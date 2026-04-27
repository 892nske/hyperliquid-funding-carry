from __future__ import annotations

from typing import Final

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
