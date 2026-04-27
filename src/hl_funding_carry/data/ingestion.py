from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hl_funding_carry.data.hyperliquid import (
    build_funding_inputs,
    fetch_hyperliquid_raw,
    normalize_hyperliquid_asset_context,
    normalize_hyperliquid_candles,
    normalize_hyperliquid_funding_history,
    normalize_hyperliquid_predicted_funding,
)
from hl_funding_carry.data.storage import make_dataset_directory, save_dataframe
from hl_funding_carry.data.validation import validate_processed_directory
from hl_funding_carry.settings import IngestConfig


@dataclass(frozen=True)
class IngestionResult:
    raw_dir: Path
    processed_dir: Path
    validation_summary: pd.DataFrame


def _as_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(str(value))
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def ingest_hyperliquid_batch(config: IngestConfig) -> IngestionResult:
    raw_frames = fetch_hyperliquid_raw(config)
    symbol = config.symbol.upper()
    start = _as_utc_timestamp(config.start)
    end = _as_utc_timestamp(config.end)

    raw_dir = make_dataset_directory(config.raw_output_dir, config.source, symbol, start, end)
    processed_dir = make_dataset_directory(
        config.processed_output_dir,
        config.source,
        symbol,
        start,
        end,
    )

    for dataset_name, frame in raw_frames.items():
        if frame.empty:
            continue
        save_dataframe(frame, raw_dir / dataset_name)

    candles = normalize_hyperliquid_candles(raw_frames["candles_raw"], symbol)
    asset_context = normalize_hyperliquid_asset_context(
        raw_frames["asset_context_raw"],
        symbol,
    )
    funding_history = normalize_hyperliquid_funding_history(
        raw_frames["funding_history_raw"],
        symbol,
    )
    predicted_funding = normalize_hyperliquid_predicted_funding(
        raw_frames["predicted_funding_raw"],
        symbol,
    )
    funding_inputs = build_funding_inputs(asset_context, predicted_funding)

    save_dataframe(candles, processed_dir / "candles")
    save_dataframe(asset_context, processed_dir / "asset_context")
    save_dataframe(funding_history, processed_dir / "funding_history")
    save_dataframe(funding_inputs, processed_dir / "funding_inputs")
    if not predicted_funding.empty:
        save_dataframe(predicted_funding, processed_dir / "predicted_funding")
    if config.execution_5m_path is not None and config.execution_5m_path.exists():
        execution_5m = (
            pd.read_parquet(config.execution_5m_path)
            if config.execution_5m_path.suffix == ".parquet"
            else pd.read_csv(config.execution_5m_path)
        )
        save_dataframe(execution_5m, processed_dir / "execution_5m")
    if config.execution_1m_path is not None and config.execution_1m_path.exists():
        execution_1m = (
            pd.read_parquet(config.execution_1m_path)
            if config.execution_1m_path.suffix == ".parquet"
            else pd.read_csv(config.execution_1m_path)
        )
        save_dataframe(execution_1m, processed_dir / "execution_1m")

    validation_summary = validate_processed_directory(processed_dir)
    save_dataframe(
        validation_summary,
        processed_dir / "data_validation_summary",
        prefer_parquet=False,
    )
    return IngestionResult(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        validation_summary=validation_summary,
    )
