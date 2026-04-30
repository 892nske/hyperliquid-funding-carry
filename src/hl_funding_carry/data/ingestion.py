from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
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
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    raw_dir: Path
    processed_dir: Path
    validation_summary: pd.DataFrame


@dataclass(frozen=True)
class BulkIngestionResult:
    summary_dir: Path
    batch_summary: pd.DataFrame


def _as_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(str(value))
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _load_optional_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _slice_raw_frame(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    for column in ("timestamp", "time", "t"):
        if column in frame.columns:
            timestamp = pd.to_datetime(frame[column], utc=True)
            return frame.loc[(timestamp >= start) & (timestamp < end)].reset_index(drop=True)
    return frame


def _save_execution_tables(
    raw_frames: dict[str, pd.DataFrame],
    symbol: str,
    processed_dir: Path,
) -> None:
    if not raw_frames["execution_1m_raw"].empty:
        execution_1m = normalize_hyperliquid_candles(
            raw_frames["execution_1m_raw"],
            symbol,
            floor_to="1min",
        )
        execution_1m = execution_1m.loc[
            :,
            ["timestamp", "symbol", "open", "high", "low", "close", "volume"],
        ]
        save_dataframe(execution_1m, processed_dir / "execution_1m")
    if not raw_frames["execution_5m_raw"].empty:
        execution_5m = normalize_hyperliquid_candles(
            raw_frames["execution_5m_raw"],
            symbol,
            floor_to="5min",
        )
        execution_5m = execution_5m.loc[
            :,
            ["timestamp", "symbol", "open", "high", "low", "close", "volume"],
        ]
        save_dataframe(execution_5m, processed_dir / "execution_5m")


def ingest_hyperliquid_batch(config: IngestConfig) -> IngestionResult:
    symbol = (config.symbol or (config.symbols[0] if config.symbols else "")).upper()
    if not symbol:
        raise ValueError("ingest_hyperliquid_batch requires config.symbol or config.symbols[0]")
    start = _as_utc_timestamp(config.start)
    end = _as_utc_timestamp(config.end)
    raw_frames = {
        key: _slice_raw_frame(frame, start, end)
        for key, frame in fetch_hyperliquid_raw(
            config,
            symbol=symbol,
            start=start,
            end=end,
        ).items()
    }

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

    candles = normalize_hyperliquid_candles(raw_frames["candles_raw"], symbol).drop(
        columns="volume",
    )
    asset_context = normalize_hyperliquid_asset_context(raw_frames["asset_context_raw"], symbol)
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
    _save_execution_tables(raw_frames, symbol, processed_dir)

    validation_summary = validate_processed_directory(processed_dir)
    save_dataframe(
        validation_summary,
        processed_dir / "data_validation_summary",
        prefer_parquet=False,
    )
    return IngestionResult(
        symbol=symbol,
        start=start,
        end=end,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        validation_summary=validation_summary,
    )


def _iter_symbols(config: IngestConfig) -> list[str]:
    if config.symbols is not None:
        return [symbol.upper() for symbol in config.symbols]
    if config.symbol is None:
        raise ValueError("ingest config requires symbol or symbols")
    return [config.symbol.upper()]


def _iter_chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_size: str | None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if chunk_size is None:
        return [(start, end)]
    delta = pd.Timedelta(chunk_size)
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start
    while current < end:
        next_end = min(current + delta, end)
        chunks.append((current, next_end))
        current = next_end
    return chunks


def _make_bulk_summary_dir(base_dir: Path, source: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary_dir = base_dir / source / "_bulk" / timestamp
    summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir


def ingest_hyperliquid_bulk(config: IngestConfig) -> BulkIngestionResult:
    symbols = _iter_symbols(config)
    start = _as_utc_timestamp(config.start)
    end = _as_utc_timestamp(config.end)
    summary_rows: list[dict[str, object]] = []
    for symbol in symbols:
        for chunk_start, chunk_end in _iter_chunks(start, end, config.chunk_size):
            chunk_config = config.model_copy(deep=True)
            chunk_config.symbol = symbol
            chunk_config.symbols = [symbol]
            chunk_config.start = chunk_start.to_pydatetime()
            chunk_config.end = chunk_end.to_pydatetime()
            result = ingest_hyperliquid_batch(chunk_config)
            summary_rows.append(
                {
                    "symbol": symbol,
                    "start": chunk_start,
                    "end": chunk_end,
                    "raw_dir": str(result.raw_dir),
                    "processed_dir": str(result.processed_dir),
                    "datasets_validated": int(len(result.validation_summary)),
                    "has_execution_1m": bool(
                        (result.processed_dir / "execution_1m.parquet").exists()
                        or (result.processed_dir / "execution_1m.csv").exists()
                    ),
                    "has_execution_5m": bool(
                        (result.processed_dir / "execution_5m.parquet").exists()
                        or (result.processed_dir / "execution_5m.csv").exists()
                    ),
                },
            )
    summary_df = pd.DataFrame(summary_rows).sort_values(["symbol", "start"]).reset_index(drop=True)
    summary_dir = _make_bulk_summary_dir(config.processed_output_dir, config.source)
    save_dataframe(summary_df, summary_dir / "ingestion_summary", prefer_parquet=False)
    validation_rows: list[pd.DataFrame] = []
    for processed_dir in summary_df["processed_dir"]:
        validation = validate_processed_directory(Path(str(processed_dir))).copy()
        validation["processed_dir"] = str(processed_dir)
        validation_rows.append(validation)
    if validation_rows:
        validation_df = pd.concat(validation_rows, ignore_index=True)
        save_dataframe(validation_df, summary_dir / "data_validation_summary", prefer_parquet=False)
    return BulkIngestionResult(summary_dir=summary_dir, batch_summary=summary_df)
