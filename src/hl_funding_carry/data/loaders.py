from __future__ import annotations

from pathlib import Path

import pandas as pd

from hl_funding_carry.types import (
    ASSET_CONTEXT_COLUMNS,
    CANDLE_COLUMNS,
    FUNDING_COLUMNS,
    RESEARCH_COLUMNS,
)


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def _validate_columns(
    df: pd.DataFrame,
    required_columns: tuple[str, ...],
    dataset_name: str,
) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def _normalize_timeseries(
    df: pd.DataFrame,
    required_columns: tuple[str, ...],
    dataset_name: str,
) -> pd.DataFrame:
    _validate_columns(df, required_columns, dataset_name)
    normalized = df.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    normalized["symbol"] = normalized["symbol"].astype(str).str.upper().str.strip()
    normalized = normalized.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    normalized = normalized.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return normalized


def load_candles(path: Path) -> pd.DataFrame:
    return _normalize_timeseries(_load_table(path), CANDLE_COLUMNS, "candles")


def load_asset_context(path: Path) -> pd.DataFrame:
    return _normalize_timeseries(_load_table(path), ASSET_CONTEXT_COLUMNS, "asset_context")


def load_funding_inputs(path: Path) -> pd.DataFrame:
    return _normalize_timeseries(_load_table(path), FUNDING_COLUMNS, "funding_inputs")


def load_research_dataset(
    candles_path: Path,
    asset_context_path: Path,
    funding_path: Path,
) -> pd.DataFrame:
    candles = load_candles(candles_path)
    asset_context = load_asset_context(asset_context_path)
    funding = load_funding_inputs(funding_path)
    merged = candles.merge(asset_context, on=["timestamp", "symbol"], how="inner")
    merged = merged.merge(funding, on=["timestamp", "symbol"], how="left")
    _validate_columns(merged, RESEARCH_COLUMNS, "research_dataset")
    return merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
