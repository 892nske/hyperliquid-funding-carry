from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hl_funding_carry.settings import DataConfig
from hl_funding_carry.types import (
    ASSET_CONTEXT_COLUMNS,
    CANDLE_COLUMNS,
    FUNDING_COLUMNS,
    FUNDING_HISTORY_COLUMNS,
    INTRABAR_COLUMNS,
    RESEARCH_COLUMNS,
)


def _load_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return pd.DataFrame(payload["data"])
        return pd.DataFrame([payload])
    raise ValueError(f"Unsupported JSON payload in {path}")


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".json":
        return _load_json(path)
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


def _resolve_existing_table(base_dir: Path, stem: str) -> Path:
    for suffix in (".parquet", ".csv", ".json"):
        candidate = base_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {stem}.parquet/csv/json in {base_dir}")


def load_candles(path: Path) -> pd.DataFrame:
    return _normalize_timeseries(_load_table(path), CANDLE_COLUMNS, "candles")


def load_asset_context(path: Path) -> pd.DataFrame:
    return _normalize_timeseries(_load_table(path), ASSET_CONTEXT_COLUMNS, "asset_context")


def load_funding_inputs(path: Path) -> pd.DataFrame:
    return _normalize_timeseries(_load_table(path), FUNDING_COLUMNS, "funding_inputs")


def load_funding_history(path: Path) -> pd.DataFrame:
    return _normalize_timeseries(_load_table(path), FUNDING_HISTORY_COLUMNS, "funding_history")


def load_execution_bars(path: Path) -> pd.DataFrame:
    return _normalize_timeseries(_load_table(path), INTRABAR_COLUMNS, "execution_bars")


def load_execution_inputs(
    execution_5m_path: Path | None,
    execution_1m_path: Path | None,
) -> dict[str, pd.DataFrame]:
    execution_inputs: dict[str, pd.DataFrame] = {}
    if execution_5m_path is not None and execution_5m_path.exists():
        execution_inputs["5m"] = load_execution_bars(execution_5m_path)
    if execution_1m_path is not None and execution_1m_path.exists():
        execution_inputs["1m"] = load_execution_bars(execution_1m_path)
    return execution_inputs


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


def load_processed_research_dataset(processed_dir: Path) -> pd.DataFrame:
    candles = load_candles(_resolve_existing_table(processed_dir, "candles"))
    asset_context = load_asset_context(_resolve_existing_table(processed_dir, "asset_context"))
    funding = load_funding_inputs(_resolve_existing_table(processed_dir, "funding_inputs"))
    merged = candles.merge(asset_context, on=["timestamp", "symbol"], how="inner")
    merged = merged.merge(funding, on=["timestamp", "symbol"], how="left")
    _validate_columns(merged, RESEARCH_COLUMNS, "processed_research_dataset")
    return merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def load_dataset_bundle(data_config: DataConfig) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if data_config.source == "processed_dir":
        if data_config.processed_dir is None:
            raise ValueError("data.processed_dir is required when data.source=processed_dir")
        dataset = load_processed_research_dataset(data_config.processed_dir)
        execution_inputs = load_execution_inputs(
            data_config.execution_5m_path,
            data_config.execution_1m_path,
        )
        if not execution_inputs:
            try:
                execution_inputs = load_execution_inputs(
                    _resolve_existing_table(data_config.processed_dir, "execution_5m"),
                    _resolve_existing_table(data_config.processed_dir, "execution_1m"),
                )
            except FileNotFoundError:
                execution_inputs = {}
        return dataset, execution_inputs

    return (
        load_research_dataset(
            candles_path=data_config.candles_path,
            asset_context_path=data_config.asset_context_path,
            funding_path=data_config.funding_path,
        ),
        load_execution_inputs(
            execution_5m_path=data_config.execution_5m_path,
            execution_1m_path=data_config.execution_1m_path,
        ),
    )


def load_processed_dataset_tables(processed_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {
        "candles": load_candles(_resolve_existing_table(processed_dir, "candles")),
        "asset_context": load_asset_context(
            _resolve_existing_table(processed_dir, "asset_context"),
        ),
        "funding_inputs": load_funding_inputs(
            _resolve_existing_table(processed_dir, "funding_inputs"),
        ),
    }
    try:
        tables["funding_history"] = load_funding_history(
            _resolve_existing_table(processed_dir, "funding_history"),
        )
    except FileNotFoundError:
        pass
    for key, stem in (("execution_5m", "execution_5m"), ("execution_1m", "execution_1m")):
        try:
            tables[key] = load_execution_bars(_resolve_existing_table(processed_dir, stem))
        except FileNotFoundError:
            continue
    return tables
