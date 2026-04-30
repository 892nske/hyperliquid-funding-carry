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


def _discover_processed_leaf_dirs(processed_dir: Path) -> list[Path]:
    if any(
        (processed_dir / f"candles{suffix}").exists()
        for suffix in (".parquet", ".csv", ".json")
    ):
        return [processed_dir]
    leaves = sorted({path.parent for path in processed_dir.rglob("candles.*")})
    if not leaves:
        raise FileNotFoundError(f"No processed dataset leaves found under {processed_dir}")
    return leaves


def _load_optional_execution(leaf_dir: Path, stem: str) -> pd.DataFrame | None:
    try:
        return load_execution_bars(_resolve_existing_table(leaf_dir, stem))
    except FileNotFoundError:
        return None


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


def load_processed_research_dataset(
    processed_dir: Path,
    recursive: bool = False,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    if recursive:
        frames = [
            load_processed_research_dataset(leaf_dir, recursive=False, symbols=symbols)
            for leaf_dir in _discover_processed_leaf_dirs(processed_dir)
        ]
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
        merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        return merged

    candles = load_candles(_resolve_existing_table(processed_dir, "candles"))
    asset_context = load_asset_context(_resolve_existing_table(processed_dir, "asset_context"))
    funding = load_funding_inputs(_resolve_existing_table(processed_dir, "funding_inputs"))
    merged = candles.merge(asset_context, on=["timestamp", "symbol"], how="inner")
    merged = merged.merge(funding, on=["timestamp", "symbol"], how="left")
    if symbols is not None:
        merged = merged[merged["symbol"].isin([symbol.upper() for symbol in symbols])]
    _validate_columns(merged, RESEARCH_COLUMNS, "processed_research_dataset")
    return merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def load_processed_execution_inputs(
    processed_dir: Path,
    recursive: bool = False,
) -> dict[str, pd.DataFrame]:
    if recursive:
        buckets: dict[str, list[pd.DataFrame]] = {"5m": [], "1m": []}
        for leaf_dir in _discover_processed_leaf_dirs(processed_dir):
            for key, stem in (("5m", "execution_5m"), ("1m", "execution_1m")):
                frame = _load_optional_execution(leaf_dir, stem)
                if frame is not None:
                    buckets[key].append(frame)
        recursive_execution_inputs: dict[str, pd.DataFrame] = {}
        for key, frames in buckets.items():
            if frames:
                recursive_execution_inputs[key] = (
                    pd.concat(frames, ignore_index=True)
                    .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
                    .sort_values(["timestamp", "symbol"])
                    .reset_index(drop=True)
                )
        return recursive_execution_inputs

    execution_inputs: dict[str, pd.DataFrame] = {}
    for key, stem in (("5m", "execution_5m"), ("1m", "execution_1m")):
        frame = _load_optional_execution(processed_dir, stem)
        if frame is not None:
            execution_inputs[key] = frame
    return execution_inputs


def load_dataset_bundle(data_config: DataConfig) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if data_config.source == "processed_dir":
        if data_config.processed_dir is None:
            raise ValueError("data.processed_dir is required when data.source=processed_dir")
        recursive = data_config.processed_recursive or not any(
            (data_config.processed_dir / f"candles{suffix}").exists()
            for suffix in (".parquet", ".csv", ".json")
        )
        dataset = load_processed_research_dataset(data_config.processed_dir, recursive=recursive)
        execution_inputs = load_execution_inputs(
            data_config.execution_5m_path,
            data_config.execution_1m_path,
        )
        if not execution_inputs:
            execution_inputs = load_processed_execution_inputs(
                data_config.processed_dir,
                recursive=recursive,
            )
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


def load_processed_dataset_tables(
    processed_dir: Path,
    recursive: bool = False,
) -> dict[str, pd.DataFrame]:
    if recursive:
        leaf_dirs = _discover_processed_leaf_dirs(processed_dir)
        recursive_tables: dict[str, list[pd.DataFrame]] = {
            "candles": [],
            "asset_context": [],
            "funding_inputs": [],
            "funding_history": [],
            "execution_5m": [],
            "execution_1m": [],
        }
        for leaf_dir in leaf_dirs:
            child_tables = load_processed_dataset_tables(leaf_dir, recursive=False)
            for key, frame in child_tables.items():
                recursive_tables.setdefault(key, []).append(frame)
        merged_tables: dict[str, pd.DataFrame] = {}
        for key, frames in recursive_tables.items():
            if frames:
                merged_tables[key] = (
                    pd.concat(frames, ignore_index=True)
                    .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
                    .sort_values(["timestamp", "symbol"])
                    .reset_index(drop=True)
                )
        return merged_tables

    tables: dict[str, pd.DataFrame] = {
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
        optional_frame: pd.DataFrame | None = _load_optional_execution(processed_dir, stem)
        if optional_frame is not None:
            tables[key] = optional_frame
    return tables
