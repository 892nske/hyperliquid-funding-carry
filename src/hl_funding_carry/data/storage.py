from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")


def make_dataset_directory(
    base_dir: Path,
    source_name: str,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Path:
    span = f"{start.strftime('%Y%m%dT%H%M%SZ')}_{end.strftime('%Y%m%dT%H%M%SZ')}"
    dataset_dir = base_dir / _safe_slug(source_name) / _safe_slug(symbol) / span
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def save_dataframe(
    df: pd.DataFrame,
    path_without_suffix: Path,
    prefer_parquet: bool = True,
) -> Path:
    if prefer_parquet:
        output_path = path_without_suffix.with_suffix(".parquet")
        df.to_parquet(output_path, index=False)
        return output_path
    output_path = path_without_suffix.with_suffix(".csv")
    df.to_csv(output_path, index=False)
    return output_path
