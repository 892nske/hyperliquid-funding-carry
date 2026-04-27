from __future__ import annotations

from pathlib import Path

import pandas as pd

from hl_funding_carry.data.loaders import load_processed_dataset_tables
from hl_funding_carry.types import ValidationReport


def summarize_validation_report(df: pd.DataFrame, dataset_name: str) -> ValidationReport:
    if df.empty:
        return ValidationReport(
            dataset=dataset_name,
            row_count=0,
            missing_ratio=0.0,
            duplicate_count=0,
            non_monotonic_count=0,
            gap_count=0,
            max_gap_minutes=0.0,
            min_timestamp=None,
            max_timestamp=None,
        )

    ordered = df.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
    ordered = ordered.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    duplicate_count = int(ordered.duplicated(subset=["timestamp", "symbol"]).sum())
    missing_ratio = float(ordered.isna().mean().mean())
    non_monotonic_count = 0
    gap_count = 0
    max_gap_minutes = 0.0

    for _, symbol_df in ordered.groupby("symbol", sort=True):
        diffs = symbol_df["timestamp"].diff()
        non_monotonic_count += int(diffs.dropna().le(pd.Timedelta(0)).sum())
        if len(symbol_df) >= 3:
            expected_gap = diffs.dropna().median()
            if isinstance(expected_gap, pd.Timedelta) and expected_gap > pd.Timedelta(0):
                large_gaps = diffs.dropna()[diffs.dropna() > expected_gap]
                gap_count += int(len(large_gaps))
                if not large_gaps.empty:
                    max_gap_minutes = max(
                        max_gap_minutes,
                        float(large_gaps.max().total_seconds() / 60.0),
                    )

    return ValidationReport(
        dataset=dataset_name,
        row_count=int(len(ordered)),
        missing_ratio=missing_ratio,
        duplicate_count=duplicate_count,
        non_monotonic_count=non_monotonic_count,
        gap_count=gap_count,
        max_gap_minutes=max_gap_minutes,
        min_timestamp=ordered["timestamp"].min(),
        max_timestamp=ordered["timestamp"].max(),
    )


def validation_report_to_frame(report: ValidationReport) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": report.dataset,
                "row_count": report.row_count,
                "missing_ratio": report.missing_ratio,
                "duplicate_count": report.duplicate_count,
                "non_monotonic_count": report.non_monotonic_count,
                "gap_count": report.gap_count,
                "max_gap_minutes": report.max_gap_minutes,
                "min_timestamp": report.min_timestamp,
                "max_timestamp": report.max_timestamp,
            },
        ],
    )


def validate_processed_directory(processed_dir: Path) -> pd.DataFrame:
    reports = [
        validation_report_to_frame(summarize_validation_report(frame, dataset_name))
        for dataset_name, frame in load_processed_dataset_tables(processed_dir).items()
    ]
    if not reports:
        return pd.DataFrame(
            columns=[
                "dataset",
                "row_count",
                "missing_ratio",
                "duplicate_count",
                "non_monotonic_count",
                "gap_count",
                "max_gap_minutes",
                "min_timestamp",
                "max_timestamp",
            ],
        )
    return pd.concat(reports, ignore_index=True)
