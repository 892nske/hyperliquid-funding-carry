from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from hl_funding_carry.settings import FundingCarryConfig
from hl_funding_carry.types import BacktestResult


def make_run_directory(base_dir: Path, run_id: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_dir / f"{timestamp}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_dataframe(df: pd.DataFrame, csv_path: Path, parquet_path: Path | None = None) -> None:
    df.to_csv(csv_path, index=False)
    if parquet_path is not None:
        df.to_parquet(parquet_path, index=False)


def save_backtest_artifacts(
    result: BacktestResult,
    config: FundingCarryConfig,
    output_dir: Path,
) -> Path:
    run_dir = make_run_directory(output_dir, result.run_id)
    summary_df = pd.DataFrame([result.summary])
    _save_dataframe(summary_df, run_dir / "summary.csv")
    _save_dataframe(result.equity_curve, run_dir / "equity_curve.csv")
    _save_dataframe(result.trades, run_dir / "trades.csv")
    _save_dataframe(result.ledger, run_dir / "ledger.csv")
    with (run_dir / "params.json").open("w", encoding="utf-8") as handle:
        json.dump(config.model_dump(mode="json"), handle, indent=2)
    return run_dir


def save_sweep_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_dataframe(
        summary_df,
        output_dir / "summary.csv",
        output_dir / "summary.parquet",
    )
