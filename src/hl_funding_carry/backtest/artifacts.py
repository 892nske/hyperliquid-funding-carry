from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from hl_funding_carry.backtest.attribution import build_attribution_tables
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
    _save_dataframe(result.pnl_attribution, run_dir / "pnl_attribution.csv")
    _save_dataframe(result.execution_summary, run_dir / "execution_summary.csv")
    _save_dataframe(result.trade_attribution, run_dir / "trade_attribution.csv")
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


def save_validation_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_dataframe(summary_df, output_dir / "data_validation_summary.csv")


def save_walkforward_artifacts(
    walkforward_summary: pd.DataFrame,
    fold_results: pd.DataFrame,
    selected_params: pd.DataFrame,
    validation_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_dataframe(walkforward_summary, output_dir / "walkforward_summary.csv")
    _save_dataframe(fold_results, output_dir / "fold_results.csv")
    _save_dataframe(selected_params, output_dir / "selected_params.csv")
    _save_dataframe(validation_summary, output_dir / "data_validation_summary.csv")


def regenerate_report(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ledger = pd.read_csv(input_dir / "ledger.csv", parse_dates=["timestamp"])
    trades = pd.read_csv(
        input_dir / "trades.csv",
        parse_dates=["entry_time", "exit_time", "funding_event_time"],
    )
    run_id = "report"
    if "run_id" in ledger.columns and not ledger.empty:
        run_id = str(ledger["run_id"].iloc[0])
    pnl_attribution, execution_summary, trade_attribution = build_attribution_tables(
        run_id,
        ledger,
        trades,
    )
    _save_dataframe(pnl_attribution, input_dir / "pnl_attribution.csv")
    _save_dataframe(execution_summary, input_dir / "execution_summary.csv")
    _save_dataframe(trade_attribution, input_dir / "trade_attribution.csv")
    return pnl_attribution, execution_summary, trade_attribution
