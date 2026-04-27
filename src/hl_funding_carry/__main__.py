from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from hl_funding_carry.backtest.artifacts import regenerate_report, save_validation_summary
from hl_funding_carry.data.ingestion import ingest_hyperliquid_batch
from hl_funding_carry.data.validation import validate_processed_directory
from hl_funding_carry.experiments.runner import run_backtest, run_sweep, run_walkforward
from hl_funding_carry.settings import (
    CONFIG_DIR,
    IngestConfig,
    load_config,
    load_ingest_config,
    load_sweep_grid,
)


def _build_backtest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a funding carry backtest.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "funding_carry.base.yaml",
        help="Path to the funding carry config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional artifact output directory.",
    )
    return parser


def _build_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a funding carry parameter sweep.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "funding_carry.base.yaml",
        help="Path to the funding carry config file.",
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=CONFIG_DIR / "funding_carry.sweep.yaml",
        help="Path to the sweep grid YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional artifact output directory.",
    )
    return parser


def _build_report_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate attribution reports from an artifact dir.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Artifact run directory containing ledger.csv and trades.csv.",
    )
    return parser


def _build_ingest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Hyperliquid research data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional ingest config file.",
    )
    parser.add_argument("--symbol", type=str, default=None, help="Symbol to ingest, e.g. BTC.")
    parser.add_argument("--start", type=str, default=None, help="UTC start timestamp.")
    parser.add_argument("--end", type=str, default=None, help="UTC end timestamp.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional raw output directory override.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Optional processed output directory override.",
    )
    return parser


def _build_validate_data_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a processed research dataset directory.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Processed dataset directory containing candles/asset_context/funding_inputs files.",
    )
    return parser


def _build_walkforward_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a funding carry walk-forward study.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "funding_carry.walkforward.yaml",
        help="Path to the walk-forward config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional artifact output directory.",
    )
    return parser


def _resolve_ingest_config(
    config_path: Path | None,
    symbol: str | None,
    start: str | None,
    end: str | None,
    raw_output_dir: Path | None,
    processed_output_dir: Path | None,
) -> IngestConfig:
    ingest_config = (
        load_ingest_config(config_path)
        if config_path is not None
        else IngestConfig(
            symbol=symbol or "BTC",
            start=datetime.fromisoformat(
                (start or "2026-01-01T00:00:00+00:00").replace("Z", "+00:00"),
            ),
            end=datetime.fromisoformat(
                (end or "2026-01-02T00:00:00+00:00").replace("Z", "+00:00"),
            ),
        )
    )
    if symbol is not None:
        ingest_config.symbol = symbol
    if start is not None:
        ingest_config.start = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if end is not None:
        ingest_config.end = datetime.fromisoformat(end.replace("Z", "+00:00"))
    if raw_output_dir is not None:
        ingest_config.raw_output_dir = raw_output_dir
    if processed_output_dir is not None:
        ingest_config.processed_output_dir = processed_output_dir
    return ingest_config


def main(argv: Sequence[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    command = "backtest"
    commands = {"backtest", "sweep", "report", "ingest", "validate-data", "walkforward"}
    if args and args[0] in commands:
        command = args.pop(0)

    if command == "sweep":
        parser = _build_sweep_parser()
        parsed = parser.parse_args(args)
        funding_config = load_config(parsed.config)
        grid = load_sweep_grid(parsed.grid)
        summary_df = run_sweep(funding_config, grid, output_dir=parsed.output_dir)
        print("Funding Carry sweep summary")
        print(summary_df.to_string(index=False))
        return

    if command == "report":
        parser = _build_report_parser()
        parsed = parser.parse_args(args)
        _, execution_summary, _ = regenerate_report(parsed.input_dir)
        print("Funding Carry attribution report regenerated")
        print(execution_summary.to_string(index=False))
        return

    if command == "ingest":
        parser = _build_ingest_parser()
        parsed = parser.parse_args(args)
        ingest_config = _resolve_ingest_config(
            parsed.config,
            parsed.symbol,
            parsed.start,
            parsed.end,
            parsed.output_dir,
            parsed.processed_dir,
        )
        ingestion_result = ingest_hyperliquid_batch(ingest_config)
        print("Funding Carry ingest completed")
        print(f"raw_dir: {ingestion_result.raw_dir}")
        print(f"processed_dir: {ingestion_result.processed_dir}")
        print(ingestion_result.validation_summary.to_string(index=False))
        return

    if command == "validate-data":
        parser = _build_validate_data_parser()
        parsed = parser.parse_args(args)
        summary_df = validate_processed_directory(parsed.input_dir)
        save_validation_summary(summary_df, parsed.input_dir)
        print("Funding Carry data validation summary")
        print(summary_df.to_string(index=False))
        return

    if command == "walkforward":
        parser = _build_walkforward_parser()
        parsed = parser.parse_args(args)
        funding_config = load_config(parsed.config)
        fold_results = run_walkforward(funding_config, output_dir=parsed.output_dir)
        print("Funding Carry walk-forward summary")
        print(fold_results.to_string(index=False))
        return

    parser = _build_backtest_parser()
    parsed = parser.parse_args(args)
    funding_config = load_config(parsed.config)
    result = run_backtest(funding_config, output_dir=parsed.output_dir, run_id="backtest")
    print("Funding Carry backtest summary")
    for key, value in result.summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    if result.artifact_dir is not None:
        print(f"artifact_dir: {result.artifact_dir}")


if __name__ == "__main__":
    main()
