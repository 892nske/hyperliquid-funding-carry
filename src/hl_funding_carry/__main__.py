from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from hl_funding_carry.experiments.runner import run_backtest, run_sweep
from hl_funding_carry.settings import CONFIG_DIR, load_config, load_sweep_grid


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


def main(argv: Sequence[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    command = "backtest"
    if args and args[0] in {"backtest", "sweep"}:
        command = args.pop(0)

    if command == "sweep":
        parser = _build_sweep_parser()
        parsed = parser.parse_args(args)
        config = load_config(parsed.config)
        grid = load_sweep_grid(parsed.grid)
        summary_df = run_sweep(config, grid, output_dir=parsed.output_dir)
        print("Funding Carry sweep summary")
        print(summary_df.to_string(index=False))
        return

    parser = _build_backtest_parser()
    parsed = parser.parse_args(args)
    config = load_config(parsed.config)
    result = run_backtest(config, output_dir=parsed.output_dir, run_id="backtest")
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
