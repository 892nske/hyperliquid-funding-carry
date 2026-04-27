from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from hl_funding_carry.backtest.artifacts import (
    make_run_directory,
    save_backtest_artifacts,
    save_sweep_summary,
)
from hl_funding_carry.backtest.simulator import simulate_backtest
from hl_funding_carry.data.loaders import load_research_dataset
from hl_funding_carry.settings import FundingCarryConfig, SweepGridConfig
from hl_funding_carry.strategies.funding_carry import FundingCarryStrategy
from hl_funding_carry.types import BacktestResult


def run_backtest(
    config: FundingCarryConfig,
    output_dir: Path | None = None,
    run_id: str = "backtest",
    save_artifacts: bool = True,
) -> BacktestResult:
    dataset = load_research_dataset(
        candles_path=config.data.candles_path,
        asset_context_path=config.data.asset_context_path,
        funding_path=config.data.funding_path,
    )
    dataset = dataset[dataset["symbol"].isin(config.strategy.symbols)].reset_index(drop=True)
    strategy = FundingCarryStrategy(config)
    features = strategy.build_features(dataset)
    targets = strategy.generate_target_positions(features)
    result = simulate_backtest(targets, config, run_id=run_id)
    if save_artifacts:
        artifact_root = output_dir or config.artifacts.root_dir
        run_dir = save_backtest_artifacts(result, config, artifact_root)
        result.artifact_dir = run_dir
    return result


def _apply_sweep_params(
    config: FundingCarryConfig,
    pred_funding_entry: float,
    basis_entry: float,
    entry_lead_minutes: int,
    max_hold_minutes: int,
) -> FundingCarryConfig:
    updated = config.model_copy(deep=True)
    updated.strategy.entry.predicted_funding_min = pred_funding_entry
    updated.strategy.entry.basis_min = basis_entry
    updated.strategy.timing.entry_lead_minutes = entry_lead_minutes
    updated.strategy.timing.max_hold_minutes = max_hold_minutes
    return updated


def run_sweep(
    config: FundingCarryConfig,
    grid: SweepGridConfig,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    sweep_root_base = output_dir or config.artifacts.root_dir
    sweep_root = make_run_directory(sweep_root_base, "sweep")
    runs_dir = sweep_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    combinations = product(
        grid.pred_funding_entry,
        grid.basis_entry,
        grid.entry_lead_minutes,
        grid.max_hold_minutes,
    )
    for index, combination in enumerate(combinations, start=1):
        pred_funding_entry, basis_entry, entry_lead_minutes, max_hold_minutes = combination
        run_id = f"run_{index:03d}"
        run_config = _apply_sweep_params(
            config,
            pred_funding_entry=pred_funding_entry,
            basis_entry=basis_entry,
            entry_lead_minutes=entry_lead_minutes,
            max_hold_minutes=max_hold_minutes,
        )
        result = run_backtest(
            run_config,
            output_dir=runs_dir,
            run_id=run_id,
            save_artifacts=True,
        )
        summaries.append(
            {
                **result.summary,
                "run_id": run_id,
                "pred_funding_entry": pred_funding_entry,
                "basis_entry": basis_entry,
                "entry_lead_minutes": entry_lead_minutes,
                "max_hold_minutes": max_hold_minutes,
                "artifact_dir": str(result.artifact_dir) if result.artifact_dir else "",
            },
        )

    summary_df = pd.DataFrame(summaries).sort_values("run_id").reset_index(drop=True)
    save_sweep_summary(summary_df, sweep_root)
    return summary_df
