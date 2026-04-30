from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from hl_funding_carry.backtest.artifacts import (
    make_run_directory,
    save_backtest_artifacts,
    save_sweep_summary,
    save_walkforward_artifacts,
)
from hl_funding_carry.backtest.simulator import simulate_backtest
from hl_funding_carry.data.loaders import load_dataset_bundle
from hl_funding_carry.data.validation import validate_processed_directory
from hl_funding_carry.settings import FundingCarryConfig, SweepGridConfig
from hl_funding_carry.strategies.funding_carry import FundingCarryStrategy
from hl_funding_carry.types import BacktestResult


def _prepare_dataset_bundle(
    config: FundingCarryConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    dataset, execution_inputs = load_dataset_bundle(config.data)
    dataset = dataset[dataset["symbol"].isin(config.strategy.symbols)].reset_index(drop=True)
    return dataset, execution_inputs


def _run_backtest_on_dataset(
    config: FundingCarryConfig,
    dataset: pd.DataFrame,
    execution_inputs: dict[str, pd.DataFrame],
    run_id: str,
) -> BacktestResult:
    strategy = FundingCarryStrategy(config)
    features = strategy.build_features(dataset)
    targets = strategy.generate_target_positions(features)
    return simulate_backtest(
        targets,
        config,
        execution_inputs=execution_inputs,
        run_id=run_id,
    )


def run_backtest(
    config: FundingCarryConfig,
    output_dir: Path | None = None,
    run_id: str = "backtest",
    save_artifacts: bool = True,
) -> BacktestResult:
    dataset, execution_inputs = _prepare_dataset_bundle(config)
    result = _run_backtest_on_dataset(config, dataset, execution_inputs, run_id=run_id)
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
    execution_model: str,
    execution_slippage_bps: float,
    execution_fee_bps: float,
) -> FundingCarryConfig:
    updated = config.model_copy(deep=True)
    updated.strategy.entry.predicted_funding_min = pred_funding_entry
    updated.strategy.entry.basis_min = basis_entry
    updated.strategy.timing.entry_lead_minutes = entry_lead_minutes
    updated.strategy.timing.max_hold_minutes = max_hold_minutes
    updated.execution.model = execution_model  # type: ignore[assignment]
    updated.execution.slippage_bps = execution_slippage_bps
    updated.execution.fee_bps = execution_fee_bps
    return updated


def _sweep_combinations(
    grid: SweepGridConfig,
) -> list[tuple[float, float, int, int, str, float, float]]:
    return list(
        product(
            grid.pred_funding_entry,
            grid.basis_entry,
            grid.entry_lead_minutes,
            grid.max_hold_minutes,
            grid.execution_model,
            grid.execution_slippage_bps,
            grid.execution_fee_bps,
        ),
    )


def run_sweep(
    config: FundingCarryConfig,
    grid: SweepGridConfig,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    sweep_root_base = output_dir or config.artifacts.root_dir
    sweep_root = make_run_directory(sweep_root_base, "sweep")
    runs_dir = sweep_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    dataset, execution_inputs = _prepare_dataset_bundle(config)

    summaries: list[dict[str, Any]] = []
    combinations = _sweep_combinations(grid)
    for index, combination in enumerate(combinations, start=1):
        (
            pred_funding_entry,
            basis_entry,
            entry_lead_minutes,
            max_hold_minutes,
            execution_model,
            execution_slippage_bps,
            execution_fee_bps,
        ) = combination
        run_id = f"run_{index:03d}"
        run_config = _apply_sweep_params(
            config,
            pred_funding_entry=pred_funding_entry,
            basis_entry=basis_entry,
            entry_lead_minutes=entry_lead_minutes,
            max_hold_minutes=max_hold_minutes,
            execution_model=execution_model,
            execution_slippage_bps=execution_slippage_bps,
            execution_fee_bps=execution_fee_bps,
        )
        result = _run_backtest_on_dataset(
            run_config,
            dataset,
            execution_inputs,
            run_id=run_id,
        )
        run_dir = save_backtest_artifacts(result, run_config, runs_dir)
        result.artifact_dir = run_dir
        summaries.append(
            {
                **result.summary,
                "run_id": run_id,
                "pred_funding_entry": pred_funding_entry,
                "basis_entry": basis_entry,
                "entry_lead_minutes": entry_lead_minutes,
                "max_hold_minutes": max_hold_minutes,
                "execution_model": execution_model,
                "execution_slippage_bps": execution_slippage_bps,
                "execution_fee_bps": execution_fee_bps,
                "artifact_dir": str(run_dir),
            },
        )

    summary_df = pd.DataFrame(summaries).sort_values("run_id").reset_index(drop=True)
    save_sweep_summary(summary_df, sweep_root)
    return summary_df


def _slice_dataset(dataset: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (dataset["timestamp"] >= start) & (dataset["timestamp"] < end)
    return dataset.loc[mask].copy().reset_index(drop=True)


def _score_run(summary: dict[str, float | str], metric: str) -> float:
    value = summary.get(metric, 0.0)
    if isinstance(value, str):
        return 0.0
    return float(value)


def _build_fold_tables(
    fold_id: str,
    result: BacktestResult,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_attribution = result.portfolio_summary.copy()
    fold_attribution["fold_id"] = fold_id
    symbol_attribution = result.symbol_summary.copy()
    symbol_attribution["fold_id"] = fold_id
    execution_model_attribution = result.execution_summary.copy()
    execution_model_attribution["fold_id"] = fold_id
    return fold_attribution, symbol_attribution, execution_model_attribution


def run_walkforward(
    config: FundingCarryConfig,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    if config.walkforward is None or not config.walkforward.enabled:
        raise ValueError("walkforward config is required and must be enabled")
    if config.sweep_grid is None:
        raise ValueError("sweep_grid is required for walkforward runs")

    dataset, execution_inputs = _prepare_dataset_bundle(config)
    if dataset.empty:
        raise ValueError("walkforward dataset is empty")

    train_window = pd.Timedelta(config.walkforward.train_window)
    test_window = pd.Timedelta(config.walkforward.test_window)
    step_size = pd.Timedelta(config.walkforward.step_size)
    selection_metric = config.walkforward.selection_metric

    min_timestamp = pd.Timestamp(dataset["timestamp"].min())
    max_timestamp = pd.Timestamp(dataset["timestamp"].max())
    fold_root = make_run_directory(output_dir or config.artifacts.root_dir, "walkforward")
    folds_dir = fold_root / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    current_start = min_timestamp
    fold_index = 1
    selected_params: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    fold_attr_frames: list[pd.DataFrame] = []
    symbol_attr_frames: list[pd.DataFrame] = []
    execution_attr_frames: list[pd.DataFrame] = []

    while current_start + train_window + test_window <= max_timestamp + pd.Timedelta(seconds=1):
        train_start = current_start
        train_end = train_start + train_window
        test_start = train_end
        test_end = test_start + test_window

        train_dataset = _slice_dataset(dataset, train_start, train_end)
        test_dataset = _slice_dataset(dataset, test_start, test_end)
        if train_dataset.empty or test_dataset.empty:
            current_start += step_size
            continue

        best_config: FundingCarryConfig | None = None
        best_summary: dict[str, float | str] | None = None
        best_params: dict[str, Any] | None = None

        for combination in _sweep_combinations(config.sweep_grid):
            (
                pred_funding_entry,
                basis_entry,
                entry_lead_minutes,
                max_hold_minutes,
                execution_model,
                execution_slippage_bps,
                execution_fee_bps,
            ) = combination
            candidate_config = _apply_sweep_params(
                config,
                pred_funding_entry=pred_funding_entry,
                basis_entry=basis_entry,
                entry_lead_minutes=entry_lead_minutes,
                max_hold_minutes=max_hold_minutes,
                execution_model=execution_model,
                execution_slippage_bps=execution_slippage_bps,
                execution_fee_bps=execution_fee_bps,
            )
            train_result = _run_backtest_on_dataset(
                candidate_config,
                train_dataset,
                execution_inputs,
                run_id=f"fold_{fold_index:03d}_train",
            )
            candidate_score = _score_run(train_result.summary, selection_metric)
            best_score = _score_run(best_summary, selection_metric) if best_summary else 0.0
            if best_summary is None or candidate_score > best_score:
                best_config = candidate_config
                best_summary = train_result.summary
                best_params = {
                    "pred_funding_entry": pred_funding_entry,
                    "basis_entry": basis_entry,
                    "entry_lead_minutes": entry_lead_minutes,
                    "max_hold_minutes": max_hold_minutes,
                    "execution_model": execution_model,
                    "execution_slippage_bps": execution_slippage_bps,
                    "execution_fee_bps": execution_fee_bps,
                }

        if best_config is None or best_summary is None or best_params is None:
            current_start += step_size
            continue

        fold_id = f"fold_{fold_index:03d}"
        test_result = _run_backtest_on_dataset(
            best_config,
            test_dataset,
            execution_inputs,
            run_id=f"{fold_id}_test",
        )
        fold_run_dir = save_backtest_artifacts(test_result, best_config, folds_dir)
        test_result.artifact_dir = fold_run_dir
        selected_params.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                **best_params,
                "train_selection_metric": _score_run(best_summary, selection_metric),
            },
        )
        fold_results.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "artifact_dir": str(fold_run_dir),
                **best_params,
                **test_result.summary,
            },
        )
        fold_attr, symbol_attr, execution_attr = _build_fold_tables(fold_id, test_result)
        fold_attr_frames.append(fold_attr)
        symbol_attr_frames.append(symbol_attr)
        execution_attr_frames.append(execution_attr)
        current_start += step_size
        fold_index += 1

    fold_results_df = pd.DataFrame(fold_results)
    selected_params_df = pd.DataFrame(selected_params)
    walkforward_summary = pd.DataFrame(
        [
            {
                "fold_count": float(len(fold_results_df)),
                "average_total_return": float(fold_results_df["total_return"].mean())
                if not fold_results_df.empty
                else 0.0,
                "average_sharpe_like": float(fold_results_df["sharpe_like"].mean())
                if not fold_results_df.empty
                else 0.0,
                "selection_metric": selection_metric,
                "data_source": config.data.source,
                "symbol_count": float(dataset["symbol"].nunique()),
            },
        ],
    )
    validation_summary = (
        validate_processed_directory(config.data.processed_dir)
        if config.data.source == "processed_dir" and config.data.processed_dir is not None
        else pd.DataFrame()
    )
    fold_attribution_df = (
        pd.concat(fold_attr_frames, ignore_index=True) if fold_attr_frames else pd.DataFrame()
    )
    symbol_attribution_df = (
        pd.concat(symbol_attr_frames, ignore_index=True)
        if symbol_attr_frames
        else pd.DataFrame()
    )
    execution_model_attribution_df = (
        pd.concat(execution_attr_frames, ignore_index=True)
        if execution_attr_frames
        else pd.DataFrame()
    )
    save_walkforward_artifacts(
        walkforward_summary=walkforward_summary,
        fold_results=fold_results_df,
        selected_params=selected_params_df,
        validation_summary=validation_summary,
        fold_attribution=fold_attribution_df,
        symbol_attribution=symbol_attribution_df,
        execution_model_attribution=execution_model_attribution_df,
        output_dir=fold_root,
    )
    return fold_results_df
