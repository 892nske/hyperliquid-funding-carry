from __future__ import annotations

from hl_funding_carry.data.loaders import load_research_dataset
from hl_funding_carry.experiments.runner import run_backtest
from hl_funding_carry.settings import load_config
from hl_funding_carry.strategies.funding_carry import FundingCarryStrategy


def test_load_config_reads_base_yaml(config_path):
    config = load_config(config_path)
    assert config.strategy.name == "funding_carry"
    assert config.strategy.signal_timeframe == "1h"
    assert config.strategy.timing.entry_lead_minutes == 60
    assert config.execution.slippage_bps == 1.0


def test_strategy_emits_long_signal_when_conditions_are_met(sample_dataset, config_path):
    config = load_config(config_path)
    strategy = FundingCarryStrategy(config)
    features = strategy.build_features(sample_dataset)
    signal_df = strategy.generate_signal(features)
    assert signal_df["entry_long_spot_short_perp"].sum() >= 1


def test_entry_exit_timing_uses_funding_windows(sample_dataset_15m, config_path):
    config = load_config(config_path)
    config.strategy.timing.entry_lead_minutes = 15
    config.strategy.timing.min_hold_minutes_after_funding = 15
    config.strategy.timing.max_hold_minutes = 90
    strategy = FundingCarryStrategy(config)
    features = strategy.build_features(sample_dataset_15m)
    targets = strategy.generate_target_positions(features)

    entry_rows = targets[targets["signal"] == "long_spot_short_perp"]
    assert not entry_rows.empty
    assert entry_rows["execution_minutes_to_funding"].eq(15.0).all()

    exit_rows = targets[targets["reason_code"].str.startswith("exit_")]
    assert not exit_rows.empty
    assert exit_rows["timestamp"].ge(exit_rows["min_exit_time"]).all()


def test_end_to_end_backtest_runs_on_sample_data(config_path):
    config = load_config(config_path)
    dataset = load_research_dataset(
        candles_path=config.data.candles_path,
        asset_context_path=config.data.asset_context_path,
        funding_path=config.data.funding_path,
    )
    strategy = FundingCarryStrategy(config)
    features = strategy.build_features(dataset)
    targets = strategy.generate_target_positions(features)

    assert not targets.empty
    assert {"next_funding_time", "is_funding_event", "target_position"}.issubset(targets.columns)

    result = run_backtest(config, save_artifacts=False)
    assert set(result.summary) == {
        "total_return",
        "sharpe_like",
        "max_drawdown",
        "trade_count",
        "average_holding_period",
        "run_id",
    }
