from __future__ import annotations

from hl_funding_carry.backtest.simulator import simulate_backtest
from hl_funding_carry.data.loaders import load_research_dataset
from hl_funding_carry.settings import load_config
from hl_funding_carry.strategies.funding_carry import FundingCarryStrategy


def test_load_config_reads_base_yaml(config_path):
    config = load_config(config_path)
    assert config.strategy.name == "funding_carry"
    assert config.strategy.signal_timeframe == "1h"
    assert config.execution.slippage_bps == 1.0


def test_strategy_emits_long_signal_when_conditions_are_met(sample_dataset, config_path):
    config = load_config(config_path)
    strategy = FundingCarryStrategy(config)
    features = strategy.build_features(sample_dataset)
    signal_df = strategy.generate_signal(features)
    assert signal_df.iloc[-5]["entry_long_spot_short_perp"] == 1


def test_backtest_separates_funding_and_total_pnl(sample_dataset, config_path):
    config = load_config(config_path)
    strategy = FundingCarryStrategy(config)
    features = strategy.build_features(sample_dataset)
    targets = strategy.generate_target_positions(features)
    backtest_df, summary = simulate_backtest(targets, config)

    assert "funding_pnl" in backtest_df.columns
    assert "price_pnl" in backtest_df.columns
    assert "total_pnl" in backtest_df.columns
    first_active = backtest_df[backtest_df["position_pair"] != 0.0].iloc[0]
    assert first_active["funding_pnl"] > 0.0
    assert summary["trade_count"] >= 1.0


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
    backtest_df, summary = simulate_backtest(targets, config)

    assert not backtest_df.empty
    assert set(summary) == {
        "total_return",
        "sharpe_like",
        "max_drawdown",
        "trade_count",
        "average_holding_period",
    }
