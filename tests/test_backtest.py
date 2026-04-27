from __future__ import annotations

from hl_funding_carry.backtest.events import add_funding_event_calendar
from hl_funding_carry.backtest.simulator import simulate_backtest
from hl_funding_carry.settings import load_config
from hl_funding_carry.strategies.funding_carry import FundingCarryStrategy


def test_funding_pnl_only_occurs_at_funding_events(sample_dataset_15m, config_path):
    config = load_config(config_path)
    strategy = FundingCarryStrategy(config)
    features = strategy.build_features(sample_dataset_15m)
    targets = strategy.generate_target_positions(features)
    result = simulate_backtest(targets, config)

    funding_rows = result.ledger[result.ledger["funding_pnl"] != 0.0]
    assert not funding_rows.empty
    assert funding_rows["is_funding_event"].eq(1).all()


def test_two_leg_total_pnl_identity(sample_dataset_15m, config_path):
    config = load_config(config_path)
    strategy = FundingCarryStrategy(config)
    features = strategy.build_features(sample_dataset_15m)
    targets = strategy.generate_target_positions(features)
    result = simulate_backtest(targets, config)
    ledger = result.ledger

    reconstructed = (
        ledger["price_pnl_spot"]
        + ledger["price_pnl_perp"]
        + ledger["funding_pnl"]
        - ledger["fee"]
        - ledger["slippage"]
    )
    assert reconstructed.equals(ledger["total_pnl"])


def test_event_calendar_marks_hourly_events(sample_dataset_15m):
    calendar = add_funding_event_calendar(sample_dataset_15m, funding_interval_minutes=60)
    funding_event_minutes = calendar.loc[calendar["is_funding_event"] == 1, "timestamp"].dt.minute
    assert funding_event_minutes.eq(0).all()
