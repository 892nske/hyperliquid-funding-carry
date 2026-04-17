from hl_funding_carry.features.funding import build_funding_features
from hl_funding_carry.strategies.funding_carry import FundingCarryStrategy


def test_strategy_emits_long_signal_when_conditions_are_met(funding_input):
    features = build_funding_features(funding_input)
    strategy = FundingCarryStrategy()
    signal = strategy.generate_signal(features)
    assert signal.iloc[-1] == 1
