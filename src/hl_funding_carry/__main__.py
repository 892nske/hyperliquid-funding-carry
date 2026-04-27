from __future__ import annotations

import argparse
from pathlib import Path

from hl_funding_carry.backtest.simulator import simulate_backtest
from hl_funding_carry.data.loaders import load_research_dataset
from hl_funding_carry.settings import CONFIG_DIR, load_config
from hl_funding_carry.strategies.funding_carry import FundingCarryStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal funding carry backtest.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "funding_carry.base.yaml",
        help="Path to the funding carry config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = load_research_dataset(
        candles_path=config.data.candles_path,
        asset_context_path=config.data.asset_context_path,
        funding_path=config.data.funding_path,
    )
    dataset = dataset[dataset["symbol"].isin(config.strategy.symbols)].reset_index(drop=True)

    strategy = FundingCarryStrategy(config)
    features = strategy.build_features(dataset)
    targets = strategy.generate_target_positions(features)
    _, summary = simulate_backtest(targets, config)

    print("Funding Carry backtest summary")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
