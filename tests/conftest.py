from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hl_funding_carry.settings import CONFIG_DIR


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=30, freq="1h", tz="UTC")
    open_prices = [100.0 + 0.02 * index for index in range(30)]
    close_prices = [price + 0.05 for price in open_prices]
    mark_prices = [price + 0.20 for price in open_prices]
    oracle_prices = [price for price in open_prices]
    spot_prices = [price + 0.01 for price in open_prices]
    predicted = [0.00020] * 26 + [0.00005] * 4

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTC"] * 30,
            "open": open_prices,
            "high": [price + 0.15 for price in open_prices],
            "low": [price - 0.10 for price in open_prices],
            "close": close_prices,
            "mark_price": mark_prices,
            "oracle_price": oracle_prices,
            "current_funding": [0.00012] * 30,
            "pred_funding_1h": predicted,
            "open_interest": [1000.0 + 5.0 * index for index in range(30)],
            "spread_bps": [2.0] * 30,
            "spot_price": spot_prices,
        },
    )


@pytest.fixture
def sample_dataset_15m() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=24, freq="15min", tz="UTC")
    base_price = [100.0 + 0.03 * index for index in range(24)]
    predicted = [0.00020] * 8 + [0.00019] * 8 + [0.00004] * 8
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTC"] * 24,
            "open": base_price,
            "high": [price + 0.10 for price in base_price],
            "low": [price - 0.08 for price in base_price],
            "close": [price + 0.04 for price in base_price],
            "mark_price": [price + 0.18 for price in base_price],
            "oracle_price": base_price,
            "current_funding": [0.00012] * 24,
            "pred_funding_1h": predicted,
            "open_interest": [1000.0 + 10.0 * index for index in range(24)],
            "spread_bps": [2.0] * 24,
            "spot_price": [price + 0.01 for price in base_price],
        },
    )


@pytest.fixture
def config_path() -> Path:
    return CONFIG_DIR / "funding_carry.base.yaml"
