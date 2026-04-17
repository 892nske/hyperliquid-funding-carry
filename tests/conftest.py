import pandas as pd
import pytest


@pytest.fixture
def funding_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=30, freq="1h", tz="UTC"),
            "symbol": ["BTC"] * 30,
            "current_funding": [0.00012] * 30,
            "predicted_funding": [0.00018] * 30,
            "mark_price": [100.2] * 30,
            "oracle_price": [100.0] * 30,
            "open_interest": [1000 + i for i in range(30)],
            "spread_bps": [2.0] * 30,
        }
    )
