from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FundingCarryParams:
    predicted_funding_min: float = 0.00015
    current_funding_min: float = 0.00010
    basis_min: float = 0.0010
    oi_change_1h_min: float = 0.0
    spread_bps_max: float = 4.0


class FundingCarryStrategy:
    def __init__(self, params: FundingCarryParams | None = None) -> None:
        self.params = params or FundingCarryParams()

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        signal = (
            (df["predicted_funding"] >= p.predicted_funding_min)
            & (df["current_funding"] >= p.current_funding_min)
            & (df["basis"] >= p.basis_min)
            & (df["oi_change_1h"].fillna(0.0) >= p.oi_change_1h_min)
            & (df["spread_bps"] <= p.spread_bps_max)
        )
        return signal.astype(int)
