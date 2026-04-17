from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FundingObservation:
    ts: datetime
    symbol: str
    current_funding: float
    predicted_funding: float
    mark_price: float
    oracle_price: float
    open_interest: float
