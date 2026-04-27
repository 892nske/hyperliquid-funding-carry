from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candles_path: Path = DATA_DIR / "raw" / "sample_candles.csv"
    asset_context_path: Path = DATA_DIR / "raw" / "sample_asset_context.csv"
    funding_path: Path = DATA_DIR / "raw" / "sample_funding.csv"


class StrategyEntryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_funding_min: float
    current_funding_min: float
    basis_min: float
    oi_change_1h_min: float
    spread_bps_max: float


class StrategyExitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    basis_exit: float
    basis_stop: float
    max_hold_hours: int
    predicted_funding_decay_ratio: float


class StrategyRiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_notional_pct: float
    max_positions: int


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["funding_carry"]
    symbols: list[str]
    signal_timeframe: str = Field(
        default="1h",
        validation_alias=AliasChoices("signal_timeframe", "timeframe"),
    )
    execution_timeframe: str = "5m"
    mode: Literal["spot_perp", "perp_only"] = "spot_perp"
    min_signal_interval_hours: int = 1
    entry: StrategyEntryConfig
    exit: StrategyExitConfig
    risk: StrategyRiskConfig


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["bar_close"] = "bar_close"
    fee_bps_maker: float
    fee_bps_taker: float
    slippage_bps: float


class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_features: bool = False
    save_trades: bool = False


class FundingCarryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig = DataConfig()
    strategy: StrategyConfig
    execution: ExecutionConfig
    research: ResearchConfig = ResearchConfig()


def load_config(path: Path) -> FundingCarryConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    return FundingCarryConfig.model_validate(raw_config)
