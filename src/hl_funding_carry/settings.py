from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


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
    predicted_funding_decay_ratio: float


class StrategyTimingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    funding_interval_minutes: int = 60
    entry_lead_minutes: int = 60
    min_hold_minutes_after_funding: int = 60
    max_hold_minutes: int = Field(
        default=120,
        validation_alias=AliasChoices("max_hold_minutes", "max_hold_hours"),
    )


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
    timing: StrategyTimingConfig = StrategyTimingConfig()
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


class ArtifactConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_dir: Path = ARTIFACTS_DIR


class FundingCarryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig = DataConfig()
    strategy: StrategyConfig
    execution: ExecutionConfig
    research: ResearchConfig = ResearchConfig()
    artifacts: ArtifactConfig = ArtifactConfig()


class SweepGridConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pred_funding_entry: list[float]
    basis_entry: list[float]
    entry_lead_minutes: list[int]
    max_hold_minutes: list[int]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in {path}")
    return loaded


def load_config(path: Path) -> FundingCarryConfig:
    return FundingCarryConfig.model_validate(_load_yaml(path))


def load_sweep_grid(path: Path) -> SweepGridConfig:
    return SweepGridConfig.model_validate(_load_yaml(path))
