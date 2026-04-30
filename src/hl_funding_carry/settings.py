from __future__ import annotations

from datetime import datetime
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

    source: Literal["local_files", "processed_dir"] = "local_files"
    candles_path: Path = DATA_DIR / "raw" / "sample_candles.csv"
    asset_context_path: Path = DATA_DIR / "raw" / "sample_asset_context.csv"
    funding_path: Path = DATA_DIR / "raw" / "sample_funding.csv"
    execution_5m_path: Path | None = DATA_DIR / "raw" / "sample_execution_5m.csv"
    execution_1m_path: Path | None = DATA_DIR / "raw" / "sample_execution_1m.csv"
    processed_dir: Path | None = None
    processed_recursive: bool = False


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
    allocation_mode: Literal["equal_weight", "fixed_notional"] = "equal_weight"
    fixed_notional_per_symbol: float | None = None
    max_gross_exposure: float = 0.15
    max_notional_per_symbol: float | None = None
    max_active_symbols: int = Field(
        default=1,
        validation_alias=AliasChoices("max_active_symbols", "max_positions"),
    )
    top_n_signals: int | None = None

    @property
    def max_positions(self) -> int:
        return self.max_active_symbols


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

    model: Literal["next_open", "twap_5m", "vwap_1m"] = Field(
        default="next_open",
        validation_alias=AliasChoices("model", "mode"),
    )
    fee_bps: float = Field(default=4.5, validation_alias=AliasChoices("fee_bps", "fee_bps_taker"))
    fee_bps_maker: float | None = None
    slippage_bps: float = 1.0
    timing_drag_bps: float = 0.0
    twap_window_minutes: int = 5
    vwap_window_minutes: int = 5


class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_features: bool = False
    save_trades: bool = False


class ArtifactConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_dir: Path = ARTIFACTS_DIR


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    train_window: str = "7D"
    test_window: str = "3D"
    step_size: str = "3D"
    selection_metric: Literal["total_return", "sharpe_like"] = "total_return"


class FundingCarryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig = DataConfig()
    strategy: StrategyConfig
    execution: ExecutionConfig
    research: ResearchConfig = ResearchConfig()
    artifacts: ArtifactConfig = ArtifactConfig()
    walkforward: WalkForwardConfig | None = None
    sweep_grid: SweepGridConfig | None = None


class SweepGridConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pred_funding_entry: list[float]
    basis_entry: list[float]
    entry_lead_minutes: list[int]
    max_hold_minutes: list[int]
    execution_model: list[Literal["next_open", "twap_5m", "vwap_1m"]] = ["next_open"]
    execution_slippage_bps: list[float] = [1.0]
    execution_fee_bps: list[float] = [4.5]


class HyperliquidTransportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["local_dump", "api"] = "local_dump"
    api_url: str = "https://api.hyperliquid.xyz/info"
    base_dir: Path | None = None
    candles_path: Path | None = None
    asset_context_path: Path | None = None
    funding_history_path: Path | None = None
    predicted_funding_path: Path | None = None
    execution_5m_path: Path | None = None
    execution_1m_path: Path | None = None


class IngestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["hyperliquid"] = "hyperliquid"
    symbol: str | None = None
    symbols: list[str] | None = None
    start: datetime
    end: datetime
    chunk_size: str | None = None
    candle_interval: str = "1h"
    execution_intervals: list[Literal["1m", "5m"]] = ["1m", "5m"]
    include_predicted_funding: bool = True
    raw_output_dir: Path = DATA_DIR / "raw" / "real"
    processed_output_dir: Path = DATA_DIR / "processed"
    execution_5m_path: Path | None = None
    execution_1m_path: Path | None = None
    hyperliquid: HyperliquidTransportConfig = HyperliquidTransportConfig()


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


def load_ingest_config(path: Path) -> IngestConfig:
    return IngestConfig.model_validate(_load_yaml(path))


FundingCarryConfig.model_rebuild()
