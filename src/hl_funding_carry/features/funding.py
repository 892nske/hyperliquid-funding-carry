from __future__ import annotations

import numpy as np
import pandas as pd

from hl_funding_carry.backtest.events import infer_bar_interval_minutes
from hl_funding_carry.types import RESEARCH_COLUMNS


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    zscore = (series - rolling_mean) / rolling_std.replace(0.0, np.nan)
    return zscore.fillna(0.0)


def _cross_sectional_rank(df: pd.DataFrame, column: str) -> pd.Series:
    rank = df.groupby("timestamp")[column].rank(method="average", pct=True)
    return rank.fillna(0.0)


def build_funding_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in RESEARCH_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    features = df.copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    features["symbol"] = features["symbol"].astype(str).str.upper().str.strip()
    features = features.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    bar_interval_minutes = infer_bar_interval_minutes(features)
    one_hour_periods = max(1, int(round(60 / bar_interval_minutes)))
    four_hour_periods = max(1, int(round(240 / bar_interval_minutes)))
    twenty_four_hour_periods = max(1, int(round(1440 / bar_interval_minutes)))

    features["basis"] = (
        (features["mark_price"] - features["oracle_price"]) / features["oracle_price"]
    )
    features["basis_bps"] = features["basis"] * 10000.0
    features["basis_change_1h"] = features.groupby("symbol")["basis"].diff(periods=one_hour_periods)
    features["oi_change_1h"] = features.groupby("symbol")["open_interest"].pct_change(
        periods=one_hour_periods,
    )
    features["oi_change_4h"] = features.groupby("symbol")["open_interest"].pct_change(
        periods=four_hour_periods,
    )
    features["funding_z_24h"] = features.groupby("symbol")["current_funding"].transform(
        lambda series: _rolling_zscore(series, window=twenty_four_hour_periods),
    )
    features["spread_ok"] = np.isfinite(features["spread_bps"]).astype(int)

    features["pred_funding_rank"] = _cross_sectional_rank(features, "pred_funding_1h")
    features["basis_rank"] = _cross_sectional_rank(features, "basis")
    features["oi_rank"] = _cross_sectional_rank(features, "oi_change_1h")
    features["carry_score"] = (
        0.5 * features["pred_funding_rank"]
        + 0.3 * features["basis_rank"]
        + 0.2 * features["oi_rank"]
    )

    return features
