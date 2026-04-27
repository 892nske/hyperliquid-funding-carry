from __future__ import annotations

import json
from pathlib import Path
from urllib import request

import pandas as pd

from hl_funding_carry.settings import IngestConfig


def _load_payload(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
            return pd.DataFrame(payload["data"])
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
    raise ValueError(f"Unsupported raw payload file: {path}")


def _post_info(api_url: str, payload: dict[str, object]) -> pd.DataFrame:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req) as response:  # noqa: S310
        parsed = json.loads(response.read().decode("utf-8"))
    if isinstance(parsed, list):
        return pd.DataFrame(parsed)
    if isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], list):
        return pd.DataFrame(parsed["data"])
    if isinstance(parsed, dict):
        return pd.DataFrame([parsed])
    raise ValueError("Unexpected Hyperliquid API response")


def fetch_hyperliquid_raw(config: IngestConfig) -> dict[str, pd.DataFrame]:
    transport = config.hyperliquid
    if transport.mode == "local_dump":
        return {
            "candles_raw": _load_payload(transport.candles_path),
            "asset_context_raw": _load_payload(transport.asset_context_path),
            "funding_history_raw": _load_payload(transport.funding_history_path),
            "predicted_funding_raw": _load_payload(transport.predicted_funding_path)
            if config.include_predicted_funding
            else pd.DataFrame(),
        }

    start_ms = int(pd.Timestamp(config.start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(config.end, tz="UTC").timestamp() * 1000)
    candles = _post_info(
        transport.api_url,
        {
            "type": "candleSnapshot",
            "req": {
                "coin": config.symbol,
                "interval": config.candle_interval,
                "startTime": start_ms,
                "endTime": end_ms,
            },
        },
    )
    funding_history = _post_info(
        transport.api_url,
        {
            "type": "fundingHistory",
            "coin": config.symbol,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    )
    predicted_funding = pd.DataFrame()
    if config.include_predicted_funding:
        predicted_funding = _post_info(
            transport.api_url,
            {"type": "predictedFundings"},
        )
    asset_context = _load_payload(transport.asset_context_path)
    return {
        "candles_raw": candles,
        "asset_context_raw": asset_context,
        "funding_history_raw": funding_history,
        "predicted_funding_raw": predicted_funding,
    }


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], output_name: str) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return df[column]
    raise ValueError(f"Missing required source column for {output_name}: {candidates}")


def _optional_column(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    default: float,
) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def normalize_hyperliquid_candles(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("Hyperliquid candles payload is empty")
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                _pick_column(raw, ("timestamp", "time", "t"), "timestamp"),
                utc=True,
            ).dt.floor("1h"),
            "symbol": symbol,
            "open": pd.to_numeric(_pick_column(raw, ("open", "o"), "open"), errors="coerce"),
            "high": pd.to_numeric(_pick_column(raw, ("high", "h"), "high"), errors="coerce"),
            "low": pd.to_numeric(_pick_column(raw, ("low", "l"), "low"), errors="coerce"),
            "close": pd.to_numeric(_pick_column(raw, ("close", "c"), "close"), errors="coerce"),
        },
    )
    return normalized.drop_duplicates(subset=["timestamp", "symbol"], keep="last").sort_values(
        ["timestamp", "symbol"],
    )


def normalize_hyperliquid_asset_context(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("Hyperliquid asset context payload is empty")
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                _pick_column(raw, ("timestamp", "time", "t"), "timestamp"),
                utc=True,
            ).dt.floor("1h"),
            "symbol": symbol,
            "mark_price": pd.to_numeric(
                _pick_column(raw, ("mark_price", "markPx", "mark_px"), "mark_price"),
                errors="coerce",
            ),
            "oracle_price": pd.to_numeric(
                _pick_column(raw, ("oracle_price", "oraclePx", "oracle_px"), "oracle_price"),
                errors="coerce",
            ),
            "current_funding": pd.to_numeric(
                _pick_column(
                    raw,
                    ("current_funding", "funding", "fundingRate", "funding_rate"),
                    "current_funding",
                ),
                errors="coerce",
            ),
            "open_interest": pd.to_numeric(
                _pick_column(raw, ("open_interest", "openInterest"), "open_interest"),
                errors="coerce",
            ),
            "spread_bps": _optional_column(raw, ("spread_bps", "spreadBps"), 0.0),
            "spot_price": pd.to_numeric(
                _pick_column(raw, ("spot_price", "spotPx", "spot_px"), "spot_price"),
                errors="coerce",
            ),
        },
    )
    return normalized.drop_duplicates(subset=["timestamp", "symbol"], keep="last").sort_values(
        ["timestamp", "symbol"],
    )


def normalize_hyperliquid_funding_history(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "funding_rate"])
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                _pick_column(raw, ("timestamp", "time", "t"), "timestamp"),
                utc=True,
            ).dt.floor("1h"),
            "symbol": symbol,
            "funding_rate": pd.to_numeric(
                _pick_column(raw, ("funding_rate", "fundingRate", "funding"), "funding_rate"),
                errors="coerce",
            ),
        },
    )
    return normalized.drop_duplicates(subset=["timestamp", "symbol"], keep="last").sort_values(
        ["timestamp", "symbol"],
    )


def normalize_hyperliquid_predicted_funding(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "pred_funding_1h"])
    filtered = raw.copy()
    if "coin" in filtered.columns:
        filtered = filtered.loc[filtered["coin"].astype(str).str.upper() == symbol.upper()]
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                _pick_column(filtered, ("timestamp", "time", "t"), "timestamp"),
                utc=True,
            ).dt.floor("1h"),
            "symbol": symbol,
            "pred_funding_1h": pd.to_numeric(
                _pick_column(
                    filtered,
                    ("pred_funding_1h", "predictedFunding", "predicted_funding", "funding"),
                    "pred_funding_1h",
                ),
                errors="coerce",
            ),
        },
    )
    return normalized.drop_duplicates(subset=["timestamp", "symbol"], keep="last").sort_values(
        ["timestamp", "symbol"],
    )


def build_funding_inputs(
    asset_context: pd.DataFrame,
    predicted_funding: pd.DataFrame,
) -> pd.DataFrame:
    base = asset_context.loc[:, ["timestamp", "symbol", "current_funding"]].rename(
        columns={"current_funding": "pred_funding_1h"},
    )
    if predicted_funding.empty:
        return base.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    merged = base.merge(
        predicted_funding,
        on=["timestamp", "symbol"],
        how="left",
        suffixes=("", "_x"),
    )
    merged["pred_funding_1h"] = merged["pred_funding_1h_x"].fillna(merged["pred_funding_1h"])
    output = merged.loc[:, ["timestamp", "symbol", "pred_funding_1h"]]
    return output.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
