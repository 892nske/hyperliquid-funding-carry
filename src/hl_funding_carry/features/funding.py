import pandas as pd


REQUIRED_COLUMNS = [
    "ts",
    "symbol",
    "current_funding",
    "predicted_funding",
    "mark_price",
    "oracle_price",
    "open_interest",
    "spread_bps",
]


def build_funding_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out = out.sort_values(["symbol", "ts"]).reset_index(drop=True)
    out["basis"] = (out["mark_price"] - out["oracle_price"]) / out["oracle_price"]
    out["oi_change_1h"] = out.groupby("symbol")["open_interest"].pct_change()
    out["funding_mean_24"] = (
        out.groupby("symbol")["current_funding"].transform(lambda s: s.rolling(24, min_periods=12).mean())
    )
    out["funding_std_24"] = (
        out.groupby("symbol")["current_funding"].transform(lambda s: s.rolling(24, min_periods=12).std())
    )
    out["funding_z_24h"] = (out["current_funding"] - out["funding_mean_24"]) / out["funding_std_24"]
    return out
