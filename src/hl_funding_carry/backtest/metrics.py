import pandas as pd


def summarize_backtest(df: pd.DataFrame) -> dict[str, float]:
    summary = {
        "rows": float(len(df)),
        "avg_position": float(df["position"].mean()) if "position" in df else 0.0,
    }
    return summary
