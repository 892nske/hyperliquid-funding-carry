import pandas as pd


def simulate_positions(df: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = signal.astype(int)
    out["position"] = out["signal"].shift(1).fillna(0).astype(int)
    return out
