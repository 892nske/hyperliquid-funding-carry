from pathlib import Path

import pandas as pd


class DataLoader:
    """File-based historical loader for research datasets."""

    def load_parquet(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def load_csv(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)
