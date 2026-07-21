from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
    df = df.copy()

    # Rolling means per device — uses only past data within the same device
    for col in ("gpu_util", "mem_util"):
        df[f"{col}_roll_mean"] = (
            df.groupby("device_id")[col]
            .transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
        )

    # Ratio and interaction features — use no global statistics
    df["temp_clock_ratio"] = df["temp_c"] / df["clock_mhz"]
    df["util_product"] = df["gpu_util"] * df["mem_util"] / 100.0

    return df.dropna().reset_index(drop=True)


def get_feature_cols(cfg: dict) -> list[str]:
    return cfg["features"]["feature_cols"]


class LeakageAwareScaler:
    """StandardScaler wrapper that must be fit only on training data."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.transform(X)
