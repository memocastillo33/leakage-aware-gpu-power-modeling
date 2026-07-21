from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_devices: int = 10,
    n_samples_per_device: int = 500,
    noise_std: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    for device_id in range(n_devices):
        base_power = rng.uniform(50, 100)
        alpha = rng.uniform(0.8, 1.5)    # gpu_util coefficient
        beta = rng.uniform(0.02, 0.05)   # clock_mhz coefficient
        gamma = rng.uniform(0.1, 0.3)    # temp coefficient

        t = np.arange(n_samples_per_device)

        gpu_util = np.clip(
            50 + 30 * np.sin(2 * np.pi * t / 100)
            + rng.normal(0, 10, n_samples_per_device),
            0, 100,
        )
        mem_util = np.clip(
            gpu_util * 0.7 + rng.normal(0, 8, n_samples_per_device),
            0, 100,
        )
        temp_c = np.clip(
            40 + 0.4 * gpu_util + rng.normal(0, 3, n_samples_per_device),
            30, 95,
        )
        clock_mhz = np.clip(
            1200 + 600 * (gpu_util / 100) + rng.normal(0, 50, n_samples_per_device),
            800, 2000,
        )
        power_w = np.clip(
            base_power
            + alpha * gpu_util
            + beta * clock_mhz
            + gamma * temp_c
            + rng.normal(0, noise_std, n_samples_per_device),
            0, None,
        )

        records.append(pd.DataFrame({
            "device_id": device_id,
            "timestep": t,
            "gpu_util": gpu_util,
            "mem_util": mem_util,
            "temp_c": temp_c,
            "clock_mhz": clock_mhz,
            "power_w": power_w,
        }))

    return pd.concat(records, ignore_index=True)


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
