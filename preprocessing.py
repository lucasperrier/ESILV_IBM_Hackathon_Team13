# data/preprocessing.py
from typing import Tuple, List
import numpy as np
import pandas as pd


def resample_signal(df: pd.DataFrame, target_fs: float) -> pd.DataFrame:
    """
    Resample le DataFrame à une fréquence fixe target_fs par interpolation.
    """
    t = df["time"].values
    duration = t[-1] - t[0]
    n_samples = int(duration * target_fs) + 1
    new_t = np.linspace(t[0], t[-1], n_samples)
    out = {"time": new_t}
    for col in df.columns:
        if col == "time":
            continue
        out[col] = np.interp(new_t, t, df[col].values)
    return pd.DataFrame(out)


def normalize_df(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    """
    Normalise un DataFrame à partir de mean/std calculés sur le train.
    """
    df_norm = df.copy()
    for col in df.columns:
        if col == "time":
            continue
        df_norm[col] = (df[col] - mean[col]) / (std[col] + 1e-8)
    return df_norm


def compute_global_mean_std(dfs: List[pd.DataFrame]) -> Tuple[pd.Series, pd.Series]:
    """
    Calcule mean/std par feature sur une liste de DataFrames (train only).
    """
    concat = pd.concat([df.drop(columns=["time"]) for df in dfs], axis=0)
    mean = concat.mean()
    std = concat.std()
    return mean, std


def sliding_windows(
    df: pd.DataFrame, win_sec: float, step_sec: float, fs: float
) -> np.ndarray:
    """
    Découpe un DataFrame en fenêtres glissantes, renvoie un array (n_win, win_len, n_feat).
    """
    win_len = int(win_sec * fs)
    step = int(step_sec * fs)
    values = df.drop(columns=["time"]).values
    windows = []

    for start in range(0, len(df) - win_len, step):
        seg = values[start : start + win_len]
        windows.append(seg)

    return np.stack(windows) if windows else np.empty((0, win_len, values.shape[1]))
