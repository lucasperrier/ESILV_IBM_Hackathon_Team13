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


# data/preprocessing.py
import numpy as np
import pandas as pd

def normalize_df(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    """
    Normalisation (x - mean) / std avec :
      - std==0 géré pour éviter division par 0
      - NaN / Inf remplacés par 0
    """
    df = df.copy()

    # On suppose que 'time' est la colonne temps si présente
    time_col = None
    if "time" in df.columns:
        time_col = df["time"].values
        df_num = df.drop(columns=["time"])
    else:
        df_num = df

    # Sécuriser std : remplacer 0 par 1.0 (pas de scaling pour ces features)
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0

    # Aligner mean/std sur les colonnes de df_num
    mean_s = mean[df_num.columns]
    std_s = std_safe[df_num.columns]

    df_norm = (df_num - mean_s) / std_s

    # Nettoyage NaN / Inf
    df_norm = df_norm.replace([np.inf, -np.inf], 0.0)
    df_norm = df_norm.fillna(0.0)

    if time_col is not None:
        df_norm.insert(0, "time", time_col)

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
