# features/time_domain.py
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def compute_time_features(window: np.ndarray, feature_prefix: str = "") -> dict:
    """
    Calcule les features temporelles de base pour une fenêtre (win_len, n_feat).
    Renvoie un dict {feature_name: value}.
    """
    feats = {}
    # window: (win_len, n_feat)
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    minimum = window.min(axis=0)
    maximum = window.max(axis=0)
    ptp = maximum - minimum
    rms = np.sqrt((window ** 2).mean(axis=0))
    sk = skew(window, axis=0)
    kt = kurtosis(window, axis=0)

    n_feat = window.shape[1]
    for i in range(n_feat):
        base = f"{feature_prefix}ch{i}"
        feats[f"{base}_mean"] = mean[i]
        feats[f"{base}_std"] = std[i]
        feats[f"{base}_min"] = minimum[i]
        feats[f"{base}_max"] = maximum[i]
        feats[f"{base}_ptp"] = ptp[i]
        feats[f"{base}_rms"] = rms[i]
        feats[f"{base}_skew"] = sk[i]
        feats[f"{base}_kurt"] = kt[i]

    return feats


def extract_time_features_from_windows(X: np.ndarray) -> pd.DataFrame:
    """
    X: (n_windows, win_len, n_feat)
    Renvoie un DataFrame de features.
    """
    features_list = []
    for w in X:
        feats = compute_time_features(w)
        features_list.append(feats)
    return pd.DataFrame(features_list)
