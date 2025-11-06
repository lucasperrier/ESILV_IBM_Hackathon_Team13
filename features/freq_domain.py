# features/freq_domain.py
import numpy as np
import pandas as pd
from typing import Tuple


def bandpower(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psd[mask], freqs[mask])


def compute_freq_features(
    window: np.ndarray, fs: float, feature_prefix: str = ""
) -> dict:
    """
    FFT / PSD simple pour extraire quelques features fréquentielles.
    """
    feats = {}
    n = window.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    for ch in range(window.shape[1]):
        sig = window[:, ch]
        fft_vals = np.fft.rfft(sig)
        psd = (np.abs(fft_vals) ** 2) / n

        # Énergie globale
        feats[f"{feature_prefix}ch{ch}_power_total"] = bandpower(freqs, psd, 0.0, fs / 2)

        # Exemple de bandes (à adapter : suivant freq rotation hélices)
        feats[f"{feature_prefix}ch{ch}_power_low"] = bandpower(freqs, psd, 0.0, 5.0)
        feats[f"{feature_prefix}ch{ch}_power_mid"] = bandpower(freqs, psd, 5.0, 20.0)
        feats[f"{feature_prefix}ch{ch}_power_high"] = bandpower(freqs, psd, 20.0, fs / 2)

        # Pic spectral
        idx_max = np.argmax(psd)
        feats[f"{feature_prefix}ch{ch}_f_peak"] = freqs[idx_max]
        feats[f"{feature_prefix}ch{ch}_psd_peak"] = psd[idx_max]

    return feats


def extract_freq_features_from_windows(X: np.ndarray, fs: float) -> pd.DataFrame:
    """
    X: (n_windows, win_len, n_feat)
    """
    features_list = []
    for w in X:
        feats = compute_freq_features(w, fs)
        features_list.append(feats)
    return pd.DataFrame(features_list)
