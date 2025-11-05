# training/train_unsup.py
import numpy as np
from sklearn.ensemble import IsolationForest


def train_isolation_forest(X_healthy: np.ndarray) -> IsolationForest:
    """
    Baseline : détecteur non supervisé entraîné uniquement sur des fenêtres saines.
    X_healthy : (n_win, dim_feat) après extraction de features.
    """
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.1,
        random_state=42,
    )
    iso.fit(X_healthy)
    return iso


def anomaly_score(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Renvoie un score d'anomalie (plus grand = plus anormal).
    """
    # IsolationForest renvoie des scores de type "anomaly score" négatifs.
    raw = model.score_samples(X)
    return -raw
