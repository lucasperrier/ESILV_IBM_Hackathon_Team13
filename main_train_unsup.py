# main_train_unsup.py
"""
Entraîne un modèle non supervisé (IsolationForest) sur des fenêtres saines,
à partir des features temps+freq, et sauvegarde le modèle + seuil d'anomalie.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump

from config import cfg
from features.time_domain import extract_time_features_from_windows
from features.freq_domain import extract_freq_features_from_windows
from training.train_unsup import train_isolation_forest, anomaly_score
from training.evaluation import evaluate_fault_detection


def main():
    processed_dir = cfg.data.processed_dir
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Chargement des données prétraitées depuis {processed_dir} ...")
    X = np.load(processed_dir / "X_windows.npy")       # (n_win, win_len, n_feat)
    y_fault = np.load(processed_dir / "y_fault.npy")   # (n_win,)

    print(f"[INFO] X.shape={X.shape}, y_fault.shape={y_fault.shape}")

    # 1) Features temps
    print("[INFO] Extraction des features temporels ...")
    df_time = extract_time_features_from_windows(X)  # DataFrame

    # 2) Features fréquentiels
    print("[INFO] Extraction des features fréquentiels ...")
    df_freq = extract_freq_features_from_windows(X, fs=cfg.data.target_fs)

    # 3) Concaténation
    X_feat = pd.concat([df_time, df_freq], axis=1)
    print(f"[INFO] Features temps+freq : shape={X_feat.shape}")

    # 4) Sélection des fenêtres saines pour apprentissage unsupervised
    healthy_mask = (y_fault == 0)
    if healthy_mask.sum() == 0:
        raise RuntimeError("Aucune fenêtre saine (y_fault == 0) trouvée pour entraîner l'IsolationForest.")

    X_healthy = X_feat[healthy_mask].values
    X_all = X_feat.values

    print(f"[INFO] Fenêtres saines pour entraînement : {X_healthy.shape[0]} / {X_all.shape[0]}")

    # 5) Entraînement de l'IsolationForest
    print("[INFO] Entraînement de l'IsolationForest (unsupervised) ...")
    iso = train_isolation_forest(X_healthy)

    # 6) Calcul des scores d'anomalie sur toutes les fenêtres
    print("[INFO] Calcul des scores d'anomalie ...")
    scores = anomaly_score(iso, X_all)  # plus grand = plus anormal

    # 7) Choix du seuil basé sur les données saines (ex: 95ème percentile)
    healthy_scores = scores[healthy_mask]
    threshold = np.percentile(healthy_scores, 95.0)

    print(f"[INFO] Seuil d'anomalie choisi (95ème percentile des sains) : {threshold:.4f}")

    # 8) Évaluation rapide en mode détection de défaut (binaire)
    metrics = evaluate_fault_detection(y_fault, scores, thr=threshold)
    print("\n=== Évaluation unsupervised (IsolationForest) ===")
    print(f"Accuracy (fault / no fault) : {metrics['accuracy']:.3f}")
    print(f"F1 (fault)                 : {metrics['f1']:.3f}")

    # 9) Sauvegarde du modèle + seuil
    model_path = models_dir / "iso_forest_timefreq.joblib"
    thr_path = models_dir / "iso_forest_threshold.npy"

    dump(iso, model_path)
    np.save(thr_path, np.array([threshold]))

    print(f"[INFO] IsolationForest sauvegardé dans {model_path}")
    print(f"[INFO] Seuil d'anomalie sauvegardé dans {thr_path}")


if __name__ == "__main__":
    main()
