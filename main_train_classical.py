# main_train_classical.py
"""
Entraîne un baseline RandomForest sur les features temps+freq
et sauvegarde le modèle entraîné.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

from config import cfg
from features.time_domain import extract_time_features_from_windows
from features.freq_domain import extract_freq_features_from_windows
from models.classical import train_classical_models, predict_with_classical
from training.evaluation import (
    evaluate_fault_detection,
    evaluate_fault_type,
    evaluate_severity,
)


def main():
    processed_dir = cfg.data.processed_dir
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Chargement des fenêtres et labels depuis {processed_dir} ...")
    X = np.load(processed_dir / "X_windows.npy")       # (n_win, win_len, n_feat)
    y_fault = np.load(processed_dir / "y_fault.npy")   # (n_win,)
    y_type = np.load(processed_dir / "y_type.npy")     # (n_win,)
    y_sev = np.load(processed_dir / "y_sev.npy")       # (n_win,)

    print(f"[INFO] X.shape={X.shape}, y_fault.shape={y_fault.shape}")

    # 1) Features temps
    print("[INFO] Extraction des features temporels ...")
    df_time = extract_time_features_from_windows(X)

    # 2) Features freq
    print("[INFO] Extraction des features fréquentiels ...")
    df_freq = extract_freq_features_from_windows(X, fs=cfg.data.target_fs)

 

    # 3) Concat temps+freq
    X_feat = pd.concat([df_time, df_freq], axis=1)
    print(f"[INFO] Features concaténées : shape={X_feat.shape}")

    # 4) Split train/test
    X_train, X_test, y_fault_train, y_fault_test, y_type_train, y_type_test, y_sev_train, y_sev_test = train_test_split(
        X_feat,
        y_fault,
        y_type,
        y_sev,
        test_size=0.2,
        random_state=42,
        stratify=y_fault,
    )

    print(f"[INFO] Train size={len(X_train)}, Test size={len(X_test)}")

    # 5) Entraînement
    print("[INFO] Entraînement des modèles RandomForest (fault / type / severity) ...")
    models = train_classical_models(
        X_train,
        y_fault_train,
        y_type_train,
        y_sev_train,
    )

    # 6) Évaluation
    print("[INFO] Évaluation sur le set de test ...")
    prob_fault_test, type_pred_test, sev_pred_test = predict_with_classical(models, X_test)

    fault_metrics = evaluate_fault_detection(y_fault_test, prob_fault_test, thr=0.5)
    print("\n=== Détection de défaut (binaire) ===")
    print(f"Accuracy : {fault_metrics['accuracy']:.3f}")
    print(f"F1       : {fault_metrics['f1']:.3f}")

    type_metrics = evaluate_fault_type(y_type_test, type_pred_test)
    print("\n=== Type de défaut (multi-classes) ===")
    print(f"Accuracy : {type_metrics['accuracy']:.3f}")
    print(f"F1 macro : {type_metrics['f1_macro']:.3f}")
    print("Matrice de confusion :")
    print(type_metrics["conf_mat"])

    sev_metrics = evaluate_severity(y_sev_test, sev_pred_test)
    print("\n=== Sévérité ===")
    print(f"MAE (valeur réelle)        : {sev_metrics['mae']:.3f}")
    print(f"MAE (niveaux arrondis 0..3) : {sev_metrics['mae_levels']:.3f}")

    # 7) Sauvegarde du modèle + colonnes de features
    model_path = models_dir / "classical_rf_timefreq.joblib"
    meta = {
        "models": models,                        # dataclass ClassicalModels
        "feature_columns": list(X_feat.columns), # ordre des colonnes important
    }
    dump(meta, model_path)
    print(f"\n[INFO] Modèle RandomForest sauvegardé dans {model_path}")


if __name__ == "__main__":
    main()
