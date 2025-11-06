# main_train_xgboost.py
"""
Entraîne un baseline XGBoost sur les features temps+freq
et sauvegarde le modèle entraîné.
"""

from pathlib import Path
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import cfg
from features.time_domain import extract_time_features_from_windows
from features.freq_domain import extract_freq_features_from_windows

from models.xgboost import train_xgboost_models, predict_with_xgboost
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
    X = np.load(processed_dir / "X_windows.npy")
    y_fault = np.load(processed_dir / "y_fault.npy")
    y_type = np.load(processed_dir / "y_type.npy")
    y_sev = np.load(processed_dir / "y_sev.npy")

    print(f"[INFO] X.shape={X.shape}, y_fault.shape={y_fault.shape}")

    print("[INFO] Extraction des features temporels ...")
    df_time = extract_time_features_from_windows(X)

    print("[INFO] Extraction des features fréquentiels ...")
    df_freq = extract_freq_features_from_windows(X, fs=cfg.data.target_fs)

    X_feat = pd.concat([df_time, df_freq], axis=1)
    print(f"[INFO] Features concaténées : shape={X_feat.shape}")

    X_train, X_test, y_fault_train, y_fault_test, y_type_train, y_type_test, y_sev_train, y_sev_test = train_test_split(
        X_feat, y_fault, y_type, y_sev, test_size=0.2, random_state=42, stratify=y_fault
    )

    print("[INFO] Entraînement des modèles XGBoost ...")
    models = train_xgboost_models(X_train, y_fault_train, y_type_train, y_sev_train)

    print("[INFO] Évaluation sur le set de test ...")
    prob_fault_test, type_pred_test, sev_pred_test = predict_with_xgboost(models, X_test)

    fault_metrics = evaluate_fault_detection(y_fault_test, prob_fault_test, thr=0.5)
    print("\n=== Détection de défaut ===")
    print(f"Accuracy : {fault_metrics['accuracy']:.3f}")
    print(f"F1       : {fault_metrics['f1']:.3f}")

    type_metrics = evaluate_fault_type(y_type_test, type_pred_test)
    print("\n=== Type de défaut ===")
    print(f"Accuracy : {type_metrics['accuracy']:.3f}")
    print(f"F1 macro : {type_metrics['f1_macro']:.3f}")
    print(type_metrics["conf_mat"])

    sev_metrics = evaluate_severity(y_sev_test, sev_pred_test)
    print("\n=== Sévérité ===")
    print(f"MAE (val réel)      : {sev_metrics['mae']:.3f}")
    print(f"MAE (niveaux 0..3)  : {sev_metrics['mae_levels']:.3f}")

    model_path = models_dir / "xgboost_timefreq.joblib"
    meta = {"models": models, "feature_columns": list(X_feat.columns)}
    dump(meta, model_path)

    print(f"\n[INFO] Modèle XGBoost sauvegardé dans {model_path}")


if __name__ == "__main__":
    main()