# main_train_knn_cuda.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch

from config import cfg
from features.time_domain import extract_time_features_from_windows
from features.freq_domain import extract_freq_features_from_windows
from models.knn_cuda import (
    CudaKNNCombined,
    encode_combined_labels,
    decode_combined_labels,
)
from training.evaluation import (
    evaluate_fault_type,
    evaluate_severity,
)


def main():
    processed_dir = cfg.data.processed_dir
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Chargement des données prétraitées depuis {processed_dir} ...")
    X = np.load(processed_dir / "X_windows.npy")       # (n_win, win_len, n_feat)
    y_fault = np.load(processed_dir / "y_fault.npy")   # (n_win,)
    y_type = np.load(processed_dir / "y_type.npy")     # (n_win,)
    y_sev = np.load(processed_dir / "y_sev.npy")       # (n_win,)

    print(f"[INFO] X.shape={X.shape}")
    print(f"[INFO] y_fault unique: {np.unique(y_fault)}")
    print(f"[INFO] y_type unique : {np.unique(y_type)}")
    print(f"[INFO] y_sev unique  : {np.unique(y_sev)}")

    # 0) Labels combinés (pour info)
    y_combined_all = encode_combined_labels(y_fault, y_type, y_sev)
    combos = np.unique(y_combined_all)
    print(f"[INFO] Combinaisons labels (code fault*100+type*10+sev) observées : {combos}")
    print(f"[INFO] Nombre de combinaisons distinctes : {len(combos)}")

    # 1) Features temps + freq
    print("[INFO] Extraction des features temporels ...")
    df_time = extract_time_features_from_windows(X)

    print("[INFO] Extraction des features fréquentiels ...")
    df_freq = extract_freq_features_from_windows(X, fs=cfg.data.target_fs)

    X_feat = pd.concat([df_time, df_freq], axis=1)
    print(f"[INFO] Features concaténées : shape={X_feat.shape}")

    # 2) Split train / test
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

    # Labels combinés pour le test (référence)
    y_combined_test = encode_combined_labels(y_fault_test, y_type_test, y_sev_test)

    # 3) Entraînement KNN CUDA (stockage sur GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device KNN : {device}")

    knn = CudaKNNCombined(k=5, device=device)
    print("[INFO] Fit du KNN (envoi des données sur GPU) ...")
    knn.fit(X_train, y_fault_train, y_type_train, y_sev_train)

    # 4) Prédiction et évaluation
    print("[INFO] Prédiction KNN sur le set de test ...")
    # a) directement les labels combinés
    y_combined_pred = knn.predict_combined(X_test, batch_size=256)

    # b) décodage en triplets (pour éventuellement comparer aux anciennes métriques)
    y_fault_pred, y_type_pred, y_sev_pred = decode_combined_labels(y_combined_pred)

    # Évaluation sur les COMBINAISONS complètes
    print("\n=== Évaluation sur les combinaisons complètes (fault, type, severity) ===")
    classes = np.unique(y_combined_test)  # les combinaisons réellement présentes

    comb_acc = accuracy_score(y_combined_test, y_combined_pred)
    comb_f1 = f1_score(y_combined_test, y_combined_pred, average="macro")
    comb_cm = confusion_matrix(y_combined_test, y_combined_pred, labels=classes)

    print(f"Accuracy (combinaisons) : {comb_acc:.3f}")
    print(f"F1-macro (combinaisons): {comb_f1:.3f}")
    print("Matrice de confusion (classes = codes fault*100+type*10+sev) :")
    print("Classes (lignes/colonnes) :", classes)
    print(comb_cm)

    # Pour aider à interpréter les codes, on affiche le mapping code -> (fault, type, sev)
    print("\nMapping des classes combinées : code -> (fault, type, severity)")
    for c in classes:
        f, t, s = decode_combined_labels(np.array([c]))
        print(f"  {c} -> (fault={int(f[0])}, type={int(t[0])}, sev={int(s[0])})")

    print("\n=== Détection de défaut (binaire, dérivé du label combiné) ===")
    from sklearn.metrics import f1_score as f1_bin, accuracy_score as acc_bin
    fault_acc = acc_bin(y_fault_test, y_fault_pred)
    fault_f1 = f1_bin(y_fault_test, y_fault_pred, average="binary")
    print(f"Accuracy : {fault_acc:.3f}")
    print(f"F1       : {fault_f1:.3f}")

    type_metrics = evaluate_fault_type(y_type_test, y_type_pred)
    print("\n=== Type de défaut (multi-classes) ===")
    print(f"Accuracy : {type_metrics['accuracy']:.3f}")
    print(f"F1 macro : {type_metrics['f1_macro']:.3f}")
    print("Matrice de confusion :")
    print(type_metrics["conf_mat"])

    sev_metrics = evaluate_severity(y_sev_test, y_sev_pred)
    print("\n=== Sévérité ===")
    print(f"MAE (valeur réelle)        : {sev_metrics['mae']:.3f}")
    print(f"MAE (niveaux arrondis 0..3) : {sev_metrics['mae_levels']:.3f}")

    # 5) Sauvegarde du modèle
    state = knn.to_state_dict()
    model_path = models_dir / "knn_cuda_combined.pth"
    torch.save(
        {
            "state": state,
            "feature_columns": list(X_feat.columns),
            "classes_combined": classes,
        },
        model_path,
    )

    print(f"\n[INFO] Modèle KNN CUDA sauvegardé dans {model_path}")


if __name__ == "__main__":
    main()
