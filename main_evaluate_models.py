# main_evaluate_models.py
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error

import torch
from joblib import load

from config import cfg
from features.time_domain import extract_time_features_from_windows
from features.freq_domain import extract_freq_features_from_windows


# XGBoost GPU
from models.classical import predict_with_xgb

# KNN CUDA
from models.knn_cuda import (
    CudaKNNCombined,
    encode_combined_labels,
    decode_combined_labels,
)

# Deep MTL
from models.deep_mtl import CNNMTL


def compute_all_metrics(
    y_fault_true,
    y_type_true,
    y_sev_true,
    y_fault_pred,
    y_type_pred,
    y_sev_pred,
):
    """Calcule toutes les métriques pour un modèle."""
    metrics = {}

    # 1) Fault detection (binaire)
    metrics["fault_acc"] = accuracy_score(y_fault_true, y_fault_pred)
    metrics["fault_f1"] = f1_score(y_fault_true, y_fault_pred, average="binary")

    # 2) Fault type (multi-classes)
    metrics["type_acc"] = accuracy_score(y_type_true, y_type_pred)
    metrics["type_f1_macro"] = f1_score(y_type_true, y_type_pred, average="macro")

    # 3) Severity (0..3)
    metrics["sev_mae"] = mean_absolute_error(y_sev_true, y_sev_pred)
    metrics["sev_acc"] = accuracy_score(y_sev_true, y_sev_pred)
    metrics["sev_f1_macro"] = f1_score(y_sev_true, y_sev_pred, average="macro")

    # 4) Combinaisons complètes
    y_comb_true = encode_combined_labels(y_fault_true, y_type_true, y_sev_true)
    y_comb_pred = encode_combined_labels(y_fault_pred, y_type_pred, y_sev_pred)

    metrics["comb_acc"] = accuracy_score(y_comb_true, y_comb_pred)
    metrics["comb_f1_macro"] = f1_score(y_comb_true, y_comb_pred, average="macro")

    classes = np.unique(np.concatenate([y_comb_true, y_comb_pred]))
    cm = confusion_matrix(y_comb_true, y_comb_pred, labels=classes)
    metrics["comb_confusion"] = cm
    metrics["comb_classes"] = classes

    return metrics


def print_metrics_table(all_results):
    """Affiche un tableau comparatif des métriques principales."""
    print("\n===== COMPARAISON DES MODÈLES (mêmes données de test) =====")
    print(
        f"{'Model':20s} | "
        f"{'fault_acc':9s} {'fault_f1':9s} | "
        f"{'type_acc':9s} {'type_f1':9s} | "
        f"{'sev_mae':9s} {'sev_acc':9s} | "
        f"{'comb_acc':9s} {'comb_f1':9s}"
    )
    print("-" * 110)
    for name, m in all_results.items():
        print(
            f"{name:20s} | "
            f"{m['fault_acc']:.3f}    {m['fault_f1']:.3f}    | "
            f"{m['type_acc']:.3f}    {m['type_f1_macro']:.3f}    | "
            f"{m['sev_mae']:.3f}    {m['sev_acc']:.3f}    | "
            f"{m['comb_acc']:.3f}    {m['comb_f1_macro']:.3f}"
        )


def print_comb_confusions(all_results):
    """Affiche les matrices de confusion combinées, avec décodage (fault,type,sev)."""
    print("\n===== MATRICES DE CONFUSION SUR LES COMBINAISONS (fault,type,sev) =====")
    for name, m in all_results.items():
        print(f"\n--- {name} ---")
        classes = m["comb_classes"]
        cm = m["comb_confusion"]
        print("Classes (codes fault*100+type*10+sev) :", classes)
        # Afficher mapping lisible
        print("Mapping code -> (fault, type, sev) :")
        for c in classes:
            f, t, s = decode_combined_labels(np.array([c]))
            print(f"  {c} -> (fault={int(f[0])}, type={int(t[0])}, sev={int(s[0])})")
        print("Confusion matrix :")
        print(cm)


def main():
    processed_dir = cfg.data.processed_dir
    models_dir = Path("models")

    # =========================
    #  Chargement des données
    # =========================
    print(f"[INFO] Chargement des données depuis {processed_dir} ...")
    X = np.load(processed_dir / "X_windows.npy")       # (n_win, win_len, n_feat)
    y_fault = np.load(processed_dir / "y_fault.npy")
    y_type = np.load(processed_dir / "y_type.npy")
    y_sev = np.load(processed_dir / "y_sev.npy")

    
    n_samples, win_len, n_feat = X.shape
    print(f"[INFO] X.shape={X.shape}, n_samples={n_samples}")

    # Features temps + freq (pour RF / XGB / KNN)
    print("[INFO] Extraction des features temporels ...")
    df_time = extract_time_features_from_windows(X)

    print("[INFO] Extraction des features fréquentiels ...")
    df_freq = extract_freq_features_from_windows(X, fs=cfg.data.target_fs)

    X_feat_full = pd.concat([df_time, df_freq], axis=1)
    print(f"[INFO] Features concaténées : shape={X_feat_full.shape}")

    # =========================
    #  Split train / test unique
    # =========================
    idx = np.arange(n_samples)
    idx_train, idx_test = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        stratify=y_fault,
    )

    # y de test
    y_fault_test = y_fault[idx_test]
    y_type_test = y_type[idx_test]
    y_sev_test = y_sev[idx_test]

    # fenêtres test pour deep MTL
    X_test_win = X[idx_test]

    # features test pour modèles classiques / KNN
    X_feat_test_full = X_feat_full.iloc[idx_test].reset_index(drop=True)

    all_results = {}


    # =========================
    # XGBoost GPU
    # =========================
    xgb_path = models_dir / "classical_xgb_timefreq_gpu.joblib"
    if xgb_path.exists():
        print(f"[INFO] Évaluation XGBoost GPU (fichier: {xgb_path.name}) ...")
        meta_xgb = load(xgb_path)
        xgb_models = meta_xgb["models"]
        xgb_features = meta_xgb["feature_columns"]

        X_xgb_test = X_feat_test_full[xgb_features]

        prob_fault_xgb, type_pred_xgb, sev_pred_xgb_cont = predict_with_xgb(xgb_models, X_xgb_test)
        fault_pred_xgb = (prob_fault_xgb >= 0.5).astype(int)
        sev_pred_xgb = np.clip(np.round(sev_pred_xgb_cont), 0, 3).astype(int)

        metrics_xgb = compute_all_metrics(
            y_fault_test,
            y_type_test,
            y_sev_test,
            fault_pred_xgb,
            type_pred_xgb,
            sev_pred_xgb,
        )
        all_results["XGBoost_GPU"] = metrics_xgb
    else:
        print("[WARN] Modèle XGBoost GPU non trouvé, skip.")

    # =========================
    # 3) KNN CUDA (labels combinés)
    # =========================
    knn_path = models_dir / "knn_cuda_combined.pth"
    if knn_path.exists():
        print(f"[INFO] Évaluation KNN CUDA (fichier: {knn_path.name}) ...")
        checkpoint = torch.load(knn_path, map_location="cpu")
        state_knn = checkpoint["state"]
        knn_features = checkpoint["feature_columns"]

        device_knn = "cuda" if torch.cuda.is_available() else "cpu"
        knn = CudaKNNCombined.from_state_dict(state_knn, device=device_knn)

        X_knn_test = X_feat_test_full[knn_features]

        y_fault_pred_knn, y_type_pred_knn, y_sev_pred_knn = knn.predict_triplet(X_knn_test, batch_size=256)

        metrics_knn = compute_all_metrics(
            y_fault_test,
            y_type_test,
            y_sev_test,
            y_fault_pred_knn,
            y_type_pred_knn,
            y_sev_pred_knn,
        )
        all_results["KNN_CUDA"] = metrics_knn
    else:
        print("[WARN] Modèle KNN CUDA non trouvé, skip.")

    # =========================
    # 4) Deep MTL (CNN multi-tâches)
    # =========================
    # adapte le nom si ton main_train_deep_mtl.py en a utilisé un autre

    deep_path = models_dir / "deep_mtl_cnn1d.pth"

    if deep_path is not None:
        print(f"[INFO] Évaluation Deep MTL (fichier: {deep_path.name}) ...")
        checkpoint = torch.load(deep_path, map_location="cpu")

        state_dict = checkpoint["state_dict"]
        in_channels = checkpoint.get("n_features", n_feat)
        n_fault_types = checkpoint.get("n_fault_types", int(y_type.max() + 1))
    

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CNNMTL(in_channels=in_channels, n_fault_types=n_fault_types).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        # Prédiction par batch sur X_test_win
        X_test_tensor = torch.from_numpy(X_test_win.astype(np.float32)).to(device)
        batch_size = 64
        all_fault_pred, all_type_pred, all_sev_pred = [], [], []

        with torch.no_grad():
            for start in range(0, X_test_tensor.size(0), batch_size):
                end = min(start + batch_size, X_test_tensor.size(0))
                xb = X_test_tensor[start:end]
                logits_fault, logits_type, logits_sev = model(xb)
                all_fault_pred.append(logits_fault.argmax(dim=1).cpu().numpy())
                all_type_pred.append(logits_type.argmax(dim=1).cpu().numpy())
                all_sev_pred.append(logits_sev.argmax(dim=1).cpu().numpy())

        y_fault_pred_deep = np.concatenate(all_fault_pred)
        y_type_pred_deep = np.concatenate(all_type_pred)
        y_sev_pred_deep = np.concatenate(all_sev_pred)

        metrics_deep = compute_all_metrics(
            y_fault_test,
            y_type_test,
            y_sev_test,
            y_fault_pred_deep,
            y_type_pred_deep,
            y_sev_pred_deep,
        )
        all_results["Deep_MTL"] = metrics_deep
    else:
        print("[WARN] Modèle Deep MTL non trouvé, skip.")

    # =========================
    #  Affichage comparatif
    # =========================
    if not all_results:
        print("[ERROR] Aucun modèle évalué (aucun fichier trouvé). Vérifie les chemins dans models_dir.")
        return

    print_metrics_table(all_results)
    print_comb_confusions(all_results)


if __name__ == "__main__":
    main()
