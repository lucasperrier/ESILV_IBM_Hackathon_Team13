# main_prepare_data.py
"""
Pipeline de préparation des données :
- charge tous les vols (.mat)
- resample à une fréquence fixe
- calcule mean/std globales
- normalise
- découpe en fenêtres
- construit les labels (fault / type / severity)
- sauvegarde le tout en .npy dans data/processed
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from config import cfg
from data.loader import load_all_flights
from data.preprocessing import (
    resample_signal,
    compute_global_mean_std,
    normalize_df,
    sliding_windows,
)


def main():
    raw_dir = cfg.data.raw_dir
    processed_dir = cfg.data.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Chargement des vols depuis {raw_dir} ...")
    flights = load_all_flights(raw_dir)
    print(f"[INFO] {len(flights)} fichiers .mat chargés")

    # 1) Resample tous les vols à target_fs
    print("[INFO] Resampling de tous les vols ...")
    resampled_dfs = []
    for f in flights:
        df = f["df"]
        df_rs = resample_signal(df, target_fs=cfg.data.target_fs)
        resampled_dfs.append(df_rs)
        f["df_rs"] = df_rs  # on stocke aussi dans la structure flights

    # 2) Calcul mean/std globale (hors colonne time)
    print("[INFO] Calcul des statistiques globales (mean/std) ...")
    mean, std = compute_global_mean_std(resampled_dfs)

    # Sauvegarde des stats pour une réutilisation éventuelle
    stats_path = processed_dir / "normalization_stats.npz"
    np.savez(stats_path, mean=mean.values, std=std.values, columns=mean.index.values)
    print(f"[INFO] Statistiques de normalisation sauvegardées dans {stats_path}")

    # 3) Normalisation + fenêtrage pour chaque vol
    print("[INFO] Fenêtrage et construction du dataset de fenêtres ...")
    all_windows = []
    all_fault_labels = []
    all_type_labels = []
    all_sev_labels = []

    # pour garder la trace du mapping type défaut -> id
    fault_type_to_id = {}
    next_fault_type_id = 0

    for idx, f in enumerate(flights):
        df_rs = f["df_rs"]
        labels = f["labels"]  # dict: healthy, fault_type, severity, ...

        # Normalisation
        # On reconstruit mean/std en Series alignées sur les colonnes de df_rs (sauf time)
        cols = [c for c in df_rs.columns if c != "time"]
        mean_s = pd.Series(mean[cols])
        std_s = pd.Series(std[cols])
        df_norm = normalize_df(df_rs[["time"] + cols], mean_s, std_s)

        # Fenêtrage
        X_flight = sliding_windows(
            df_norm,
            win_sec=cfg.data.win_sec,
            step_sec=cfg.data.step_sec,
            fs=cfg.data.target_fs,
        )  # (n_win, win_len, n_feat)

        n_win = X_flight.shape[0]
        if n_win == 0:
            continue

        # Labels pour ce vol (identiques sur toutes les fenêtres du vol)
        healthy = labels["healthy"]  # 1 / 0
        fault_type = labels["fault_type"]  # "none", "crack", ...
        severity = labels["severity"]      # 0..3

        # Label binaire défaut
        y_fault_flight = np.zeros(n_win, dtype=np.int64)
        y_fault_flight[:] = 0 if healthy == 1 else 1

        # Mapping type -> id
        if fault_type not in fault_type_to_id:
            fault_type_to_id[fault_type] = next_fault_type_id
            next_fault_type_id += 1
        type_id = fault_type_to_id[fault_type]

        y_type_flight = np.full(n_win, type_id, dtype=np.int64)
        y_sev_flight = np.full(n_win, severity, dtype=np.int64)

        all_windows.append(X_flight)
        all_fault_labels.append(y_fault_flight)
        all_type_labels.append(y_type_flight)
        all_sev_labels.append(y_sev_flight)

        print(f"[INFO] Vol {idx+1}/{len(flights)} : {n_win} fenêtres")

    if not all_windows:
        print("[ERREUR] Aucune fenêtre générée. Vérifie win_sec/step_sec et la longueur des vols.")
        return

    # 4) Concaténation sur tous les vols
    X = np.concatenate(all_windows, axis=0)
    y_fault = np.concatenate(all_fault_labels, axis=0)
    y_type = np.concatenate(all_type_labels, axis=0)
    y_sev = np.concatenate(all_sev_labels, axis=0)

    print(f"[INFO] Dataset global : X.shape={X.shape}")
    print(f"       y_fault shape={y_fault.shape}, y_type shape={y_type.shape}, y_sev shape={y_sev.shape}")
    

    print("[CHECK] Avant nettoyage : NaN dans X_windows ? ", np.isnan(X).any())

    # 5) Sauvegarde
    np.save(processed_dir / "X_windows.npy", X)
    np.save(processed_dir / "y_fault.npy", y_fault)
    np.save(processed_dir / "y_type.npy", y_type)
    np.save(processed_dir / "y_sev.npy", y_sev)

    # mapping type -> id (json)
    mapping_path = processed_dir / "fault_type_mapping.json"
    with open(mapping_path, "w") as f_json:
        json.dump(fault_type_to_id, f_json, indent=2)

    print(f"[INFO] Sauvegardes effectuées dans {processed_dir}")
    print(f"       fault_type_to_id = {fault_type_to_id}")


if __name__ == "__main__":
    main()
