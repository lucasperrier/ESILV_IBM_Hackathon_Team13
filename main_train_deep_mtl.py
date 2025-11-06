# main_train_deep_mtl.py
from pathlib import Path
import numpy as np
import torch

from config import cfg
from training.train_mtl import main_train_mtl


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
    print(f"[INFO] y_fault.shape={y_fault.shape}, y_type.shape={y_type.shape}, y_sev.shape={y_sev.shape}")

    print("[INFO] Démarrage de l'entraînement deep MTL ...")
    model, val_metrics = main_train_mtl(X, y_fault, y_type, y_sev)

    # Sauvegarde du modèle + méta-données
    model_path = models_dir / "deep_mtl_cnn1d.pth"

    checkpoint = {
        "state_dict": model.state_dict(),
        "n_features": X.shape[2],
        "n_fault_types": int(y_type.max() + 1),
        "config": {
            "target_fs": cfg.data.target_fs,
            "win_sec": cfg.data.win_sec,
            "step_sec": cfg.data.step_sec,
            "batch_size": cfg.training.batch_size,
            "num_epochs": cfg.training.num_epochs,
            "lr": cfg.training.lr,
            "lambda_fault": cfg.training.weight_fault,
            "lambda_type": cfg.training.weight_type,
            "lambda_sev": cfg.training.weight_sev,
        },
        "val_metrics": val_metrics,
    }

    torch.save(checkpoint, model_path)
    print(f"[INFO] Modèle deep MTL sauvegardé dans {model_path}")
    print("[INFO] Métriques de validation sauvegardées avec le checkpoint.")


if __name__ == "__main__":
    main()
