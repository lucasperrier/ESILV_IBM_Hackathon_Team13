# models/knn_cuda.py
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch


def encode_combined_labels(
    y_fault: np.ndarray,
    y_type: np.ndarray,
    y_sev: np.ndarray,
) -> np.ndarray:
    """
    Combine (fault_label, fault_type, severity) en un seul entier.
    Convention : code = fault*100 + type*10 + sev

    Ex:
      (0,0,0) -> 0          (drone sain)
      (1,2,3) -> 123
    """
    y_fault = y_fault.astype(int)
    y_type = y_type.astype(int)
    y_sev = y_sev.astype(int)
    code = y_fault * 100 + y_type * 10 + y_sev
    return code


def decode_combined_labels(code: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inverse de encode_combined_labels.
    """
    code = code.astype(int)
    y_fault = code // 100
    y_type = (code // 10) % 10
    y_sev = code % 10
    return y_fault, y_type, y_sev


@dataclass
class CudaKNNCombined:
    """
    KNN implémenté à la main avec PyTorch sur GPU (CUDA).

    - X_train : tensor (N, D) sur device (cuda)
    - y_train : tensor (N,) de labels combinés (int64)
    """
    k: int = 5
    device: str = "cuda"

    def __post_init__(self):
        self.X_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None
        self.mean_: torch.Tensor | None = None
        self.std_: torch.Tensor | None = None

    # ---------- FIT ----------

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y_fault: np.ndarray,
        y_type: np.ndarray,
        y_sev: np.ndarray,
    ):
        """
        Enregistre les données d'entraînement sur GPU.
        - X : (N, D)
        - y_* : (N,)
        """
        # Convertir X en numpy float32
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float32)
            self.feature_names_ = list(X.columns)
        else:
            X_np = np.asarray(X, dtype=np.float32)
            self.feature_names_ = None

        # Encode des labels combinés
        y_combined = encode_combined_labels(y_fault, y_type, y_sev)

        # Normalisation simple (z-score) sur les features
        mean = X_np.mean(axis=0)
        std = X_np.std(axis=0)
        std_safe = std.copy()
        std_safe[std_safe == 0] = 1.0

        X_norm = (X_np - mean) / std_safe

        # Nettoyage NaN/Inf (au cas où)
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Passage en torch sur device
        dev = self.device if torch.cuda.is_available() and "cuda" in self.device else "cpu"

        self.X_train = torch.from_numpy(X_norm).to(dev)
        self.y_train = torch.from_numpy(y_combined.astype(np.int64)).to(dev)
        self.mean_ = torch.from_numpy(mean.astype(np.float32)).to(dev)
        self.std_ = torch.from_numpy(std_safe.astype(np.float32)).to(dev)
        self.device = dev

        return self

    # ---------- PREDICT (labels combinés) ----------

    def _prepare_X(self, X: pd.DataFrame | np.ndarray) -> torch.Tensor:
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)

        # Normalisation avec le mean/std du train
        mean = self.mean_.cpu().numpy()
        std = self.std_.cpu().numpy()
        X_norm = (X_np - mean) / std
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

        dev = self.device
        return torch.from_numpy(X_norm).to(dev)

    @torch.no_grad()
    def predict_combined(self, X: pd.DataFrame | np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        Prédiction des labels combinés pour X, par KNN sur GPU.
        """
        assert self.X_train is not None, "Le modèle KNN n'a pas été fit."
        assert self.y_train is not None, "Le modèle KNN n'a pas été fit."

        X_tensor = self._prepare_X(X)
        N = X_tensor.size(0)
        preds = []

        # On traite par batch pour éviter d'exploser la mémoire GPU
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X_tensor[start:end]  # (B, D)

            # Distances euclidiennes (carrées) sur GPU
            # torch.cdist est supporté et va utiliser CUDA
            dists = torch.cdist(X_batch, self.X_train, p=2)  # (B, N_train)

            # Top-k plus proches voisins
            knn_dists, knn_idx = torch.topk(dists, k=min(self.k, self.X_train.size(0)), largest=False, dim=1)

            # Labels des voisins
            knn_labels = self.y_train[knn_idx]  # (B, k)

            # Vote majoritaire
            # torch.mode marche bien ici
            mode_labels, _ = torch.mode(knn_labels, dim=1)
            preds.append(mode_labels.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        return preds

    # ---------- PREDICT triplet (fault/type/sev) ----------

    def predict_triplet(
        self,
        X: pd.DataFrame | np.ndarray,
        batch_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retourne (y_fault_pred, y_type_pred, y_sev_pred).
        """
        y_combined_pred = self.predict_combined(X, batch_size=batch_size)
        y_fault_pred, y_type_pred, y_sev_pred = decode_combined_labels(y_combined_pred)
        return y_fault_pred, y_type_pred, y_sev_pred

    # ---------- SAVE / LOAD ----------

    def to_state_dict(self) -> Dict[str, Any]:
        """
        Prépare un dict sérialisable (pour torch.save ou joblib).
        On repasse tout sur CPU pour éviter les soucis device.
        """
        return {
            "k": self.k,
            "X_train": self.X_train.cpu().numpy(),
            "y_train": self.y_train.cpu().numpy(),
            "mean": self.mean_.cpu().numpy(),
            "std": self.std_.cpu().numpy(),
            "feature_names": self.feature_names_,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any], device: str = "cuda") -> "CudaKNNCombined":
        dev = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        obj = cls(k=state["k"], device=dev)

        obj.X_train = torch.from_numpy(state["X_train"].astype(np.float32)).to(dev)
        obj.y_train = torch.from_numpy(state["y_train"].astype(np.int64)).to(dev)
        obj.mean_ = torch.from_numpy(state["mean"].astype(np.float32)).to(dev)
        obj.std_ = torch.from_numpy(state["std"].astype(np.float32)).to(dev)
        obj.feature_names_ = state.get("feature_names", None)

        return obj
