# training/train_mtl.py
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from models.deep_mtl import CNNMTL, mtl_loss
from config import cfg


class WindowDataset(Dataset):
    """
    Dataset PyTorch pour les fenêtres temporelles + labels.
    """

    def __init__(
        self,
        X: np.ndarray,
        y_fault: np.ndarray,
        y_type: np.ndarray,
        y_sev: np.ndarray,
    ):
        self.X = X.astype(np.float32)
        self.y_fault = y_fault.astype(np.int64)
        self.y_type = y_type.astype(np.int64)
        self.y_sev = y_sev.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y_fault[idx],
            self.y_type[idx],
            self.y_sev[idx],
        )


def split_train_val(
    X: np.ndarray,
    y_fault: np.ndarray,
    y_type: np.ndarray,
    y_sev: np.ndarray,
    test_size: float = 0.2,
):
    idx_train, idx_val = train_test_split(
        np.arange(len(X)), test_size=test_size, stratify=y_fault
    )
    def sel(a): return a[idx_train], a[idx_val]

    y_fault_tr, y_fault_va = sel(y_fault)
    y_type_tr, y_type_va = sel(y_type)
    y_sev_tr, y_sev_va = sel(y_sev)
    X_tr, X_va = X[idx_train], X[idx_val]

    return X_tr, X_va, y_fault_tr, y_fault_va, y_type_tr, y_type_va, y_sev_tr, y_sev_va


def train_one_epoch(
    model: CNNMTL,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    lambdas: Dict[str, float],
):
    model.train()
    total_loss = 0.0
    for X, y_fault, y_type, y_sev in loader:
        X = X.to(device)
        y_fault = y_fault.to(device)
        y_type = y_type.to(device)
        y_sev = y_sev.to(device)

        optimizer.zero_grad()
        logits_fault, logits_type, logits_sev = model(X)
        loss, _ = mtl_loss(
            logits_fault,
            logits_type,
            logits_sev,
            y_fault,
            y_type,
            y_sev,
            lambda_fault=lambdas["fault"],
            lambda_type=lambdas["type"],
            lambda_sev=lambdas["sev"],
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


def evaluate(
    model: CNNMTL,
    loader: DataLoader,
    device: str,
):
    model.eval()
    loss_sum = 0.0
    correct_fault = 0
    total = 0
    with torch.no_grad():
        for X, y_fault, y_type, y_sev in loader:
            X = X.to(device)
            y_fault = y_fault.to(device)
            y_type = y_type.to(device)
            y_sev = y_sev.to(device)

            logits_fault, logits_type, logits_sev = model(X)
            loss, _ = mtl_loss(logits_fault, logits_type, logits_sev, y_fault, y_type, y_sev)
            loss_sum += loss.item() * X.size(0)

            pred_fault = logits_fault.argmax(dim=1)
            correct_fault += (pred_fault == y_fault).sum().item()
            total += y_fault.size(0)

    return {
        "loss": loss_sum / total,
        "fault_acc": correct_fault / total,
    }


def main_train_mtl(X: np.ndarray, y_fault: np.ndarray, y_type: np.ndarray, y_sev: np.ndarray):
    
    device = cfg.training.device if torch.cuda.is_available() else "cpu"

    (
        X_tr,
        X_va,
        y_fault_tr,
        y_fault_va,
        y_type_tr,
        y_type_va,
        y_sev_tr,
        y_sev_va,
    ) = split_train_val(X, y_fault, y_type, y_sev)

    train_ds = WindowDataset(X_tr, y_fault_tr, y_type_tr, y_sev_tr)
    val_ds = WindowDataset(X_va, y_fault_va, y_type_va, y_sev_va)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False)

    n_feat = X.shape[2]
    n_types = int(y_type.max() + 1)
    model = CNNMTL(in_channels=n_feat, n_fault_types=n_types).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    lambdas = {
        "fault": cfg.training.weight_fault,
        "type": cfg.training.weight_type,
        "sev": cfg.training.weight_sev,
    }

    for epoch in range(cfg.training.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, lambdas)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"[Epoch {epoch+1}/{cfg.training.num_epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"fault_acc={val_metrics['fault_acc']:.3f} | "
            f"fault_f1={val_metrics['fault_f1']:.3f} | "
            f"type_acc={val_metrics['type_acc']:.3f} | "
            f"sev_MAE={val_metrics['sev_mae']:.3f}"
        )

        # Suivi du meilleur modèle (plus petite loss de validation)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = model.state_dict()
            best_metrics = val_metrics

    # On recharge le meilleur état
    if best_state is not None:
        model.load_state_dict(best_state)

    # Et on ré-affiche un résumé des meilleures métriques
    print("\n[INFO] Meilleures métriques de validation (deep MTL) :")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")

    return model, best_metrics

