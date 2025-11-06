# models/deep_mtl.py
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMTL(nn.Module):
    """
    Modèle CNN 1D multi-tâches : détection défaut, type, sévérité.
    Entrée : (batch, seq_len, n_feat)
    """

    def __init__(self, in_channels: int, n_fault_types: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head_fault = nn.Linear(256, 2)
        self.head_type = nn.Linear(256, n_fault_types)
        self.head_sev = nn.Linear(256, 4)  # 4 niveaux de sévérité 0..3

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, n_feat) -> (batch, n_feat, seq_len)
        x = x.transpose(1, 2)
        h = self.backbone(x).squeeze(-1)
        out_fault = self.head_fault(h)
        out_type = self.head_type(h)
        out_sev = self.head_sev(h)
        return out_fault, out_type, out_sev


def mtl_loss(
    logits_fault,
    logits_type,
    logits_sev,
    y_fault,
    y_type,
    y_sev,
    lambda_fault: float = 2.0,
    lambda_type: float = 1.0,
    lambda_sev: float = 1.0,
):
    """
    Combinaison de pertes pour multi-tâches.
    y_fault : labels 0/1
    y_type : labels [0..N_types-1]
    y_sev : labels [0..3]
    """
    loss_fault = F.cross_entropy(logits_fault, y_fault)
    loss_type = F.cross_entropy(logits_type, y_type)
    loss_sev = F.cross_entropy(logits_sev, y_sev)  # ou MSE si régression

    loss = lambda_fault * loss_fault + lambda_type * loss_type + lambda_sev * loss_sev
    return loss, {"fault": loss_fault.item(), "type": loss_type.item(), "sev": loss_sev.item()}
