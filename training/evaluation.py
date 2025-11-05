# training/evaluation.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
)


def evaluate_fault_detection(y_true: np.ndarray, y_prob_fault: np.ndarray, thr: float = 0.5):
    y_pred = (y_prob_fault >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def evaluate_fault_type(y_true: np.ndarray, y_pred: np.ndarray):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "conf_mat": confusion_matrix(y_true, y_pred),
    }


def evaluate_severity(y_true: np.ndarray, y_pred: np.ndarray):
    y_pred_rounded = np.round(y_pred).astype(int)
    y_pred_rounded = np.clip(y_pred_rounded, 0, 3)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mae_levels": mean_absolute_error(y_true, y_pred_rounded),
    }
