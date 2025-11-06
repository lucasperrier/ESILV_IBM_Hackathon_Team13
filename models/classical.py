# models/classical_xgb.py
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class ClassicalXGBModels:
    """Modèles XGBoost pour les 3 tâches (fault/type/sévérité)."""
    fault_clf: xgb.XGBClassifier
    type_clf: xgb.XGBClassifier
    severity_reg: xgb.XGBRegressor


def _to_numpy_float32(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.values.astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def _to_numpy_int32(y) -> np.ndarray:
    if isinstance(y, pd.Series):
        return y.values.astype(np.int32)
    return np.asarray(y, dtype=np.int32)


def train_xgb_models(
    X_train: pd.DataFrame,
    y_fault_train: np.ndarray,
    y_type_train: np.ndarray,
    y_sev_train: np.ndarray,
    use_gpu: bool = True,
) -> ClassicalXGBModels:
    """
    Entraîne 3 modèles XGBoost :
      - binaire : défaut / pas défaut
      - multi-classe : type de défaut
      - régression : sévérité (0..3)
    """
    X_np = _to_numpy_float32(X_train)
    y_fault_np = _to_numpy_int32(y_fault_train)
    y_type_np = _to_numpy_int32(y_type_train)
    y_sev_np = np.asarray(y_sev_train, dtype=np.float32)

    device = "cuda" if use_gpu else "cpu"

    # Détection de défaut
    fault_clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        tree_method="hist",  # ✅ nouvelle API
        device=device,
        eval_metric="logloss",
        n_jobs=0,
    )
    fault_clf.fit(X_np, y_fault_np)

    # Type de défaut (multi-classes)
    type_clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device=device,
        eval_metric="mlogloss",
        n_jobs=0,
    )
    type_clf.fit(X_np, y_type_np)

    # Sévérité (régression)
    severity_reg = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        device=device,
        n_jobs=0,
    )
    severity_reg.fit(X_np, y_sev_np)

    return ClassicalXGBModels(fault_clf, type_clf, severity_reg)


def predict_with_xgb(
    models: ClassicalXGBModels, X: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne proba défaut, type préd., sévérité préd."""
    X_np = _to_numpy_float32(X)
    prob_fault = models.fault_clf.predict_proba(X_np)[:, 1]
    type_pred = models.type_clf.predict(X_np)
    sev_pred = models.severity_reg.predict(X_np)
    return prob_fault, type_pred.astype(int), sev_pred.astype(float)

