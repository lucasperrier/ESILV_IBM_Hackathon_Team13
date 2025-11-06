# models/xgboost.py
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class XGBoostModels:
    fault_clf: XGBClassifier
    type_clf: XGBClassifier
    severity_reg: XGBRegressor


def train_xgboost_models(
    X_train: pd.DataFrame,
    y_fault_train: np.ndarray,
    y_type_train: np.ndarray,
    y_sev_train: np.ndarray,
) -> XGBoostModels:
    """
    Entraîne trois modèles XGBoost : détection défaut, type défaut, sévérité.
    """

    fault_clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=None,   # Tu peux calculer le ratio si classes déséquilibrées
        tree_method="hist"
    )
    fault_clf.fit(X_train, y_fault_train)

    type_clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method="hist"
    )
    type_clf.fit(X_train, y_type_train)

    severity_reg = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist"
    )
    severity_reg.fit(X_train, y_sev_train)

    return XGBoostModels(
        fault_clf=fault_clf, type_clf=type_clf, severity_reg=severity_reg
    )


def predict_with_xgboost(
    models: XGBoostModels, X: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Renvoie : prob_fault, type_pred, sev_pred
    """
    prob_fault = models.fault_clf.predict_proba(X)[:, 1]
    type_pred = models.type_clf.predict(X)
    sev_pred = models.severity_reg.predict(X)
    return prob_fault, type_pred, sev_pred