# models/classical.py
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@dataclass
class ClassicalModels:
    fault_clf: RandomForestClassifier
    type_clf: RandomForestClassifier
    severity_reg: RandomForestRegressor


def train_classical_models(
    X_train: pd.DataFrame,
    y_fault_train: np.ndarray,
    y_type_train: np.ndarray,
    y_sev_train: np.ndarray,
) -> ClassicalModels:
    """
    Entraîne trois modèles : détection défaut, type défaut, sévérité.
    """
    fault_clf = RandomForestClassifier(n_estimators=200, class_weight="balanced")
    fault_clf.fit(X_train, y_fault_train)

    type_clf = RandomForestClassifier(n_estimators=200, class_weight="balanced")
    type_clf.fit(X_train, y_type_train)

    severity_reg = RandomForestRegressor(n_estimators=200)
    severity_reg.fit(X_train, y_sev_train)

    return ClassicalModels(
        fault_clf=fault_clf, type_clf=type_clf, severity_reg=severity_reg
    )


def predict_with_classical(
    models: ClassicalModels, X: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Renvoie : prob_fault, type_pred, sev_pred
    """
    prob_fault = models.fault_clf.predict_proba(X)[:, 1]
    type_pred = models.type_clf.predict(X)
    sev_pred = models.severity_reg.predict(X)
    return prob_fault, type_pred, sev_pred
