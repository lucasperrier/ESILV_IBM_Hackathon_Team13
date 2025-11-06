# models/knn_combined.py
"""
Modèle KNN avec labels combinés.
Fournit des fonctions pour entraîner et faire des prédictions avec KNN.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from typing import Tuple, Dict


class KNNCombinedModel:
    """
    Wrapper pour le modèle KNN avec labels combinés.
    """
    
    def __init__(self, n_neighbors: int = 5):
        """
        Args:
            n_neighbors: nombre de voisins pour KNN
        """
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        self.label_mapping = None
        self.reverse_mapping = None
    
    def combine_labels(self, y_fault, y_type, y_sev):
        """
        Combine les 3 arrays de labels en un seul label unique.
        """
        combinations = np.column_stack([y_fault, y_type, y_sev])
        unique_combinations = np.unique(combinations, axis=0)
        
        self.label_mapping = {}
        self.reverse_mapping = {}
        
        for idx, (f, t, s) in enumerate(unique_combinations):
            self.label_mapping[tuple([int(f), int(t), int(s)])] = idx
            self.reverse_mapping[idx] = tuple([int(f), int(t), int(s)])
        
        y_combined = np.array([
            self.label_mapping[tuple([int(f), int(t), int(s)])] 
            for f, t, s in combinations
        ])
        
        return y_combined
    
    def flatten_windows(self, X_windows):
        """
        Aplatit les fenêtres 3D en 2D.
        """
        n_samples = X_windows.shape[0]
        return X_windows.reshape(n_samples, -1)
    
    def fit(self, X, y_fault, y_type, y_sev):
        """
        Entraîne le modèle KNN.
        
        Args:
            X: features, shape (n_samples, window_size, n_features) ou (n_samples, n_features)
            y_fault: labels de défaut
            y_type: labels de type
            y_sev: labels de sévérité
        """
        # Aplatir si nécessaire
        if len(X.shape) == 3:
            X_flat = self.flatten_windows(X)
        else:
            X_flat = X
        
        # Combiner les labels
        y_combined = self.combine_labels(y_fault, y_type, y_sev)
        
        # Entraîner le modèle
        self.model.fit(X_flat, y_combined)
        
        return self
    
    def predict(self, X):
        """
        Prédit les labels combinés.
        
        Args:
            X: features
        
        Returns:
            y_pred: labels combinés prédits
        """
        if len(X.shape) == 3:
            X_flat = self.flatten_windows(X)
        else:
            X_flat = X
        
        return self.model.predict(X_flat)
    
    def predict_separate(self, X):
        """
        Prédit et décompose en (fault, type, sev).
        
        Args:
            X: features
        
        Returns:
            y_fault_pred, y_type_pred, y_sev_pred: arrays séparés
        """
        y_combined_pred = self.predict(X)
        
        y_fault_pred = np.array([self.reverse_mapping[y][0] for y in y_combined_pred])
        y_type_pred = np.array([self.reverse_mapping[y][1] for y in y_combined_pred])
        y_sev_pred = np.array([self.reverse_mapping[y][2] for y in y_combined_pred])
        
        return y_fault_pred, y_type_pred, y_sev_pred
    
    def predict_proba(self, X):
        """
        Prédit les probabilités pour chaque classe combinée.
        """
        if len(X.shape) == 3:
            X_flat = self.flatten_windows(X)
        else:
            X_flat = X
        
        return self.model.predict_proba(X_flat)
    
    def score(self, X, y_fault, y_type, y_sev):
        """
        Calcule l'accuracy du modèle.
        """
        if len(X.shape) == 3:
            X_flat = self.flatten_windows(X)
        else:
            X_flat = X
        
        y_combined = self.combine_labels(y_fault, y_type, y_sev)
        return self.model.score(X_flat, y_combined)
    
    def cross_validate(self, X, y_fault, y_type, y_sev, cv=5):
        """
        Validation croisée.
        """
        if len(X.shape) == 3:
            X_flat = self.flatten_windows(X)
        else:
            X_flat = X
        
        y_combined = self.combine_labels(y_fault, y_type, y_sev)
        scores = cross_val_score(self.model, X_flat, y_combined, cv=cv, n_jobs=-1)
        
        return scores
    
    def save(self, path: Path):
        """
        Sauvegarde le modèle et les mappings.
        """
        data = {
            "model": self.model,
            "n_neighbors": self.n_neighbors,
            "label_mapping": self.label_mapping,
            "reverse_mapping": self.reverse_mapping
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: Path):
        """
        Charge un modèle sauvegardé.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        instance = cls(n_neighbors=data["n_neighbors"])
        instance.model = data["model"]
        instance.label_mapping = data["label_mapping"]
        instance.reverse_mapping = data["reverse_mapping"]
        
        return instance


def train_knn_combined(
    X: np.ndarray,
    y_fault: np.ndarray,
    y_type: np.ndarray,
    y_sev: np.ndarray,
    n_neighbors: int = 5
) -> KNNCombinedModel:
    """
    Fonction helper pour entraîner rapidement un modèle KNN.
    
    Args:
        X: features
        y_fault: labels de défaut
        y_type: labels de type
        y_sev: labels de sévérité
        n_neighbors: nombre de voisins
    
    Returns:
        model: KNNCombinedModel entraîné
    """
    model = KNNCombinedModel(n_neighbors=n_neighbors)
    model.fit(X, y_fault, y_type, y_sev)
    
    return model
