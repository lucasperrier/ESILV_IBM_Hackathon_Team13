from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from config import cfg
from features.time_domain import extract_time_features_from_windows
from features.freq_domain import extract_freq_features_from_windows
from models.classical import predict_with_xgb

# === Chargement du modèle ===
model_file = Path("models/classical_xgb_timefreq_gpu.joblib")
saved = load(model_file)
models = saved["models"]
feature_columns = saved["feature_columns"]

print("[OK] Modèle chargé.")
processed_dir = cfg.data.processed_dir

X = np.load(processed_dir / "X_windows.npy")
y_fault = np.load(processed_dir / "y_fault.npy")
y_type = np.load(processed_dir / "y_type.npy")
y_sev = np.load(processed_dir / "y_sev.npy")

df_time = extract_time_features_from_windows(X)
df_freq = extract_freq_features_from_windows(X, fs=cfg.data.target_fs)
X_feat = pd.concat([df_time, df_freq], axis=1)[feature_columns]
from sklearn.model_selection import train_test_split

X_train, X_test, y_fault_train, y_fault_test, y_type_train, y_type_test, y_sev_train, y_sev_test = train_test_split(
    X_feat, y_fault, y_type, y_sev,
    test_size=0.2, random_state=42, stratify=y_fault
)
prob_fault_test, type_pred_test, sev_pred_test = predict_with_xgb(models, X_test)
print("\n=== MATRICE DE CONFUSION : TYPE DE DEFAUT ===")
cm_type = confusion_matrix(y_type_test, type_pred_test)
print(cm_type)

print("\n=== REPORT ===")
print(classification_report(y_type_test, type_pred_test))
sev_pred_levels = np.clip(np.round(sev_pred_test), 0, 3).astype(int)

print("\n=== MATRICE DE CONFUSION : SEVERITE (CLASSES 0..3) ===")
cm_sev = confusion_matrix(y_sev_test, sev_pred_levels)
print(cm_sev)

print("\n=== REPORT ===")
print(classification_report(y_sev_test, sev_pred_levels))
