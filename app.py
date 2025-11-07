# app.py
import time
import warnings
from pathlib import Path
import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from joblib import load
from PIL import Image
import plotly.graph_objects as go

from config import cfg
from data.loader import load_flight
from data.metadata import parse_filename
from data.preprocessing import (
    resample_signal,
    normalize_df,
    sliding_windows,
)
from features.time_domain import extract_time_features_from_windows
from features.freq_domain import extract_freq_features_from_windows
from models.classical import predict_with_xgb  # garde ton import actuel

# Masquer le warning "precision loss" des moments statistiques
warnings.filterwarnings(
    "ignore",
    message="Precision loss occurred in moment calculation*",
    category=RuntimeWarning,
)

# ========================================
#  Mapping des codes de défaut → noms lisibles
# ========================================
FAULT_TYPE_NAMES = {
    0: "none",
    1: "crack",
    2: "edge_cut",
    3: "surface_cut",
}


def decode_fault_type(code):
    """Retourne le nom du type de défaut à partir du code."""
    try:
        return FAULT_TYPE_NAMES.get(int(code), f"Unknown ({code})")
    except Exception:
        return f"Invalid ({code})"


# =========================
#  UTILITAIRES MODÈLES
# =========================

@st.cache_resource
def load_xgb_model():
    model_path = Path("models/classical_xgb_timefreq_gpu.joblib")
    if not model_path.exists():
        st.error(f"XGBoost model not found: {model_path}. Train it with main_train_xgb_gpu.py.")
        return None, None

    meta = load(model_path)
    models = meta["models"]
    feature_columns = meta["feature_columns"]

    # 🔧 Forcer les 3 modèles XGBoost à tourner sur CPU (évite les soucis GPU+Streamlit)
    def _force_cpu(m):
        try:
            m.set_params(device="cpu", predictor="cpu_predictor")
        except TypeError:
            try:
                m.set_params(predictor="cpu_predictor")
            except Exception:
                pass
        try:
            booster = m.get_booster()
            booster.set_param({"device": "cpu", "predictor": "cpu_predictor"})
        except Exception:
            pass

    _force_cpu(models.fault_clf)
    _force_cpu(models.type_clf)
    _force_cpu(models.severity_reg)

    return models, feature_columns


@st.cache_resource
def load_normalization_stats():
    stats_path = cfg.data.processed_dir / "normalization_stats.npz"
    if not stats_path.exists():
        st.error(f"Normalization stats file not found: {stats_path}. Run main_prepare_data.py.")
        return None, None, None

    data = np.load(stats_path, allow_pickle=True)
    mean = pd.Series(data["mean"], index=data["columns"])
    std = pd.Series(data["std"], index=data["columns"])
    cols = list(data["columns"])
    return mean, std, cols


# =========================
#  VISUELS DRONE / TRAJECTOIRE
# =========================

FAULT_PARTS = {
    1: "Front motor",
    2: "Left arm",
    3: "Right arm",
}


def render_drone_3d(fault_group=None, severity=0):
    """Crée un drone 3D stylisé avec la partie endommagée en rouge (non utilisé mais dispo si besoin)."""
    colors = ["lightgrey"] * 4
    labels = ["Front motor", "Left arm", "Right arm", "Body"]

    parts = {1: "Front motor", 2: "Left arm", 3: "Right arm"}
    if fault_group in parts:
        idx = list(parts.keys()).index(fault_group)
        colors[idx] = f"rgba(255,0,0,{0.3 + 0.2*severity})"

    fig = go.Figure(data=[
        go.Scatter3d(
            x=[0, 1, 0, -1], y=[1, 0, -1, 0], z=[0, 0, 0, 0],
            mode='markers+text',
            marker=dict(size=10, color=colors),
            text=labels,
            textposition="top center"
        )
    ])
    fig.update_layout(
        width=400, height=400,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )
    return fig


def compute_trajectory_xy(df_raw: pd.DataFrame, t_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule une trajectoire XY :
      - si df_raw contient x,y -> on les réutilise (échantillonnés)
      - sinon -> on génère un joli trajet fictif en forme de boucle
    """
    n = len(t_windows)
    if df_raw is not None and "x" in df_raw.columns and "y" in df_raw.columns:
        x_raw = df_raw["x"].values
        y_raw = df_raw["y"].values
        if len(x_raw) >= n:
            x = x_raw[:n]
            y = y_raw[:n]
        else:
            idx = np.linspace(0, len(x_raw) - 1, n).astype(int)
            x = x_raw[idx]
            y = y_raw[idx]
    else:
        # Trajectoire fictive : boucle légèrement déformée
        t_norm = np.linspace(0, 1, n)
        theta = 2 * np.pi * t_norm
        radius = 1.0 + 0.3 * np.sin(3 * theta)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    return x, y


def compute_trajectory_xyz(
    df_raw: pd.DataFrame,
    t_windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule une trajectoire 3D (X,Y,Z) alignée sur t_windows :
      - si df_raw contient x,y,z -> on les réutilise (ré-échantillonnage simple)
      - si x,y seulement -> on les réutilise et on génère Z synthétique
      - sinon -> on génère X,Y,Z fictifs (boucle en XY + ondulation en Z)
    """
    n = len(t_windows)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    def _resample_to_n(arr: np.ndarray, n_out: int) -> np.ndarray:
        if len(arr) >= n_out:
            return arr[:n_out]
        idx = np.linspace(0, len(arr) - 1, n_out).astype(int)
        return arr[idx]

    has_x = df_raw is not None and ("x" in df_raw.columns)
    has_y = df_raw is not None and ("y" in df_raw.columns)
    has_z = df_raw is not None and ("z" in df_raw.columns)

    if has_x and has_y and has_z:
        x = _resample_to_n(df_raw["x"].values, n)
        y = _resample_to_n(df_raw["y"].values, n)
        z = _resample_to_n(df_raw["z"].values, n)
        return x, y, z
    elif has_x and has_y:
        x = _resample_to_n(df_raw["x"].values, n)
        y = _resample_to_n(df_raw["y"].values, n)
        t_norm = np.linspace(0, 1, n)
        z = 0.3 * np.sin(2 * np.pi * 1.2 * t_norm) + 0.1 * np.sin(2 * np.pi * 0.4 * t_norm)
        return x, y, z
    else:
        t_norm = np.linspace(0, 1, n)
        theta = 2 * np.pi * t_norm
        radius = 1.0 + 0.3 * np.sin(3 * theta)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 0.3 * np.sin(2 * np.pi * 1.2 * t_norm) + 0.1 * np.sin(2 * np.pi * 0.4 * t_norm)
        return x, y, z


def generate_synthetic_trajectory(
    n: int,
    shape: str = "lemniscate",
    scale: float = 1.5,
    loops: float = 1.0,
    noise: float = 0.02,
    rotation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, n)
    theta = 2 * np.pi * loops * t

    if shape == "circle":
        r = np.ones_like(theta) * scale
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    elif shape == "lemniscate":
        a = scale
        x = a * np.sin(theta)
        y = a * np.sin(theta) * np.cos(theta)
    elif shape == "square":
        p = (loops * t) % 1.0
        x = np.zeros_like(p)
        y = np.zeros_like(p)
        for i, u in enumerate(p):
            u4 = u * 4
            if u4 < 1:
                x[i] = -1 + 2 * u4
                y[i] = -1
            elif u4 < 2:
                x[i] = 1
                y[i] = -1 + 2 * (u4 - 1)
            elif u4 < 3:
                x[i] = 1 - 2 * (u4 - 2)
                y[i] = 1
            else:
                x[i] = -1
                y[i] = 1 - 2 * (u4 - 3)
        x *= scale
        y *= scale
    elif shape == "spiral":
        r = (0.4 + 0.6 * t) * scale
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    else:  # lissajous
        a = max(1.0, loops)
        b = a + 1.0
        x = scale * np.sin(a * theta)
        y = scale * np.sin(b * theta + np.pi / 2)

    if rotation_deg:
        ang = np.deg2rad(rotation_deg)
        xr = x * np.cos(ang) - y * np.sin(ang)
        yr = x * np.sin(ang) + y * np.cos(ang)
        x, y = xr, yr

    if noise and noise > 0:
        rng = np.random.default_rng(123)
        x = x + noise * rng.standard_normal(size=n)
        y = y + noise * rng.standard_normal(size=n)

    # Z synthétique (ondulation)
    z = 0.3 * np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.sin(2 * np.pi * 0.4 * t)

    return x, y, z


# =========================
#  PIPELINE DE PREPROCESS / PREDICTION
# =========================

def preprocess_flight_df(df_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample -> normalisation -> fenêtrage.
    Retourne :
      - X_windows : (n_win, win_len, n_feat)
      - t_windows : temps (sec) au centre de chaque fenêtre
    """
    mean, std, cols_stats = load_normalization_stats()
    if mean is None:
        return np.empty((0,)), np.empty((0,))

    df_rs = resample_signal(df_raw, target_fs=cfg.data.target_fs)

    cols = ["time"] + [c for c in df_rs.columns if c != "time" and c in cols_stats]
    df_rs = df_rs[cols]

    mean_s = mean[df_rs.columns[df_rs.columns != "time"]]
    std_s = std[df_rs.columns[df_rs.columns != "time"]]

    df_norm = normalize_df(df_rs, mean_s, std_s)

    X = sliding_windows(
        df_norm,
        win_sec=cfg.data.win_sec,
        step_sec=cfg.data.step_sec,
        fs=cfg.data.target_fs,
    )

    if len(df_norm) > 0 and X.shape[0] > 0:
        t = df_norm["time"].values
        win_len = int(cfg.data.win_sec * cfg.data.target_fs)
        step = int(cfg.data.step_sec * cfg.data.target_fs)
        centers = []
        for start in range(0, len(df_norm) - win_len, step):
            mid_idx = start + win_len // 2
            centers.append(t[mid_idx])
        t_windows = np.array(centers)
    else:
        t_windows = np.empty((0,))

    return X, t_windows


def predict_on_windows(X_windows: np.ndarray, models, feature_columns):
    """
    Features temps+freq + prédiction XGBoost sur un lot de fenêtres.
    """
    if X_windows.shape[0] == 0:
        return pd.DataFrame()

    df_time = extract_time_features_from_windows(X_windows)
    df_freq = extract_freq_features_from_windows(X_windows, fs=cfg.data.target_fs)
    X_feat_all = pd.concat([df_time, df_freq], axis=1)

    X_feat = X_feat_all[feature_columns]

    prob_fault, type_pred, sev_pred = predict_with_xgb(models, X_feat)

    results = pd.DataFrame({
        "Fault_Prob": prob_fault,
        "Fault_Label": (prob_fault >= 0.5).astype(int),
        "Fault_Type": type_pred,
        "Severity_Continuous": sev_pred,
        "Severity_Level": np.clip(np.round(sev_pred), 0, 3).astype(int),
    })

    # Ajout du label texte lisible
    results["Fault_Type_Name"] = results["Fault_Type"].apply(decode_fault_type)

    return results


def render_flight_overview(df: pd.DataFrame):
    st.subheader("Flight signals overview")

    if df.empty:
        st.write("Aucune donnée à afficher.")
        return

    sensor_cols = [c for c in df.columns if c.startswith("q_")][:3]
    if not sensor_cols:
        sensor_cols = [c for c in df.columns if c != "time"][:3]

    fig, ax = plt.subplots(figsize=(8, 3))
    for c in sensor_cols:
        ax.plot(df["time"].values, df[c].values, label=c, alpha=0.7)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Signal")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def get_demo_flight():
    """
    Utilise X_windows.npy comme vol de démo (fenêtres déjà prêtes).
    """
    X_all = np.load(cfg.data.processed_dir / "X_windows.npy")
    n = min(200, X_all.shape[0])
    X_demo = X_all[:n]
    t_demo = np.arange(n) * cfg.data.step_sec
    return X_demo, t_demo


# =========================
#  INTERFACE STREAMLIT
# =========================

st.set_page_config(
    page_title="TrackUAVFault - Maintenance prédictive",
    layout="wide"
)

st.title("TrackUAVFault - Predictive Maintenance for Drones")

st.markdown("""
Interface de démonstration de la maintenance prédictive pour drones :
- chargement d’un vol (fichier `.mat` DronePropA ou démo),
- analyse des signaux,
- détection de défaut, type et sévérité,
- monitoring simulé pour la vidéo de présentation.
""")

models, feature_columns = load_xgb_model()
if models is None:
    st.stop()

st.sidebar.header("Demo Configuration")

mode = st.sidebar.radio(
    "Mode",
    ["Demo flight (preprocessed data)", "Upload .mat DronePropA"],
)

simulate = st.sidebar.checkbox("Activer le mode simulation", value=True)
sim_speed = st.sidebar.slider("Vitesse de simulation (sec entre fenêtres)", 0.05, 0.5, 0.15, 0.05)

df_raw = None
X_windows = None
t_windows = None

# Variables pour trajectoire XY
uploaded_file_name = None
traj_id_guess = None
synth_traj_opts = None

# Flag pour trajet/séquence fictifs
use_synth = False
synth_fault_ratio = 0.3
synth_seed = 42

# -------- Mode upload .mat --------
if mode == "Upload .mat DronePropA":
    st.sidebar.subheader("Upload")
    uploaded_file = st.sidebar.file_uploader("Choisir un fichier .mat", type=["mat"])

    if uploaded_file is not None:
        st.sidebar.success("Fichier chargé ✅")

        tmp_path = Path("tmp_uploaded.mat")
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        df_raw = load_flight(tmp_path)
        uploaded_file_name = uploaded_file.name
        try:
            traj_id_guess = parse_filename(uploaded_file_name).trajectory
        except Exception:
            traj_id_guess = None
        st.success("Flight loaded and converted to DataFrame.")
        render_flight_overview(df_raw)
        st.subheader("Flight preprocessing")
        with st.spinner("Resampling, normalisation et fenêtrage..."):
            X_windows, t_windows = preprocess_flight_df(df_raw)

        st.write(f"Nombre de fenêtres générées : **{X_windows.shape[0]}**")
        if X_windows.shape[0] == 0:
            st.warning("No usable window after preprocessing.")

# -------- Mode demo flight --------
else:
    st.sidebar.subheader("Demo")
    st.sidebar.info("Using a demo flight from preprocessed data.")
    X_windows, t_windows = get_demo_flight()

    st.subheader("Trajectoire fictive 3D")
    df_raw = pd.DataFrame({"time": t_windows, "q_1": np.zeros_like(t_windows)})
    # --- Trajectoire fictive XYZ (configurable) ---
    st.sidebar.markdown("### Trajectoire fictive (XYZ)")
    synth_shape = st.sidebar.selectbox(
        "Forme",
        ["lemniscate", "circle", "square", "spiral", "lissajous"],
        index=0,
    )
    synth_scale = st.sidebar.slider("Echelle", 0.5, 5.0, 1.5, 0.1)
    synth_loops = st.sidebar.slider("Boucles", 0.5, 3.0, 1.0, 0.1)
    synth_noise = st.sidebar.slider("Bruit XY", 0.0, 0.2, 0.02, 0.01)
    synth_rot = st.sidebar.slider("Rotation (deg)", -180, 180, 0)

    # Aperçu 3D de la trajectoire fictive (réactif aux réglages)
    if t_windows is not None and len(t_windows) > 0:
        demo_x, demo_y, demo_z = generate_synthetic_trajectory(
            len(t_windows),
            shape=synth_shape,
            scale=float(synth_scale),
            loops=float(synth_loops),
            noise=float(synth_noise),
            rotation_deg=float(synth_rot),
        )
        fig_demo = go.Figure()
        fig_demo.add_trace(
            go.Scatter3d(
                x=demo_x, y=demo_y, z=demo_z,
                mode="lines+markers",
                line=dict(color="#cccccc", width=3),
                marker=dict(size=2, color="#1f77b4", opacity=0.7),
                name="Trajectoire"
            )
        )
        fig_demo.update_layout(
            height=420,
            margin=dict(l=0, r=0, b=0, t=10),
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        )
        st.plotly_chart(fig_demo, use_container_width=True)
    synth_traj_opts = {
        "shape": synth_shape,
        "scale": float(synth_scale),
        "loops": float(synth_loops),
        "noise": float(synth_noise),
        "rotation_deg": float(synth_rot),
    }

    # --- Trajet fictif de fautes / types / sévérités ---
    # Mode démo: override automatique (pas de case à cocher)
    use_synth = True
    synth_fault_ratio = st.sidebar.slider("Proportion de fenêtres en défaut", 0.0, 1.0, 0.3, 0.05)
    synth_seed = st.sidebar.number_input("Seed aléatoire", 0, 10000, 42)

if X_windows is None or X_windows.shape[0] == 0:
    st.info("Waiting for a flight to start prediction...")
    st.stop()

# =========================
#  Trajectoire XY (global)
# =========================
def generate_synthetic_trajectory(
    n: int,
    shape: str = "lemniscate",
    scale: float = 1.5,
    loops: float = 1.0,
    noise: float = 0.02,
    rotation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, n)
    theta = 2 * np.pi * loops * t

    if shape == "circle":
        r = np.ones_like(theta) * scale
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    elif shape == "lemniscate":
        a = scale
        x = a * np.sin(theta)
        y = a * np.sin(theta) * np.cos(theta)
    elif shape == "square":
        p = (loops * t) % 1.0
        x = np.zeros_like(p)
        y = np.zeros_like(p)
        for i, u in enumerate(p):
            u4 = u * 4
            if u4 < 1:
                x[i] = -1 + 2 * u4
                y[i] = -1
            elif u4 < 2:
                x[i] = 1
                y[i] = -1 + 2 * (u4 - 1)
            elif u4 < 3:
                x[i] = 1 - 2 * (u4 - 2)
                y[i] = 1
            else:
                x[i] = -1
                y[i] = 1 - 2 * (u4 - 3)
        x *= scale
        y *= scale
    elif shape == "spiral":
        r = (0.4 + 0.6 * t) * scale
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    else:  # lissajous
        a = max(1.0, loops)
        b = a + 1.0
        x = scale * np.sin(a * theta)
        y = scale * np.sin(b * theta + np.pi / 2)

    if rotation_deg:
        ang = np.deg2rad(rotation_deg)
        xr = x * np.cos(ang) - y * np.sin(ang)
        yr = x * np.sin(ang) + y * np.cos(ang)
        x, y = xr, yr

    if noise and noise > 0:
        rng = np.random.default_rng(123)
        x = x + noise * rng.standard_normal(size=n)
        y = y + noise * rng.standard_normal(size=n)

    # Z synthétique (ondulation)
    z = 0.3 * np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.sin(2 * np.pi * 0.4 * t)

    return x, y, z


traj_x, traj_y, traj_z = compute_trajectory_xyz(df_raw, t_windows)
# Override XY with dataset-adapted or demo-configured synthetic path
n_pts = len(t_windows) if t_windows is not None else 0
if n_pts > 0:
    if traj_id_guess is not None:
        _map = {1: "circle", 2: "lemniscate", 3: "square", 4: "spiral", 5: "lissajous"}
        _shape = _map.get(int(traj_id_guess), "lemniscate")
        _opts = synth_traj_opts or {}
        traj_x, traj_y, traj_z = generate_synthetic_trajectory(
            n_pts,
            shape=_shape,
            scale=float(_opts.get("scale", 1.5)),
            loops=float(_opts.get("loops", 1.0)),
            noise=float(_opts.get("noise", 0.02)),
            rotation_deg=float(_opts.get("rotation_deg", 0.0)),
        )
    elif synth_traj_opts is not None:
        traj_x, traj_y, traj_z = generate_synthetic_trajectory(n_pts, **synth_traj_opts)

# =========================
#  Prédictions pré-calculées
# =========================
with st.spinner("Resampling, normalisation et fenêtrage..."):
    results_all = predict_on_windows(X_windows, models, feature_columns)

# Override par un trajet fictif si demandé (demo mode)
if mode.startswith("Demo flight"):
    n = len(results_all)
    rng = np.random.default_rng(int(synth_seed))

    fault_labels = (rng.random(n) < synth_fault_ratio).astype(int)
    fault_types = np.zeros(n, dtype=int)
    severities = np.zeros(n, dtype=int)

    mask_fault = fault_labels == 1
    if mask_fault.any():
        fault_types[mask_fault] = rng.integers(1, len(FAULT_TYPE_NAMES), size=mask_fault.sum())
        severities[mask_fault] = rng.integers(1, 4, size=mask_fault.sum())

    fault_probs = np.where(
        fault_labels == 1,
        0.7 + 0.3 * rng.random(n),
        0.3 * rng.random(n),
    )

    results_all["Fault_Label"] = fault_labels
    results_all["Fault_Type"] = fault_types
    results_all["Severity_Level"] = severities
    results_all["Severity_Continuous"] = severities.astype(float)
    results_all["Fault_Prob"] = fault_probs
    results_all["Fault_Type_Name"] = results_all["Fault_Type"].apply(decode_fault_type)


# =========================
#  Analyse globale
# =========================

st.markdown("---")
st.header("Predictive maintenance analysis")

if st.button("Lancer l'analyse du vol complet"):
    results = results_all

    if results.empty:
        st.warning("No prediction results generated.")
    else:
        st.subheader("Global summary")

        avg_fault = results["Fault_Prob"].mean()
        frac_fault = results["Fault_Label"].mean()
        avg_sev = results["Severity_Level"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Probabilité moyenne de défaut", f"{avg_fault:.2f}")
        col2.metric("Temps en état défectueux", f"{100 * frac_fault:.1f} %")
        col3.metric("Sévérité moyenne (0–3)", f"{avg_sev:.2f}")

        # 📊 Distribution des niveaux de sévérité (labels verticaux)
        st.subheader("Distribution of predicted severity levels")
        sev_counts = results["Severity_Level"].value_counts().sort_index()

        fig, ax = plt.subplots()
        ax.bar(sev_counts.index.astype(str), sev_counts.values)
        ax.set_xlabel("Severity level")
        ax.set_ylabel("Count")
        fig.tight_layout()
        st.pyplot(fig)

        # Timeline sévérité + probabilité + fault_label
        if t_windows is not None and len(t_windows) == len(results):
            st.subheader("🕒 Sévérité, probabilité & label de défaut dans le temps")
            df_timeline = pd.DataFrame({
                "time": t_windows,
                "Severity_Level": results["Severity_Level"].values,
                "Fault_Prob": results["Fault_Prob"].values,
                "Fault_Label": results["Fault_Label"].values.astype(float),
            }).set_index("time")
            st.line_chart(df_timeline[["Severity_Level", "Fault_Prob"]])

        # 🛠️ Trajectoire du drone globale
        st.subheader("Drone trajectory (fault segments highlighted)")
        n_points = min(len(traj_x), len(results))
        fault_mask = results["Fault_Label"].values[:n_points] == 1
        sev_vals = results["Severity_Level"].values[:n_points]

        fig_traj_3d = go.Figure()
        # Trajectoire globale
        fig_traj_3d.add_trace(
            go.Scatter3d(
                x=traj_x[:n_points], y=traj_y[:n_points], z=traj_z[:n_points],
                mode="lines",
                line=dict(color="#d3d3d3", width=4),
                name="Trajectoire"
            )
        )
        # Points normaux
        if np.any(~fault_mask):
            fig_traj_3d.add_trace(
                go.Scatter3d(
                    x=traj_x[:n_points][~fault_mask],
                    y=traj_y[:n_points][~fault_mask],
                    z=traj_z[:n_points][~fault_mask],
                    mode="markers",
                    marker=dict(size=3, color="#1f77b4", opacity=0.8),
                    name="Normal",
                )
            )
        # Points en défaut colorés par sévérité
        if np.any(fault_mask):
            fig_traj_3d.add_trace(
                go.Scatter3d(
                    x=traj_x[:n_points][fault_mask],
                    y=traj_y[:n_points][fault_mask],
                    z=traj_z[:n_points][fault_mask],
                    mode="markers",
                    marker=dict(
                        size=5 + 2 * sev_vals[fault_mask],
                        color=sev_vals[fault_mask],
                        colorscale="Reds",
                        cmin=0,
                        cmax=3,
                        showscale=True,
                        colorbar=dict(title="Severity"),
                    ),
                    name="Fault",
                )
            )

        fig_traj_3d.update_layout(
            height=520,
            margin=dict(l=0, r=0, b=0, t=10),
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
            ),
        )
        st.plotly_chart(fig_traj_3d, use_container_width=True)

        # Tableau des prédictions
        st.subheader("Predictions table (first windows)")
        cols_to_show = ["Fault_Prob", "Fault_Label", "Fault_Type_Name", "Severity_Level"]
        st.dataframe(results[cols_to_show].head(30))


# =========================
#  Simulation
# =========================

st.markdown("---")
st.header("Flight simulation & real-time monitoring")

st.write("""
This mode simulates real-time drone monitoring:
At each time window, health indicators update.
Ideal to record for your demo video.
""")
if simulate and st.button("Démarrer la simulation"):
    placeholder_header = st.empty()
    placeholder_metrics = st.empty()
    placeholder_plot = st.empty()
    placeholder_traj = st.empty()  # Trajectoire XY

    n_win = X_windows.shape[0]
    max_steps = min(n_win, 150)

    for i in range(max_steps):
        row = results_all.iloc[i]
        fault_prob = float(row["Fault_Prob"])
        sev_level = int(row["Severity_Level"])
        fault_label = int(row["Fault_Label"])
        fault_type_name = str(row.get("Fault_Type_Name", "Unknown"))
        fault_group = int(row["Fault_Type"]) if fault_label == 1 else 0
        t_cur = t_windows[i] if t_windows is not None and len(t_windows) > i else i * cfg.data.step_sec

        # --- Header + image pivotée ---
        with placeholder_header.container():
            st.markdown(
                f"### {'🔴' if fault_label else '🟢'} t = {t_cur:.2f}s — "
                f"État : **{'FAULT' if fault_label else 'NORMAL'}**, "
                f"sévérité = **{sev_level}/3**"
                + (f" — Type : **{fault_type_name}**" if fault_label else "")
            )

            img_filename = f"images/F{fault_group}SV{sev_level}.png"
            if os.path.exists(img_filename):
                img = Image.open(img_filename)
                img = img.rotate(90, expand=True)
                st.image(img, width=360)
            else:
                st.write(f"Image manquante : {img_filename}")

        # --- Metrics ---
        with placeholder_metrics.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("Fault probability", f"{fault_prob:.2f}")
            c2.metric("Severity level", f"{sev_level}/3")
            c3.metric("Fault type", fault_type_name)

        # --- Timeline temps réel (même style que l'analyse globale) ---
        df_sim = pd.DataFrame({
            "time": t_windows[: i + 1],
            "Severity_Level": results_all["Severity_Level"].values[: i + 1],
            "Fault_Prob": results_all["Fault_Prob"].values[: i + 1],
            "Fault_Label": results_all["Fault_Label"].values[: i + 1].astype(float),
        }).set_index("time")
        placeholder_plot.line_chart(df_sim[["Severity_Level", "Fault_Prob"]])

        # --- Trajectoire XY (plus jolie) ---
        n_points = i + 1
        fault_mask = results_all["Fault_Label"].values[:n_points] == 1
        sev_vals = results_all["Severity_Level"].values[:n_points]

        # 2D supprimée; affichage uniquement en 3D ci-dessous

        # Affichage 3D de la trajectoire en simulation (override visuel)
        fig_s3d = go.Figure()
        fig_s3d.add_trace(
            go.Scatter3d(
                x=traj_x[:n_points], y=traj_y[:n_points], z=traj_z[:n_points],
                mode="lines",
                line=dict(color="#d3d3d3", width=3),
                name="Trajectoire",
            )
        )
        if np.any(~fault_mask):
            fig_s3d.add_trace(
                go.Scatter3d(
                    x=traj_x[:n_points][~fault_mask], y=traj_y[:n_points][~fault_mask], z=traj_z[:n_points][~fault_mask],
                    mode="markers",
                    marker=dict(size=3, color="#1f77b4", opacity=0.8),
                    name="Normal",
                )
            )
        if np.any(fault_mask):
            fig_s3d.add_trace(
                go.Scatter3d(
                    x=traj_x[:n_points][fault_mask], y=traj_y[:n_points][fault_mask], z=traj_z[:n_points][fault_mask],
                    mode="markers",
                    marker=dict(size=5 + 2 * sev_vals[fault_mask], color=sev_vals[fault_mask], colorscale="Reds", cmin=0, cmax=3),
                    name="Fault",
                )
            )
        fig_s3d.add_trace(
            go.Scatter3d(
                x=[traj_x[i]], y=[traj_y[i]], z=[traj_z[i]],
                mode="markers",
                marker=dict(size=8, color="yellow", line=dict(color="black", width=2)),
                name="Position actuelle",
            )
        )
        fig_s3d.update_layout(
            height=360,
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        )
        placeholder_traj.plotly_chart(fig_s3d, use_container_width=True)

        time.sleep(sim_speed)
    st.success("Simulation completed. Ready to be recorded for your video!")


# =========================
#  Explorer un instant précis
# =========================

st.markdown("---")
st.header("Explore a flight instant")

if not results_all.empty:
    # Sélection par temps t (s) pour plus de clarté
    if t_windows is not None and len(t_windows) > 0:
        t_min = float(t_windows[0])
        t_max = float(t_windows[-1])
        t_sel = st.slider("Sélectionner un instant t (s)", t_min, t_max, t_min)
        idx = int(np.argmin(np.abs(t_windows - t_sel)))
    row = results_all.iloc[idx]
    t_cur = t_windows[idx] if t_windows is not None and len(t_windows) > idx else idx * cfg.data.step_sec

    fault_prob = float(row["Fault_Prob"])
    sev_level = int(row["Severity_Level"])
    fault_label = int(row["Fault_Label"])
    fault_type_name = str(row.get("Fault_Type_Name", "Unknown"))

    st.markdown(
        f"### {'🔴' if fault_label else '🟢'} t = {t_cur:.2f}s — "
        f"État : **{'FAULT' if fault_label else 'NORMAL'}**, "
        f"sévérité = **{sev_level}/3**"
        + (f" — Type : **{fault_type_name}**" if fault_label else "")
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Fault probability", f"{fault_prob:.2f}")
    c2.metric("Severity level", f"{sev_level}/3")
    c3.metric("Fault type", fault_type_name)

    # Timeline jusqu'à cet instant
    df_view = pd.DataFrame({
        "time": t_windows[: idx + 1],
        "Severity_Level": results_all["Severity_Level"].values[: idx + 1],
        "Fault_Prob": results_all["Fault_Prob"].values[: idx + 1],
        "Fault_Label": results_all["Fault_Label"].values[: idx + 1].astype(float),
    }).set_index("time")
    st.line_chart(df_view[["Severity_Level", "Fault_Prob"]])

    # Trajectoire avec point courant
    n_points = len(traj_x)
    fault_mask = results_all["Fault_Label"].values[:n_points] == 1
    sev_vals = results_all["Severity_Level"].values[:n_points]

    # Affichage 3D de la trajectoire au même instant
    fig_v3d = go.Figure()
    fig_v3d.add_trace(
        go.Scatter3d(
            x=traj_x, y=traj_y, z=traj_z,
            mode="lines",
            line=dict(color="#d3d3d3", width=4),
            name="Trajectoire",
        )
    )
    # Points en faute colorés par sévérité
    if np.any(fault_mask):
        fig_v3d.add_trace(
            go.Scatter3d(
                x=traj_x[fault_mask], y=traj_y[fault_mask], z=traj_z[fault_mask],
                mode="markers",
                marker=dict(size=5 + 2 * sev_vals[fault_mask], color=sev_vals[fault_mask], colorscale="Reds", cmin=0, cmax=3),
                name="Fault",
            )
        )
    # Position sélectionnée
    fig_v3d.add_trace(
        go.Scatter3d(
            x=[traj_x[idx]], y=[traj_y[idx]], z=[traj_z[idx]],
            mode="markers",
            marker=dict(size=10, color="yellow", line=dict(color="black", width=2)),
            name="Position sélectionnée",
        )
    )
    fig_v3d.update_layout(
        height=420,
        margin=dict(l=0, r=0, b=0, t=10),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
    )
    st.plotly_chart(fig_v3d, use_container_width=True)
