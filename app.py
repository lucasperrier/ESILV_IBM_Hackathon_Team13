# app.py
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from joblib import load

from config import cfg
from data.loader import load_flight
from data.preprocessing import (
    resample_signal,
    normalize_df,
    sliding_windows,
)
from features.time_domain import extract_time_features_from_windows
from features.freq_domain import extract_freq_features_from_windows
from models.classical import predict_with_xgb

# Masquer le warning "precision loss" des moments statistiques
warnings.filterwarnings(
    "ignore",
    message="Precision loss occurred in moment calculation*",
    category=RuntimeWarning,
)

# =========================
#  UTILITAIRES
# =========================

@st.cache_resource
def load_xgb_model():
    model_path = Path("models/classical_xgb_timefreq_gpu.joblib")
    if not model_path.exists():
        st.error(f"⚠️ Modèle XGBoost non trouvé : {model_path}. Entraîne-le avec main_train_xgb_gpu.py.")
        return None, None

    meta = load(model_path)
    models = meta["models"]
    feature_columns = meta["feature_columns"]

    # 🔧 Forcer les 3 modèles XGBoost à tourner sur CPU
    def _force_cpu(m):
        try:
            # nouvelle API : device / predictor
            m.set_params(device="cpu", predictor="cpu_predictor")
        except TypeError:
            # vieille version d'xgboost sans param `device`
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
        st.error(f"⚠️ Fichier de stats de normalisation non trouvé : {stats_path}. Lance main_prepare_data.py.")
        return None, None, None

    data = np.load(stats_path, allow_pickle=True)
    mean = pd.Series(data["mean"], index=data["columns"])
    std = pd.Series(data["std"], index=data["columns"])
    cols = list(data["columns"])
    return mean, std, cols


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

    # 🆕 Ajout du label texte
    results["Fault_Type_Name"] = results["Fault_Type"].apply(decode_fault_type)

    return results



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


def render_flight_overview(df: pd.DataFrame):
    st.subheader("📈 Flight signals overview")

    if df.empty:
        st.write("No data to show.")
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
    page_title="TrackUAVFault - Predictive Maintenance",
    layout="wide",
    page_icon="🛰️",
)

st.title("🛠️ TrackUAVFault – Predictive Maintenance for Drones")

st.markdown("""
Interface de démonstration de la **maintenance prédictive pour drones** :
- chargement d’un vol (fichier `.mat` DronePropA ou démo),
- analyse des signaux,
- détection de défaut, type et sévérité,
- monitoring simulé pour ta vidéo de présentation.
""")


models, feature_columns = load_xgb_model()
if models is None:
    st.stop()


st.sidebar.header("🎛️ Configuration de la démo")

mode = st.sidebar.radio(
    "Mode d'utilisation",
    ["Demo flight (données déjà prétraitées)", "Upload .mat DronePropA"],
)

simulate = st.sidebar.checkbox("Activer le mode simulation", value=True)
sim_speed = st.sidebar.slider("Vitesse de simulation (sec entre fenêtres)", 0.05, 0.5, 0.15, 0.05)


df_raw = None
X_windows = None
t_windows = None

# -------- Mode upload .mat --------
if mode == "Upload .mat DronePropA":
    st.sidebar.subheader("📂 Upload")
    uploaded_file = st.sidebar.file_uploader("Choisir un fichier .mat", type=["mat"])

    if uploaded_file is not None:
        st.sidebar.success("Fichier chargé ✅")

        tmp_path = Path("tmp_uploaded.mat")
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        df_raw = load_flight(tmp_path)

        st.success("Vol chargé et converti en DataFrame ✅")
        render_flight_overview(df_raw)

        st.subheader("⚙️ Prétraitement du vol")
        with st.spinner("Resampling, normalisation et fenêtrage..."):
            X_windows, t_windows = preprocess_flight_df(df_raw)

        st.write(f"Nombre de fenêtres générées : **{X_windows.shape[0]}**")
        if X_windows.shape[0] == 0:
            st.warning("Aucune fenêtre utilisable après prétraitement.")

# -------- Mode demo flight --------
else:
    st.sidebar.subheader("🧪 Demo")
    st.sidebar.info("Utilisation d'un vol de démonstration à partir des données prétraitées.")
    X_windows, t_windows = get_demo_flight()
    st.subheader("📈 Demo flight (synthetic timeline)")
    st.line_chart(
        pd.DataFrame({
            "time": t_windows,
            "severity_demo": np.zeros_like(t_windows)
        }).set_index("time")
    )
    df_raw = pd.DataFrame({"time": t_windows, "q_1": np.zeros_like(t_windows)})


if X_windows is None or X_windows.shape[0] == 0:
    st.info("En attente d’un vol pour lancer la prédiction...")
    st.stop()


# =========================
#  Prédictions pré-calculées
# =========================

with st.spinner("Pré-calcul des prédictions sur toutes les fenêtres..."):
    results_all = predict_on_windows(X_windows, models, feature_columns)


# =========================
#  Analyse globale
# =========================

st.markdown("---")
st.header("🧠 Predictive maintenance analysis")

if st.button("Lancer l'analyse du vol complet"):
    results = results_all

    if results.empty:
        st.warning("Aucun résultat de prédiction généré.")
    else:
        st.subheader("🔍 Résumé global")

        avg_fault = results["Fault_Prob"].mean()
        frac_fault = results["Fault_Label"].mean()
        avg_sev = results["Severity_Level"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Probabilité moyenne de défaut", f"{avg_fault:.2f}")
        col2.metric("Temps en état défectueux", f"{100 * frac_fault:.1f} %")
        col3.metric("Sévérité moyenne (0–3)", f"{avg_sev:.2f}")

        # 📊 Distribution des niveaux de sévérité (labels verticaux)
        st.subheader("📊 Distribution des niveaux de sévérité prédits")
        sev_counts = results["Severity_Level"].value_counts().sort_index()

        fig, ax = plt.subplots()
        ax.bar(sev_counts.index.astype(str), sev_counts.values)
        ax.set_xlabel("Severity level")
        ax.set_ylabel("Count")

    

        # Pour que tout rentre bien dans la figure
        fig.tight_layout()

        st.pyplot(fig)


        # Timeline sévérité + probabilité
        if t_windows is not None and len(t_windows) == len(results):
            st.subheader("🕒 Sévérité & probabilité de défaut dans le temps")
            df_timeline = pd.DataFrame({
                "time": t_windows,
                "Severity_Level": results["Severity_Level"].values,
                "Fault_Prob": results["Fault_Prob"].values,
            }).set_index("time")
            st.line_chart(df_timeline[["Severity_Level", "Fault_Prob"]])

        st.subheader("🧾 Tableau des prédictions (premières fenêtres)")
        cols_to_show = ["Fault_Prob", "Fault_Label", "Fault_Type_Name", "Severity_Level"]
        st.dataframe(results[cols_to_show].head(30))



# =========================
#  Simulation
# =========================

st.markdown("---")
st.header("🎬 Simulation de vol & monitoring en temps réel")

st.write("""
Ce mode simule la surveillance du drone en temps réel :
à chaque fenêtre de temps, les indicateurs de santé sont mis à jour.
Idéal à enregistrer pour ta vidéo de démo.
""")

if simulate and st.button("Démarrer la simulation"):
    placeholder_header = st.empty()
    placeholder_metrics = st.empty()
    placeholder_plot = st.empty()

    n_win = X_windows.shape[0]
    max_steps = min(n_win, 150)

    for i in range(max_steps):
        row = results_all.iloc[i]
        fault_prob = float(row["Fault_Prob"])
        sev_level = int(row["Severity_Level"])
        fault_label = int(row["Fault_Label"])
        fault_type_name = str(row.get("Fault_Type_Name", "Unknown"))
        t_cur = (
            t_windows[i]
            if t_windows is not None and len(t_windows) > i
            else i * cfg.data.step_sec
        )

        state_str = "FAULT" if fault_label == 1 else "NORMAL"
        state_color = "🔴" if fault_label == 1 else "🟢"

        # 🆕 Afficher aussi le type de défaut (si FAULT)
        if fault_label == 1:
            fault_desc = f"— Type : **{fault_type_name}**"
        else:
            fault_desc = ""

        placeholder_header.markdown(
            f"### {state_color} t = {t_cur:.2f} s — État : **{state_str}**, "
            f"sévérité prédite = **{sev_level} / 3** {fault_desc}"
        )

        with placeholder_metrics.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("Probabilité de défaut", f"{fault_prob:.2f}")
            c2.metric("Niveau de sévérité", f"{sev_level} / 3")
            c3.metric("Type de défaut", fault_type_name)

        # Graphique temps réel
        if t_windows is not None and len(t_windows) >= i + 1:
            df_sim = pd.DataFrame({
                "time": t_windows[: i + 1],
                "Severity_Level": results_all["Severity_Level"].values[: i + 1],
            }).set_index("time")

            with placeholder_plot.container():
                st.line_chart(df_sim)

        time.sleep(sim_speed)


    st.success("Simulation terminée ✅ – prête à être filmée pour ta vidéo !")
