# features/control_loop.py
import re
import numpy as np
import pandas as pd


def _compute_error_stats(err: pd.Series, base: str, threshold: float = 0.1) -> dict:
    """
    Calcule un petit set de stats sur une erreur de suivi (consigne - mesure).
    err : Series (valeurs de l'erreur)
    base : préfixe du nom de feature (ex: "err_cmd_1_q_1")
    threshold : seuil pour la fraction du temps au-dessus d'une erreur acceptable.
    """
    feats = {}
    err_abs = err.abs()

    feats[f"{base}_mean"] = err.mean()
    feats[f"{base}_std"] = err.std()
    feats[f"{base}_rms"] = np.sqrt((err**2).mean())
    feats[f"{base}_max_abs"] = err_abs.max()

    # fraction du temps où l'erreur dépasse le seuil
    feats[f"{base}_time_over_thr"] = (err_abs > threshold).mean()

    return feats


def _find_generic_cmd_measure_pairs(columns):
    """
    Trouve les paires génériques (cmd_i, q_i) à partir des noms de colonnes.
    Exemple :
      q_1, q_2, q_3, ...
      cmd_1, cmd_2, ...
    Renvoie une liste de tuples (cmd_col, meas_col).
    """
    cmd_pattern = re.compile(r"^cmd_(\d+)$")
    q_pattern = re.compile(r"^q_(\d+)$")

    cmd_indices = {}
    q_indices = {}

    for col in columns:
        m_cmd = cmd_pattern.match(col)
        if m_cmd:
            idx = int(m_cmd.group(1))
            cmd_indices[idx] = col
        m_q = q_pattern.match(col)
        if m_q:
            idx = int(m_q.group(1))
            q_indices[idx] = col

    # intersection des indices
    common = sorted(set(cmd_indices.keys()) & set(q_indices.keys()))

    pairs = []
    for idx in common:
        pairs.append((cmd_indices[idx], q_indices[idx]))

    return pairs


def _find_named_cmd_measure_pairs(columns):
    """
    Paires explicites pour des noms physiques, SI tu décides de renommer
    certaines colonnes (optionnel).
    Exemple :
      'roll_cmd' / 'roll'
      'pitch_cmd' / 'pitch'
      'yaw_cmd' / 'yaw'
      'alt_cmd' / 'alt'
    """
    candidate_pairs = [
        ("roll_cmd", "roll"),
        ("pitch_cmd", "pitch"),
        ("yaw_cmd", "yaw"),
        ("alt_cmd", "alt"),
    ]
    pairs = []
    for cmd_col, meas_col in candidate_pairs:
        if cmd_col in columns and meas_col in columns:
            pairs.append((cmd_col, meas_col))
    return pairs


def compute_control_loop_features(window_df: pd.DataFrame) -> dict:
    """
    Calcule des features de loop de contrôle pour une fenêtre de signaux.

    Fonctionne de deux façons :
    1) Si tu as des colonnes "physiques" type :
          roll_cmd / roll, pitch_cmd / pitch, ...
       => les erreurs correspondantes sont utilisées.

    2) Si tu n'as que les colonnes génériques sorties par loader.py :
          q_1..q_N, cmd_1..cmd_M
       => la fonction va automatiquement coupler cmd_i avec q_i
          et calculer les erreurs pour chaque i.

    Paramètres
    ----------
    window_df : pd.DataFrame
        DataFrame correspondant à une fenêtre temporelle (plusieurs lignes),
        avec au minimum :
          - une colonne "time"
          - des colonnes 'q_i' (mesures), 'cmd_i' (consignes) générées
            par data/loader.py.

    Retour
    ------
    feats : dict
        Dictionnaire de features scalaires.
    """
    feats = {}

    cols = list(window_df.columns)

    # 1) Paires nommées explicitement (si existantes)
    named_pairs = _find_named_cmd_measure_pairs(cols)
    for cmd_col, meas_col in named_pairs:
        err = window_df[cmd_col] - window_df[meas_col]
        base = f"err_{cmd_col}_{meas_col}"
        feats.update(_compute_error_stats(err, base, threshold=0.1))

    # 2) Paires génériques cmd_i / q_i
    generic_pairs = _find_generic_cmd_measure_pairs(cols)
    for cmd_col, meas_col in generic_pairs:
        err = window_df[cmd_col] - window_df[meas_col]
        base = f"err_{cmd_col}_{meas_col}"
        # Seuil d'erreur : heuristique basée sur l'amplitude du signal
        # (plus robuste que 0.1 fixe pour des grandeurs très petites/grandes)
        max_amp = max(window_df[cmd_col].abs().max(), window_df[meas_col].abs().max())
        thr = 0.05 * max_amp if max_amp > 1e-6 else 0.1  # 5% de l'amplitude ou 0.1 par défaut
        feats.update(_compute_error_stats(err, base, threshold=thr))

    return feats
