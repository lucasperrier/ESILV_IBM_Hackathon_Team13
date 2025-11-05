# data/loader.py
from pathlib import Path
from typing import Dict, List
import scipy.io as sio
import pandas as pd

from .metadata import get_labels_for_file


def load_flight(mat_path: Path) -> pd.DataFrame:
    """
    Charge un fichier .mat DronePropA au format :
      - QDrone_data    : (56, N)  -> données capteurs drone
      - commander_data : (37, N)  -> données de consigne/commande
      - stabilizer_data: (21, N)  -> états internes du contrôleur

    Convention utilisée :
      - Ligne 0 de chaque tableau = temps (en secondes)
      - Lignes suivantes = signaux, nommés de manière générique :
          q_1, q_2, ..., q_55
          cmd_1, cmd_2, ...
          stab_1, stab_2, ...
    """

    mat = sio.loadmat(mat_path)

    if "QDrone_data" not in mat:
        raise ValueError(f"QDrone_data non trouvé dans {mat_path}")

    q = mat["QDrone_data"]  # (56, N)
    time = q[0, :]          # première ligne = temps

    df_dict = {"time": time}

    # Signaux du drone (QDrone_data)
    for i in range(1, q.shape[0]):
        df_dict[f"q_{i}"] = q[i, :]

    # Commander data (si présent)
    c = mat.get("commander_data", None)
    if c is not None:
        # On suppose même longueur temporelle
        if c.shape[1] == time.shape[0]:
            for i in range(1, c.shape[0]):
                df_dict[f"cmd_{i}"] = c[i, :]
        else:
            # Longueur différente : on ignore par sécurité
            print(f"[WARN] commander_data longueur différente pour {mat_path}, ignoré.")

    # Stabilizer data (si présent)
    s = mat.get("stabilizer_data", None)
    if s is not None:
        if s.shape[1] == time.shape[0]:
            for i in range(1, s.shape[0]):
                df_dict[f"stab_{i}"] = s[i, :]
        else:
            print(f"[WARN] stabilizer_data longueur différente pour {mat_path}, ignoré.")

    df = pd.DataFrame(df_dict)
    return df


def load_all_flights(root_dir: Path) -> List[Dict]:
    """
    Parcourt root_dir, charge tous les .mat DronePropA,
    et associe les labels extraits du nom de fichier.
    Renvoie une liste de dicts :
        { "df": DataFrame, "labels": {...}, "file": Path }
    """
    flights = []
    mat_files = list(root_dir.glob("*.mat"))
    print(f"[INFO] {len(mat_files)} fichiers .mat trouvés dans {root_dir}")

    for mat_path in mat_files:
        df = load_flight(mat_path)
        labels = get_labels_for_file(mat_path.name)
        flights.append({"df": df, "labels": labels, "file": mat_path})

    return flights
