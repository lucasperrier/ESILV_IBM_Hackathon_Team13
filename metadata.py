# data/metadata.py
from dataclasses import dataclass
from typing import Dict
import re


@dataclass
class FlightLabels:
    healthy: int        # 1 = sain, 0 = défaut
    fault_type: str     # "none", "crack", "edge_cut", "surface_cut", ...
    severity: int       # 0 (sain) à 3 (grave)
    trajectory: int     # id de trajectoire (1..5)
    drone_id: str       # id du drone (D1/D2/D3/Unknown)
    speed_case: int     # index de scénario de vitesse (SP1/SP2/…)


FAULT_ID_TO_TYPE = {
    0: "none",
    1: "crack",
    2: "edge_cut",
    3: "surface_cut",
}


def parse_filename(fname: str) -> FlightLabels:
    """
    Parse un nom de fichier DronePropA du type :
        F3_SV2_SP2_t5.mat
    ou éventuellement :
        D1_F0_SV0_SP1_t3.mat

    - F<id>   : type de défaut (0 = sain)
    - SV<k>   : sévérité (0..3)
    - SP<k>   : scénario de vitesse
    - t<k>    : trajectoire

    Si certains éléments sont absents, on met des valeurs par défaut.
    """

    # Extraire F, SV, SP, t via regex on ignore la partie "D1_" éventuelle au début
    m = re.search(r"F(\d+)_SV(\d+)_SP(\d+)_t(\d+)", fname)
    if not m:
        # fallback très simple si pattern inattendu on considère sain, severité 0, traj 0
        return FlightLabels(
            healthy=1,
            fault_type="none",
            severity=0,
            trajectory=0,
            drone_id=_infer_drone_id(fname),
            speed_case=0,
        )

    fault_id = int(m.group(1))
    sev_id = int(m.group(2))
    sp_id = int(m.group(3))
    traj_id = int(m.group(4))

    fault_type = FAULT_ID_TO_TYPE.get(fault_id, "other")

    # Healthy si F0
    healthy = 1 if fault_id == 0 else 0
    severity = 0 if healthy == 1 else sev_id

    drone_id = _infer_drone_id(fname)

    return FlightLabels(
        healthy=healthy,
        fault_type=fault_type,
        severity=severity,
        trajectory=traj_id,
        drone_id=drone_id,
        speed_case=sp_id,
    )


def _infer_drone_id(fname: str) -> str:
    """
    Déduit l'id du drone à partir du nom de fichier.
    Exemples : D1_, D2_, D3_...
    Si rien n'est trouvé, on renvoie 'Unknown'.
    """
    if "D1" in fname:
        return "D1"
    if "D2" in fname:
        return "D2"
    if "D3" in fname:
        return "D3"
    return "Unknown"


def get_labels_for_file(fname: str) -> Dict:
    labels = parse_filename(fname)
    return {
        "healthy": labels.healthy,
        "fault_type": labels.fault_type,
        "severity": labels.severity,
        "trajectory": labels.trajectory,
        "drone_id": labels.drone_id,
        "speed_case": labels.speed_case,
    }
