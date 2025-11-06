# create_combined_y.py
"""
Combine les trois fichiers y (y_fault, y_type, y_sev) en un seul fichier y_combined.
Chaque combinaison unique reçoit un label séquentiel.
"""

import numpy as np
import json
from pathlib import Path


def combine_y_labels(y_fault, y_type, y_sev, save_dir=None):
    """
    Combine trois arrays de labels en un seul array avec mapping.
    
    Args:
        y_fault: array (n,) avec valeurs binaires 0/1
        y_type: array (n,) avec valeurs 0-3
        y_sev: array (n,) avec valeurs 0-3
        save_dir: Path où sauvegarder le mapping (optionnel)
    
    Returns:
        y_combined: array (n,) avec labels combinés séquentiels
        mapping: dict (fault, type, sev) -> label_id
        reverse_mapping: dict label_id -> (fault, type, sev)
    """
    print("[INFO] Combinaison des labels...")
    print(f"  y_fault: {y_fault.shape}, unique: {np.unique(y_fault)}")
    print(f"  y_type: {y_type.shape}, unique: {np.unique(y_type)}")
    print(f"  y_sev: {y_sev.shape}, unique: {np.unique(y_sev)}")
    
    # Créer un array de tuples (fault, type, sev)
    combinations = np.column_stack([y_fault, y_type, y_sev])
    
    # Trouver toutes les combinaisons uniques dans l'ordre d'apparition
    unique_combinations = []
    seen = set()
    
    for combo in combinations:
        combo_tuple = tuple(combo.astype(int))
        if combo_tuple not in seen:
            unique_combinations.append(combo_tuple)
            seen.add(combo_tuple)
    
    # Créer le mapping: (fault, type, sev) -> label_id
    mapping = {}
    reverse_mapping = {}
    
    for label_id, combo in enumerate(unique_combinations):
        mapping[combo] = label_id
        reverse_mapping[label_id] = combo
    
    # Créer le label combiné pour chaque échantillon
    y_combined = np.array([
        mapping[tuple(combo.astype(int))] 
        for combo in combinations
    ])
    
    print(f"\n[INFO] Nombre de combinaisons uniques trouvées: {len(mapping)}")
    print("\n[INFO] Mapping des combinaisons (fault, type, sev) -> label:")
    print("-" * 60)
    
    for label_id in sorted(reverse_mapping.keys()):
        fault, type_, sev = reverse_mapping[label_id]
        count = np.sum(y_combined == label_id)
        description = "No Fault" if fault == 0 else f"Fault (type={type_}, sev={sev})"
        print(f"  ({fault}, {type_}, {sev}) -> {label_id:2d}  |  {description:30s} |  {count:5d} samples")
    
    # Sauvegarder le mapping si demandé
    if save_dir:
        save_dir = Path(save_dir)
        
        # Sauvegarder en JSON (converti en strings et int pour JSON)
        mapping_json = {str(k): int(v) for k, v in mapping.items()}
        reverse_mapping_json = {str(k): [int(x) for x in v] for k, v in reverse_mapping.items()}
        
        mapping_path = save_dir / "y_combined_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump({
                "mapping": mapping_json,
                "reverse_mapping": reverse_mapping_json,
                "description": "Mapping between (fault, type, severity) combinations and combined label IDs"
            }, f, indent=2)
        
        print(f"\n[INFO] Mapping sauvegardé dans {mapping_path}")
    
    return y_combined, mapping, reverse_mapping


def main():
    # Chemins
    processed_dir = Path("data_processed")
    
    print("[INFO] Chargement des fichiers y...")
    y_fault = np.load(processed_dir / "y_fault.npy")
    y_type = np.load(processed_dir / "y_type.npy")
    y_sev = np.load(processed_dir / "y_sev.npy")
    
    # Combiner les labels
    y_combined, mapping, reverse_mapping = combine_y_labels(
        y_fault, y_type, y_sev, save_dir=processed_dir
    )
    
    # Sauvegarder y_combined
    output_path = processed_dir / "y_combined.npy"
    np.save(output_path, y_combined)
    print(f"\n[INFO] y_combined sauvegardé dans {output_path}")
    print(f"  Shape: {y_combined.shape}")
    print(f"  Dtype: {y_combined.dtype}")
    print(f"  Min: {y_combined.min()}, Max: {y_combined.max()}")
    
    # Vérification
    print("\n[INFO] Vérification de la cohérence...")
    for i in [0, len(y_combined)//2, len(y_combined)-1]:
        original = (int(y_fault[i]), int(y_type[i]), int(y_sev[i]))
        combined = int(y_combined[i])
        reconstructed = reverse_mapping[combined]
        match = "✓" if original == reconstructed else "✗"
        print(f"  Sample {i}: {original} -> {combined} -> {reconstructed} {match}")
    
    print("\n[INFO] Terminé!")


if __name__ == "__main__":
    main()
