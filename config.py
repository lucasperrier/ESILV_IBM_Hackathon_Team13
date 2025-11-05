# config.py
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DataConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    target_fs: float = 100.0
    win_sec: float = 1.0
    step_sec: float = 0.5


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_epochs: int = 50
    lr: float = 1e-3
    weight_fault: float = 2.0  # lambda1
    weight_type: float = 1.0   # lambda2
    weight_sev: float = 1.0    # lambda3
    device: str = "cuda"


@dataclass
class ProjectConfig:
    # ⚠️ Utilisation de default_factory pour éviter l’erreur
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# instance globale du projet
cfg = ProjectConfig()
