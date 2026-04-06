"""Unified experiment configuration for FNN and Transformer models."""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, List


@dataclass
class ExperimentConfig:
    # Model type
    model_type: Literal['fnn', 'transformer'] = 'fnn'

    # FNN params
    model_width: int = 256
    model_depth: int = 4
    activation: str = 'relu'
    dropout_rate: float = 0.0
    weight_initializer: str = 'glorot_uniform'

    # Transformer params
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 3
    d_ff: Optional[int] = None
    use_normalization: bool = True

    # Training params
    learning_rate: float = 1e-3
    batch_size_train: int = 8192
    batch_size_valid: int = 65536
    num_epochs: int = 5000
    early_stopping_patience: int = 500

    # Loss
    loss_type: str = 'mse'
    loss_alpha: float = 1.0
    loss_beta: float = 0.5

    # Data
    train_data: str = 'data/processed/train.csv'
    valid_data: str = 'data/processed/valid.csv'

    # Input/output dimensions
    num_inputs: int = 12
    num_outputs: int = 4

    # Data generation
    cut_type: Literal['line', 'arc', 'both'] = 'line'
    num_samples_start_edge: int = 15
    num_samples_end_edge: int = 15
    use_all_edges_as_start: bool = False
    arc_min_radius: float = 0.5
    arc_max_radius: float = 10.0
    arc_num_radius: int = 10
    include_arc_features: bool = False  # append radius/direction to input

    # Target point configuration (controls num_inputs)
    target_types: List[str] = field(
        default_factory=lambda: ['cell vertices', '2 points at given ratio']
    )
    target_ratio: List[float] = field(default_factory=lambda: [1e-3])

    # Reproducibility
    seed: int = 42

    # Output
    model_name: Optional[str] = None
    output_dir: Optional[str] = None
    gradient_clip_norm: float = 1.0

    # MLflow
    mlflow_experiment_name: Optional[str] = None

    # Sweep (optional, for sweep mode)
    sweep: Optional[dict] = None


def load_config(path: str) -> ExperimentConfig:
    """Load config from JSON file, return as ExperimentConfig dataclass."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Handle list of configs (return first one)
    if isinstance(data, list):
        data = data[0]

    # Filter to only known fields
    known_fields = {f.name for f in ExperimentConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known_fields}

    return ExperimentConfig(**filtered)


def load_configs(path: str) -> list:
    """Load all configs from a JSON file (single or list)."""
    with open(path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    known_fields = {f.name for f in ExperimentConfig.__dataclass_fields__.values()}
    configs = []
    for item in data:
        filtered = {k: v for k, v in item.items() if k in known_fields}
        configs.append(ExperimentConfig(**filtered))

    return configs


def save_config(config: ExperimentConfig, path: str):
    """Save config to JSON file."""
    with open(path, 'w') as f:
        json.dump(asdict(config), f, indent=4)
