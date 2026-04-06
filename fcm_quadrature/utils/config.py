"""Unified experiment configuration for FNN and Transformer models."""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal


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

    # Output
    model_name: Optional[str] = None
    num_outputs: int = 4
    output_dir: Optional[str] = None
    gradient_clip_norm: float = 1.0


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


def save_config(config: ExperimentConfig, path: str):
    """Save config to JSON file."""
    with open(path, 'w') as f:
        json.dump(asdict(config), f, indent=4)
