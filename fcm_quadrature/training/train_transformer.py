#!/usr/bin/env python
"""
FT-Transformer Training and CLI for FCM Quadrature Weight Prediction

Unified script for training Feature Tokenizer Transformer models.
Supports single-GPU, multi-GPU, and CPU-only execution via tf.distribute.MirroredStrategy.

Usage:
    # Train with default settings (auto-detects GPUs)
    python -m fcm_quadrature.training.train_transformer

    # Train with a config file
    python -m fcm_quadrature.training.train_transformer --config transformer_config.json

    # Train with config file + CLI overrides (CLI wins)
    python -m fcm_quadrature.training.train_transformer --config transformer_config.json --lr 1e-3 --epochs 500

    # Force CPU-only training (no GPUs)
    python -m fcm_quadrature.training.train_transformer --cpu-only --config transformer_config.json

    # Evaluate a saved model on new data (no training)
    python -m fcm_quadrature.training.train_transformer --evaluate-only --model-path results/best_model.keras --eval-data Data/Valid.csv

    # Generate a template config file with all options
    python -m fcm_quadrature.training.train_transformer --template

GPU Strategy:
    - MirroredStrategy is used universally (works with 0, 1, or N GPUs)
    - --batch-size is PER REPLICA (global batch = batch_size * num_replicas)
    - --cpu-only hides all GPUs before TensorFlow initialization

Config Precedence:
    Hardcoded defaults < config.json < CLI arguments
"""

import os
import sys

# Pre-scan for --cpu-only before TF import (CUDA_VISIBLE_DEVICES must be set first)
if '--cpu-only' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from LiangNet_MultiOutput import preprocessData, saveHistory
from fcm_quadrature.training.losses import create_loss
from fcm_quadrature.models.transformer import FTTransformer, CUSTOM_OBJECTS
from fcm_quadrature.training.schedules import WarmupSchedule
from fcm_quadrature.analysis.evaluate import evaluate_model


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULTS = {
    # Data paths
    'train_data': 'Data/Training_1M_NoVertices.csv',
    'valid_data': 'Data/Valid_1M_NoVertices.csv',

    # Model architecture
    'd_model': 64,
    'num_heads': 4,
    'num_layers': 3,
    'd_ff': None,           # Default: 4 * d_model
    'dropout_rate': 0.1,
    'use_normalization': True,

    # Training
    'batch_size': 4096,     # Per replica
    'learning_rate': 1e-5,
    'num_epochs': 600,
    'early_stopping_patience': 100,

    # Loss
    'loss_type': 'combined',  # 'mse', 'moment', or 'combined'
    'loss_alpha': 1.0,
    'loss_beta': 0.5,

    # Output
    'output_dir': None,     # Auto-generated if None
}


# =============================================================================
# Configuration Helpers
# =============================================================================

def load_config(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a JSON object, not an array.")
    return config


def merge_config(defaults, file_config, cli_overrides):
    """Merge configuration sources: defaults < file < CLI.

    Args:
        defaults: dict of default values.
        file_config: dict from config file (may be empty).
        cli_overrides: dict of explicitly-provided CLI arguments.

    Returns:
        Merged configuration dict.
    """
    merged = dict(defaults)

    # Apply config file values
    for key, value in file_config.items():
        if key.startswith('_'):  # Skip comment keys like "_comment"
            continue
        if key in merged:
            merged[key] = value
        else:
            print(f"  Warning: Unknown config key '{key}' (ignored)")

    # Apply CLI overrides (only explicitly provided ones)
    for key, value in cli_overrides.items():
        merged[key] = value

    return merged


def generate_template():
    """Generate a template config with all options documented."""
    template = {
        '_comment': 'FT-Transformer training config. All fields are optional; defaults shown. Set early_stopping_patience to 0 or null to disable early stopping.',
        'train_data': DEFAULTS['train_data'],
        'valid_data': DEFAULTS['valid_data'],
        'd_model': DEFAULTS['d_model'],
        'num_heads': DEFAULTS['num_heads'],
        'num_layers': DEFAULTS['num_layers'],
        'd_ff': DEFAULTS['d_ff'],
        'dropout_rate': DEFAULTS['dropout_rate'],
        'use_normalization': DEFAULTS['use_normalization'],
        'batch_size': DEFAULTS['batch_size'],
        'learning_rate': DEFAULTS['learning_rate'],
        'num_epochs': DEFAULTS['num_epochs'],
        'early_stopping_patience': DEFAULTS['early_stopping_patience'],
        'loss_type': DEFAULTS['loss_type'],
        'loss_alpha': DEFAULTS['loss_alpha'],
        'loss_beta': DEFAULTS['loss_beta'],
        'output_dir': DEFAULTS['output_dir'],
    }
    return template


# =============================================================================
# Dataset Creation
# =============================================================================

def create_dataset(X, y, batch_size, shuffle=True, buffer_size=100000, repeat=False):
    """Create TensorFlow dataset with proper shuffling and prefetching.

    Args:
        X: Input features.
        y: Target labels.
        batch_size: Batch size.
        shuffle: Whether to shuffle the dataset.
        buffer_size: Buffer size for shuffling.
        repeat: Whether to repeat infinitely (needed for distributed training).
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# =============================================================================
# Training
# =============================================================================

def train_transformer(config):
    """
    Train a transformer model for quadrature weight prediction.

    Uses tf.distribute.MirroredStrategy universally. The strategy automatically
    handles single-GPU, multi-GPU, and CPU-only cases.

    Args:
        config: dict with all hyperparameters (from merge_config).

    Returns:
        results: dict with training results and evaluation metrics.
        output_dir: path to output directory.
    """

    print("=" * 80)
    print("TRANSFORMER MODEL TRAINING")
    print("=" * 80)

    # =========================================================================
    # 0. Setup Distribution Strategy
    # =========================================================================

    print("\n" + "=" * 80)
    print("STEP 0: Device Setup")
    print("=" * 80)

    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync

    devices = tf.config.list_physical_devices('GPU')
    if devices:
        print(f"GPUs detected: {len(devices)}")
        for i, device in enumerate(devices):
            print(f"  GPU {i}: {device.name}")
    else:
        print("No GPUs detected, using CPU")
    print(f"Number of replicas: {num_replicas}")

    # Extract config values
    d_model = config['d_model']
    num_heads = config['num_heads']
    num_layers = config['num_layers']
    d_ff = config['d_ff']
    dropout_rate = config['dropout_rate']
    use_normalization = config['use_normalization']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    early_stopping_patience = config['early_stopping_patience']
    loss_type = config['loss_type']
    loss_alpha = config['loss_alpha']
    loss_beta = config['loss_beta']

    global_batch_size = batch_size * num_replicas
    print(f"\nBatch size per replica: {batch_size}")
    print(f"Global batch size: {global_batch_size}")

    # Create output directory
    output_dir = config.get('output_dir')
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'transformer_results/train_{timestamp}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Resolve d_ff
    if d_ff is None:
        d_ff = 4 * d_model

    # Build saved config (what gets written to model_config.json)
    saved_config = {
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'd_ff': d_ff,
        'dropout_rate': dropout_rate,
        'use_normalization': use_normalization,
        'batch_size_per_replica': batch_size,
        'global_batch_size': global_batch_size,
        'num_replicas': num_replicas,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'early_stopping_patience': early_stopping_patience,
        'loss_type': loss_type,
        'loss_alpha': loss_alpha,
        'loss_beta': loss_beta,
    }

    print(f"\nConfiguration:")
    for k, v in saved_config.items():
        print(f"  {k}: {v}")
    print(f"\nOutput directory: {output_dir}")

    # =========================================================================
    # 1. Load Data
    # =========================================================================

    print("\n" + "=" * 80)
    print("STEP 1: Loading Data")
    print("=" * 80)

    data = preprocessData(
        config['train_data'],
        config['valid_data'],
        testSetPath=None,
        dtype=np.float32,
        num_outputs=4,
    )

    XTrain, yTrain, XValid, yValid = data

    print(f"Training samples: {XTrain.shape[0]:,}")
    print(f"Validation samples: {XValid.shape[0]:,}")
    print(f"Input features: {XTrain.shape[1]}")
    print(f"Output targets: {yTrain.shape[1]}")

    # Create datasets (batch size is per replica, repeat for distributed training)
    train_dataset = create_dataset(XTrain, yTrain, batch_size, shuffle=True, repeat=True)
    valid_dataset = create_dataset(XValid, yValid, batch_size, shuffle=False, repeat=True)

    # Distribute datasets across replicas
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)

    steps_per_epoch = int(np.ceil(len(XTrain) / global_batch_size))
    validation_steps = int(np.ceil(len(XValid) / global_batch_size))
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # =========================================================================
    # 2. Build Model (inside strategy scope)
    # =========================================================================

    print("\n" + "=" * 80)
    print("STEP 2: Building Transformer Model")
    print("=" * 80)

    with strategy.scope():
        model = FTTransformer(
            num_features=XTrain.shape[1],
            num_outputs=yTrain.shape[1],
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            use_normalization=use_normalization,
        )

        # Adapt normalization layer
        if use_normalization:
            print("Adapting input normalization...")
            model.adapt_normalization(XTrain)
        else:
            print("Input normalization disabled")

        # Build model by calling it once
        _ = model(XTrain[:2])

        total_params = model.count_params()
        print(f"Total parameters: {total_params:,}")
        model.summary()

    # Save model config
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(saved_config, f, indent=4)

    # =========================================================================
    # 3. Compile Model (inside strategy scope)
    # =========================================================================

    print("\n" + "=" * 80)
    print("STEP 3: Compiling Model")
    print("=" * 80)

    with strategy.scope():
        # Learning rate schedule: warm-up + cosine decay
        warmup_epochs = 10
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = num_epochs * steps_per_epoch

        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_steps - warmup_steps,
            alpha=0.01  # Final LR = 1% of initial
        )

        lr_with_warmup = WarmupSchedule(learning_rate, warmup_steps, lr_schedule)

        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_with_warmup,
            weight_decay=0.01,
            clipnorm=1.0
        )

        loss_fn = create_loss(loss_type, alpha=loss_alpha, beta=loss_beta)

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['mae']
        )

    print(f"Optimizer: AdamW (weight_decay=0.01, clipnorm=1.0)")
    print(f"Learning rate: {learning_rate} with warmup ({warmup_epochs} epochs) + cosine decay")
    print(f"Loss: {loss_type}" + (f" (alpha={loss_alpha}, beta={loss_beta})" if loss_type == 'combined' else ""))
    print(f"Gradients synchronized across {num_replicas} replica(s)")

    # =========================================================================
    # 4. Setup Callbacks
    # =========================================================================

    print("\n" + "=" * 80)
    print("STEP 4: Setting up Callbacks")
    print("=" * 80)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'training_log.csv')
        ),
    ]

    if early_stopping_patience and early_stopping_patience > 0:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ))
        print(f"Early stopping patience: {early_stopping_patience} epochs")
    else:
        print("Early stopping: DISABLED (training will run all epochs)")
    print(f"Model checkpoint: best_model.keras")

    # =========================================================================
    # 5. Train Model
    # =========================================================================

    print("\n" + "=" * 80)
    print("STEP 5: Training")
    print("=" * 80)

    train_start = time.time()

    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=1
    )

    train_time = time.time() - train_start
    actual_epochs = len(history.history['loss'])
    stopped_early = actual_epochs < num_epochs

    print(f"\nTraining completed!")
    print(f"Time: {train_time:.2f}s ({train_time/60:.2f} min)")
    print(f"Epochs: {actual_epochs}" + (" (early stopped)" if stopped_early else ""))

    # Save history
    saveHistory(history.history, os.path.join(output_dir, 'history.pickle'))

    # =========================================================================
    # 6. Load Best Model & Evaluate
    # =========================================================================

    print("\n" + "=" * 80)
    print("STEP 6: Loading Best Model")
    print("=" * 80)

    try:
        model = keras.models.load_model(
            os.path.join(output_dir, 'best_model.keras'),
            custom_objects=CUSTOM_OBJECTS,
        )
        print("Loaded best model from checkpoint")
    except Exception as e:
        print(f"Warning: Could not load checkpoint ({e}), using final weights")

    # Save final model
    model.save(os.path.join(output_dir, 'final_model.keras'))
    print("Saved: final_model.keras")

    # Add training-specific info to config for evaluation
    saved_config['training_time_seconds'] = train_time
    saved_config['actual_epochs'] = actual_epochs
    saved_config['stopped_early'] = stopped_early

    # Run evaluation
    results, _ = evaluate_model(
        model=model,
        XEval=XValid,
        yEval=yValid,
        output_dir=output_dir,
        batch_size=batch_size,
        history=history,
        train_config=saved_config,
    )

    # =========================================================================
    # Final Report
    # =========================================================================

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nKey Results:")
    print(f"  Replicas: {num_replicas}")
    print(f"  Best validation loss: {results.get('mse', 'N/A')}")
    print(f"  Mean relative error: {results.get('mean_rel_error', 'N/A')}")
    print(f"  Mean AF error: {results.get('mean_af_error', 'N/A')}")
    print(f"  Training time: {train_time/60:.1f} minutes")
    print(f"  Parameters: {total_params:,}")
    print("=" * 80 + "\n")

    return results, output_dir


# =============================================================================
# Argument Parser & Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train or evaluate FT-Transformer for quadrature weight prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config transformer_config.json
  %(prog)s --config transformer_config.json --lr 1e-3 --epochs 500
  %(prog)s --cpu-only --epochs 10
  %(prog)s --evaluate-only --model-path results/best_model.keras --eval-data Data/Valid.csv
  %(prog)s --template
        """
    )

    # Mode selection
    mode_group = parser.add_argument_group('Mode')
    mode_group.add_argument('--evaluate-only', action='store_true',
                            help='Evaluate a saved model without training')
    mode_group.add_argument('--template', action='store_true',
                            help='Print a template config JSON to stdout and exit')
    mode_group.add_argument('--cpu-only', action='store_true',
                            help='Force CPU-only mode (hide all GPUs)')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file (CLI args override config values)')

    # Data paths
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--train-data', type=str, default=None,
                            help=f'Path to training CSV (default: {DEFAULTS["train_data"]})')
    data_group.add_argument('--valid-data', type=str, default=None,
                            help=f'Path to validation CSV (default: {DEFAULTS["valid_data"]})')

    # Evaluate-only specific
    eval_group = parser.add_argument_group('Evaluate-only mode')
    eval_group.add_argument('--model-path', type=str, default=None,
                            help='Path to saved .keras model (required for --evaluate-only)')
    eval_group.add_argument('--eval-data', type=str, default=None,
                            help='Path to evaluation CSV (required for --evaluate-only)')

    # Model hyperparameters (all default=None to detect explicit CLI usage)
    model_group = parser.add_argument_group('Model hyperparameters')
    model_group.add_argument('--d-model', type=int, default=None,
                             help=f'Embedding dimension (default: {DEFAULTS["d_model"]})')
    model_group.add_argument('--n-heads', type=int, default=None,
                             help=f'Number of attention heads (default: {DEFAULTS["num_heads"]})')
    model_group.add_argument('--n-layers', type=int, default=None,
                             help=f'Number of transformer layers (default: {DEFAULTS["num_layers"]})')
    model_group.add_argument('--d-ff', type=int, default=None,
                             help='FFN hidden dim (default: 4*d_model)')
    model_group.add_argument('--dropout', type=float, default=None,
                             help=f'Dropout rate (default: {DEFAULTS["dropout_rate"]})')
    model_group.add_argument('--no-normalization', action='store_true',
                             help='Disable input normalization')

    # Training hyperparameters
    train_group = parser.add_argument_group('Training hyperparameters')
    train_group.add_argument('--batch-size', type=int, default=None,
                             help=f'Batch size per replica (default: {DEFAULTS["batch_size"]})')
    train_group.add_argument('--lr', type=float, default=None,
                             help=f'Learning rate (default: {DEFAULTS["learning_rate"]})')
    train_group.add_argument('--epochs', type=int, default=None,
                             help=f'Max epochs (default: {DEFAULTS["num_epochs"]})')
    train_group.add_argument('--patience', type=int, default=None,
                             help=f'Early stopping patience (default: {DEFAULTS["early_stopping_patience"]})')

    # Loss function
    loss_group = parser.add_argument_group('Loss function')
    loss_group.add_argument('--loss-type', type=str, default=None,
                            choices=['mse', 'moment', 'combined'],
                            help=f'Loss type (default: {DEFAULTS["loss_type"]})')
    loss_group.add_argument('--loss-alpha', type=float, default=None,
                            help=f'Weight for MSE in combined loss (default: {DEFAULTS["loss_alpha"]})')
    loss_group.add_argument('--loss-beta', type=float, default=None,
                            help=f'Weight for moment in combined loss (default: {DEFAULTS["loss_beta"]})')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-generated)')
    parser.add_argument('--num-outputs', type=int, default=None,
                        help='Number of model outputs (default: 4)')

    args = parser.parse_args()

    # --- Template mode ---
    if args.template:
        print(json.dumps(generate_template(), indent=4))
        return

    # --- Evaluate-only mode ---
    if args.evaluate_only:
        if not args.model_path:
            parser.error("--evaluate-only requires --model-path")
        if not args.eval_data:
            parser.error("--evaluate-only requires --eval-data")

        results, output_dir = evaluate_model(
            model_path=args.model_path,
            eval_data_path=args.eval_data,
            output_dir=args.output_dir,
            batch_size=args.batch_size or DEFAULTS['batch_size'],
            num_outputs=args.num_outputs or 4,
        )
        print(f"\nResults saved to: {output_dir}/")
        print("Done!")
        return

    # --- Training mode ---
    file_config = load_config(args.config) if args.config else {}

    # Build CLI overrides (only explicitly-provided values)
    cli_overrides = {}
    if args.train_data is not None:     cli_overrides['train_data'] = args.train_data
    if args.valid_data is not None:     cli_overrides['valid_data'] = args.valid_data
    if args.d_model is not None:        cli_overrides['d_model'] = args.d_model
    if args.n_heads is not None:        cli_overrides['num_heads'] = args.n_heads
    if args.n_layers is not None:       cli_overrides['num_layers'] = args.n_layers
    if args.d_ff is not None:           cli_overrides['d_ff'] = args.d_ff
    if args.dropout is not None:        cli_overrides['dropout_rate'] = args.dropout
    if args.no_normalization:           cli_overrides['use_normalization'] = False
    if args.batch_size is not None:     cli_overrides['batch_size'] = args.batch_size
    if args.lr is not None:             cli_overrides['learning_rate'] = args.lr
    if args.epochs is not None:         cli_overrides['num_epochs'] = args.epochs
    if args.patience is not None:       cli_overrides['early_stopping_patience'] = args.patience
    if args.loss_type is not None:      cli_overrides['loss_type'] = args.loss_type
    if args.loss_alpha is not None:     cli_overrides['loss_alpha'] = args.loss_alpha
    if args.loss_beta is not None:      cli_overrides['loss_beta'] = args.loss_beta
    if args.output_dir is not None:     cli_overrides['output_dir'] = args.output_dir

    config = merge_config(DEFAULTS, file_config, cli_overrides)

    results, output_dir = train_transformer(config)
    print("Done!")


if __name__ == '__main__':
    main()
