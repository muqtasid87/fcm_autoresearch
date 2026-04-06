#!/usr/bin/env python
"""
Training script for FCM quadrature neural networks.

Supports:
  - Single model training
  - Parallel multi-model training (round-robin across GPUs)
  - Structured sweeps (grid, random, Optuna/Bayesian)
  - Auto hardware detection

Usage:
    # Generate a template config
    python scripts/train.py --template > my_config.json

    # Train a single model
    python scripts/train.py my_config.json --train-data data/train.csv --valid-data data/valid.csv

    # Train multiple models in parallel on GPUs
    python scripts/train.py configs.json --num-gpus 8

    # Grid/random sweep
    python scripts/train.py --sweep sweep_config.json --train-data data/train.csv --valid-data data/valid.csv

    # Optuna Bayesian optimization
    python scripts/train.py --sweep optuna_config.json --train-data data/train.csv --valid-data data/valid.csv

    # Auto-detect hardware
    python scripts/train.py configs.json --auto

    # Preview without training
    python scripts/train.py my_configs.json --dry-run
"""

import os
import sys
import json
import argparse

# Default values for all supported hyperparameters
DEFAULTS = {
    'model_width': 256,
    'model_depth': 4,
    'activation': 'relu',
    'weight_initializer': 'glorot_uniform',
    'learning_rate': 1e-3,
    'batch_size_train': 2**13,       # 8192
    'batch_size_valid': 2**16,       # 65536
    'num_epochs': 5000,
    'early_stopping_patience': 500,
    'loss_type': 'combined',
    'loss_alpha': 1.0,
    'loss_beta': 0.5,
    'dropout_rate': 0.0,
    'num_inputs': 12,
    'num_outputs': 4,
    'use_normalization': False,
    'gradient_clip_norm': 1.0,
    'seed': 42,
}


def generate_template(count=1):
    """Generate a template config with all supported hyperparameters."""
    config = {
        '_comment': 'All fields below are optional. Defaults are shown.',
        'model_width': 256,
        'model_depth': 4,
        'activation': 'relu',
        'weight_initializer': 'glorot_uniform',
        'learning_rate': 1e-3,
        'batch_size_train': 8192,
        'batch_size_valid': 65536,
        'num_epochs': 5000,
        'early_stopping_patience': 500,
        'loss_type': 'mse',
        'loss_alpha': 1.0,
        'loss_beta': 0.5,
        'dropout_rate': 0.0,
        'num_inputs': 12,
        'num_outputs': 4,
        'use_normalization': True,
        'gradient_clip_norm': 1.0,
        'seed': 42,
    }

    if count == 1:
        return config

    configs = []
    for i in range(count):
        c = dict(config)
        del c['_comment']
        configs.append(c)
    return configs


def make_model_name(config, index=None):
    """Generate a descriptive model name from config."""
    parts = []
    if index is not None:
        parts.append(f"model_{index + 1:02d}")
    parts.append(f"w{config['model_width']}")
    parts.append(f"d{config['model_depth']}")
    parts.append(f"lr{config['learning_rate']:.0e}")
    parts.append(f"bs{config['batch_size_train']}")
    parts.append(config['activation'])
    if config.get('loss_type', 'mse') != 'mse':
        parts.append(config['loss_type'])
    if config.get('dropout_rate', 0) > 0:
        parts.append(f"drop{config['dropout_rate']}")
    return '_'.join(parts)


def fill_defaults(config):
    """Fill in default values for any missing keys."""
    filled = dict(DEFAULTS)
    for key, value in config.items():
        if key.startswith('_'):
            continue
        filled[key] = value
    return filled


def train_single(config, train_data, valid_data, output_dir):
    """Train a single model directly (no subprocess)."""
    from fcm_quadrature.training.train_single import train_model

    config = fill_defaults(config)
    if 'model_name' not in config:
        config['model_name'] = make_model_name(config)

    data_paths = {
        'train': os.path.abspath(train_data),
        'valid': os.path.abspath(valid_data),
    }

    model_output_dir = os.path.join(output_dir, config['model_name'])

    results = train_model(
        config, data_paths, model_output_dir,
        gpu_id=None,
        num_threads=4,
    )

    print(f"\nTraining complete: {results['model_name']}")
    print(f"  Best val loss    : {results['best_val_loss']:.6e}")
    print(f"  Mean rel error   : {results['mean_abs_rel_error']:.6f}")
    print(f"  Training time    : {results['training_time_seconds']:.1f}s")
    print(f"  Actual epochs    : {results['actual_epochs']}"
          + (" (early stopped)" if results['stopped_early'] else ""))
    print(f"  Output           : {model_output_dir}")

    return results


def train_parallel(configs, train_data, valid_data, output_dir,
                   num_gpus=8, max_parallel=None, threads_per_job=3,
                   cpu_only=False, mlflow_experiment=None):
    """Train multiple models in parallel using the orchestrator."""
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    from fcm_quadrature.training.train_parallel import ParallelGPUHyperparameterSearch

    # Fill defaults and assign names
    full_configs = []
    for idx, config in enumerate(configs):
        filled = fill_defaults(config)
        if 'model_name' not in filled:
            filled['model_name'] = make_model_name(filled, idx)
        full_configs.append(filled)

    if max_parallel is None:
        max_parallel = len(full_configs)

    data_paths = {
        'train': os.path.abspath(train_data),
        'valid': os.path.abspath(valid_data),
    }

    search = ParallelGPUHyperparameterSearch(
        output_dir=output_dir,
        num_gpus=num_gpus,
        max_parallel=max_parallel,
        threads_per_job=threads_per_job,
        cpu_only=cpu_only,
        mlflow_experiment_name=mlflow_experiment,
    )

    return search.run(full_configs, data_paths)


def run_sweep(sweep_path, train_data, valid_data, output_dir,
              num_gpus=8, max_parallel=None, threads_per_job=3,
              cpu_only=False):
    """Run a structured hyperparameter sweep."""
    from fcm_quadrature.training.sweeps import load_sweep, generate_configs, run_optuna_sweep

    sweep = load_sweep(sweep_path)
    print(f"Loaded {sweep.sweep_type} sweep from {sweep_path}")

    data_paths = {
        'train': os.path.abspath(train_data),
        'valid': os.path.abspath(valid_data),
    }

    if sweep.sweep_type in ('grid', 'random'):
        configs = generate_configs(sweep)
        print(f"Generated {len(configs)} configurations")

        if not configs:
            print("ERROR: No configurations generated from sweep")
            sys.exit(1)

        # Merge base_config defaults
        final_configs = []
        for c in configs:
            merged = dict(DEFAULTS)
            merged.update(sweep.base_config)
            merged.update(c)
            final_configs.append(merged)

        # Dry run preview
        print(f"\nSweep preview ({sweep.sweep_type}):")
        for key in sweep.param_space:
            vals = sorted(set(str(c.get(key, '?')) for c in final_configs))
            if len(vals) <= 10:
                print(f"  {key}: {', '.join(vals)}")
            else:
                print(f"  {key}: {len(vals)} unique values")
        print()

        train_parallel(
            final_configs, train_data, valid_data, output_dir,
            num_gpus=num_gpus,
            max_parallel=max_parallel,
            threads_per_job=threads_per_job,
            cpu_only=cpu_only,
            mlflow_experiment=sweep.base_config.get('mlflow_experiment_name'),
        )

    elif sweep.sweep_type == 'optuna':
        gpu_ids = list(range(num_gpus)) if not cpu_only else None
        run_optuna_sweep(
            sweep, data_paths, output_dir,
            gpu_ids=gpu_ids,
            num_threads=threads_per_job,
        )


def dry_run(configs, cpu_only=False, num_gpus=8):
    """Preview what would be trained."""
    print("=" * 70)
    print("DRY RUN - Configuration Preview")
    print("=" * 70)
    print(f"\nTotal models: {len(configs)}\n")

    for i, raw_config in enumerate(configs):
        config = fill_defaults(raw_config)
        name = config.get('model_name', make_model_name(config, i))
        if cpu_only:
            device = "CPU"
        else:
            device = f"GPU {i % num_gpus}"
        print(f"  {i + 1:2d}. [{device}] {name}")
        print(f"       width={config['model_width']}, depth={config['model_depth']}, "
              f"lr={config['learning_rate']:.0e}, bs={config['batch_size_train']}, "
              f"act={config['activation']}, loss={config['loss_type']}, "
              f"dropout={config['dropout_rate']}")

    filled_configs = [fill_defaults(c) for c in configs]
    print(f"\nHyperparameter ranges:")
    for key in ['model_width', 'model_depth', 'learning_rate', 'batch_size_train',
                'activation', 'loss_type', 'dropout_rate', 'weight_initializer']:
        values = sorted(set(str(c[key]) for c in filled_configs))
        print(f"  {key:<25s}: {', '.join(values)}")


def main():
    parser = argparse.ArgumentParser(
        description='Train FCM quadrature neural networks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py --template > config.json
  python scripts/train.py config.json --train-data data/train.csv --valid-data data/valid.csv
  python scripts/train.py configs.json --num-gpus 8
  python scripts/train.py --sweep sweep.json --train-data data/train.csv --valid-data data/valid.csv
  python scripts/train.py configs.json --auto
  python scripts/train.py configs.json --dry-run
        """
    )

    parser.add_argument('config', nargs='?', type=str,
                        help='Path to JSON config file (model configs or list)')
    parser.add_argument('--sweep', type=str, default=None,
                        help='Path to sweep config JSON (grid/random/optuna)')
    parser.add_argument('--template', action='store_true',
                        help='Print a template config and exit')
    parser.add_argument('--count', type=int, default=1,
                        help='Number of model configs to generate with --template')
    parser.add_argument('--train-data', type=str,
                        default='Data/Training_1M_NoVertices.csv',
                        help='Path to training CSV')
    parser.add_argument('--valid-data', type=str,
                        default='Data/Valid_1M_NoVertices.csv',
                        help='Path to validation CSV')
    parser.add_argument('--output-dir', type=str, default='experiments',
                        help='Base output directory (default: experiments)')
    parser.add_argument('--num-gpus', type=str, default='8',
                        help='Number of GPUs (integer or "auto")')
    parser.add_argument('--max-parallel', type=int, default=None,
                        help='Max concurrent jobs (default: number of models)')
    parser.add_argument('--threads-per-job', type=int, default=3,
                        help='CPU threads per job')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Train on CPU only')
    parser.add_argument('--auto', action='store_true',
                        help='Auto-detect hardware and configure GPUs/threads')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview configurations without training')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override seed for all configs')
    parser.add_argument('--mlflow-experiment', type=str, default=None,
                        help='MLflow experiment name for tracking')

    args = parser.parse_args()

    # Template mode
    if args.template:
        template = generate_template(args.count)
        print(json.dumps(template, indent=4))
        return

    # Auto hardware detection
    num_gpus = 8
    if args.auto or args.num_gpus == 'auto':
        from fcm_quadrature.utils.hardware import detect_hardware, auto_configure_training
        hw = detect_hardware()
        hw_config = auto_configure_training(hw)
        num_gpus = hw_config['num_gpus']
        if args.max_parallel is None:
            args.max_parallel = hw_config['max_parallel']
        args.threads_per_job = hw_config['threads_per_job']
        args.cpu_only = hw_config['cpu_only']
        print(f"Auto-detected: {hw_config['summary']}")
    else:
        try:
            num_gpus = int(args.num_gpus)
        except ValueError:
            print(f"ERROR: --num-gpus must be an integer or 'auto', got '{args.num_gpus}'")
            sys.exit(1)

    # Sweep mode
    if args.sweep:
        if not os.path.exists(args.train_data):
            print(f"ERROR: Training data not found: {args.train_data}")
            sys.exit(1)
        if not os.path.exists(args.valid_data):
            print(f"ERROR: Validation data not found: {args.valid_data}")
            sys.exit(1)

        run_sweep(
            args.sweep, args.train_data, args.valid_data, args.output_dir,
            num_gpus=num_gpus,
            max_parallel=args.max_parallel,
            threads_per_job=args.threads_per_job,
            cpu_only=args.cpu_only,
        )
        return

    # Standard config mode
    if not args.config:
        parser.print_help()
        sys.exit(1)

    # Load config
    with open(args.config, 'r') as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        configs = [raw]
    elif isinstance(raw, list):
        configs = raw
    else:
        print("ERROR: Config must be a JSON object or array")
        sys.exit(1)

    # Apply CLI overrides
    if args.seed is not None:
        for c in configs:
            c['seed'] = args.seed
    if args.mlflow_experiment:
        for c in configs:
            c['mlflow_experiment_name'] = args.mlflow_experiment

    print(f"Loaded {len(configs)} model config(s) from {args.config}")

    # Dry run
    if args.dry_run:
        dry_run(configs, cpu_only=args.cpu_only, num_gpus=num_gpus)
        return

    # Check data files
    if not os.path.exists(args.train_data):
        print(f"ERROR: Training data not found: {args.train_data}")
        sys.exit(1)
    if not os.path.exists(args.valid_data):
        print(f"ERROR: Validation data not found: {args.valid_data}")
        sys.exit(1)

    # Single model: train directly
    if len(configs) == 1:
        train_single(configs[0], args.train_data, args.valid_data, args.output_dir)
    # Multiple models: train in parallel
    else:
        train_parallel(
            configs, args.train_data, args.valid_data, args.output_dir,
            num_gpus=num_gpus,
            max_parallel=args.max_parallel,
            threads_per_job=args.threads_per_job,
            cpu_only=args.cpu_only,
            mlflow_experiment=args.mlflow_experiment,
        )


if __name__ == '__main__':
    main()
