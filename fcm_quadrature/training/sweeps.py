"""Structured hyperparameter sweeps: grid, random, and Optuna (Bayesian).

Usage:
    from fcm_quadrature.training.sweeps import load_sweep, generate_configs

    sweep = load_sweep('configs/sweep_grid.json')
    configs = generate_configs(sweep)
    # configs is a list of dicts ready for ParallelGPUHyperparameterSearch.run()
"""

import json
import itertools
import random
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class SweepConfig:
    sweep_type: Literal['grid', 'random', 'optuna'] = 'grid'
    base_config: dict = field(default_factory=dict)
    param_space: dict = field(default_factory=dict)
    n_trials: int = 50
    seed: int = 42
    optuna_sampler: str = 'tpe'
    optuna_pruner: Optional[str] = None  # 'median', 'hyperband', or None


def load_sweep(path: str) -> SweepConfig:
    """Load a sweep config from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return SweepConfig(
        sweep_type=data.get('sweep_type', 'grid'),
        base_config=data.get('base_config', {}),
        param_space=data.get('param_space', {}),
        n_trials=data.get('n_trials', 50),
        seed=data.get('seed', 42),
        optuna_sampler=data.get('optuna_sampler', 'tpe'),
        optuna_pruner=data.get('optuna_pruner', None),
    )


def generate_configs(sweep: SweepConfig) -> list:
    """Generate a list of experiment configs from a sweep definition.

    For grid sweeps, returns all combinations.
    For random sweeps, returns n_trials random samples.
    For optuna, returns empty list (Optuna generates configs dynamically).
    """
    if sweep.sweep_type == 'grid':
        return _generate_grid(sweep.base_config, sweep.param_space)
    elif sweep.sweep_type == 'random':
        return _generate_random(
            sweep.base_config, sweep.param_space,
            sweep.n_trials, sweep.seed
        )
    elif sweep.sweep_type == 'optuna':
        return []  # Optuna generates configs on-the-fly
    else:
        raise ValueError(f"Unknown sweep_type: {sweep.sweep_type}")


def _generate_grid(base_config: dict, param_space: dict) -> list:
    """Cartesian product of all parameter lists."""
    # Separate list params from distribution params
    grid_params = {}
    for key, val in param_space.items():
        if isinstance(val, list):
            grid_params[key] = val
        elif isinstance(val, dict) and val.get('type') in ('log_uniform', 'uniform', 'int_uniform'):
            raise ValueError(
                f"Grid sweep does not support distributions for '{key}'. "
                f"Use explicit lists or switch to random/optuna sweep."
            )
        else:
            grid_params[key] = [val]  # single value

    keys = list(grid_params.keys())
    values = [grid_params[k] for k in keys]

    configs = []
    for combo in itertools.product(*values):
        config = deepcopy(base_config)
        for k, v in zip(keys, combo):
            config[k] = v
        configs.append(config)

    return configs


def _sample_param(spec, rng):
    """Sample a single parameter value from its specification."""
    if isinstance(spec, list):
        return rng.choice(spec)
    elif isinstance(spec, dict):
        ptype = spec.get('type', 'choice')
        if ptype == 'uniform':
            return rng.uniform(spec['low'], spec['high'])
        elif ptype == 'log_uniform':
            log_low = math.log(spec['low'])
            log_high = math.log(spec['high'])
            return math.exp(rng.uniform(log_low, log_high))
        elif ptype == 'int_uniform':
            return rng.randint(spec['low'], spec['high'])
        elif ptype == 'choice':
            return rng.choice(spec['values'])
        else:
            raise ValueError(f"Unknown param type: {ptype}")
    else:
        return spec  # fixed value


def _generate_random(base_config: dict, param_space: dict,
                     n_trials: int, seed: int) -> list:
    """Random sampling from parameter distributions."""
    rng = random.Random(seed)
    configs = []
    for _ in range(n_trials):
        config = deepcopy(base_config)
        for key, spec in param_space.items():
            config[key] = _sample_param(spec, rng)
        configs.append(config)
    return configs


def _suggest_optuna_params(trial, param_space: dict) -> dict:
    """Map param_space to Optuna trial suggestions."""
    import optuna  # noqa: F811

    params = {}
    for key, spec in param_space.items():
        if isinstance(spec, list):
            params[key] = trial.suggest_categorical(key, spec)
        elif isinstance(spec, dict):
            ptype = spec.get('type', 'choice')
            if ptype == 'uniform':
                params[key] = trial.suggest_float(key, spec['low'], spec['high'])
            elif ptype == 'log_uniform':
                params[key] = trial.suggest_float(key, spec['low'], spec['high'], log=True)
            elif ptype == 'int_uniform':
                params[key] = trial.suggest_int(key, spec['low'], spec['high'])
            elif ptype == 'choice':
                params[key] = trial.suggest_categorical(key, spec['values'])
            else:
                raise ValueError(f"Unknown param type: {ptype}")
        else:
            params[key] = spec
    return params


class OptunaObjective:
    """Wraps train_model as an Optuna objective function.

    Parameters
    ----------
    base_config : dict
        Base experiment config (non-sweep params).
    param_space : dict
        Parameter space for Optuna to search.
    data_paths : dict
        {'train': path, 'valid': path}
    output_dir : str
        Base output directory.
    gpu_id : int or None
        GPU to use for this objective.
    num_threads : int
        CPU threads per training job.
    """

    def __init__(self, base_config, param_space, data_paths, output_dir,
                 gpu_id=None, num_threads=4):
        self.base_config = base_config
        self.param_space = param_space
        self.data_paths = data_paths
        self.output_dir = output_dir
        self.gpu_id = gpu_id
        self.num_threads = num_threads

    def __call__(self, trial):
        from fcm_quadrature.training.train_single import train_model

        # Build config from trial suggestions
        config = deepcopy(self.base_config)
        suggested = _suggest_optuna_params(trial, self.param_space)
        config.update(suggested)

        # Name from trial number
        config['model_name'] = f"trial_{trial.number:04d}"
        trial_output = f"{self.output_dir}/trial_{trial.number:04d}"

        results = train_model(
            config, self.data_paths, trial_output,
            gpu_id=self.gpu_id, num_threads=self.num_threads, verbose=0
        )

        return results['best_val_loss']


def run_optuna_sweep(sweep: SweepConfig, data_paths: dict, output_dir: str,
                     gpu_ids: list = None, num_threads: int = 4):
    """Run an Optuna hyperparameter search.

    Parameters
    ----------
    sweep : SweepConfig
        Sweep configuration with param_space and n_trials.
    data_paths : dict
        {'train': path, 'valid': path}
    output_dir : str
        Base output directory.
    gpu_ids : list of int, optional
        GPU IDs to distribute trials across. If None, uses CPU.
    num_threads : int
        CPU threads per trial.

    Returns
    -------
    optuna.Study
        The completed Optuna study.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for Bayesian optimization sweeps. "
            "Install with: pip install optuna"
        )

    # Create sampler
    if sweep.optuna_sampler == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=sweep.seed)
    elif sweep.optuna_sampler == 'random':
        sampler = optuna.samplers.RandomSampler(seed=sweep.seed)
    elif sweep.optuna_sampler == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler(seed=sweep.seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=sweep.seed)

    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name=sweep.base_config.get('mlflow_experiment_name', 'optuna_sweep'),
    )

    # Use first GPU for sequential trials (or round-robin for parallel)
    gpu_id = gpu_ids[0] if gpu_ids else None

    objective = OptunaObjective(
        base_config=sweep.base_config,
        param_space=sweep.param_space,
        data_paths=data_paths,
        output_dir=output_dir,
        gpu_id=gpu_id,
        num_threads=num_threads,
    )

    study.optimize(objective, n_trials=sweep.n_trials)

    # Save study results
    import os
    results_path = os.path.join(output_dir, 'optuna_results.json')
    best = {
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
    }
    with open(results_path, 'w') as f:
        json.dump(best, f, indent=2)

    print(f"\nOptuna sweep complete: {len(study.trials)} trials")
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"  Best val loss: {study.best_value:.6e}")
    print(f"  Best params: {study.best_params}")

    return study
