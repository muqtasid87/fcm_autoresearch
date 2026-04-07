"""Reproducibility utilities: seed management and environment capture."""

import os
import json
import random
import platform
import subprocess
from pathlib import Path


def set_global_seeds(seed: int):
    """Set seeds for TensorFlow, NumPy, Python random, and PYTHONHASHSEED.

    Call this at the very start of training before any TF operations.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_git_hash() -> str:
    """Return the current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def get_git_diff_status() -> str:
    """Return 'clean' or 'dirty' based on uncommitted changes."""
    try:
        result = subprocess.run(
            ['git', 'diff', '--quiet'],
            capture_output=True, timeout=5
        )
        return 'clean' if result.returncode == 0 else 'dirty'
    except Exception:
        return 'unknown'


def capture_environment() -> dict:
    """Capture full environment info for reproducibility."""
    env = {
        'git_hash': get_git_hash(),
        'git_status': get_git_diff_status(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
    }

    # TensorFlow version and GPU info
    try:
        import tensorflow as tf
        env['tensorflow_version'] = tf.__version__
        gpus = tf.config.list_physical_devices('GPU')
        env['gpu_count'] = len(gpus)
        env['gpu_devices'] = [g.name for g in gpus]
    except ImportError:
        env['tensorflow_version'] = 'not installed'

    # NumPy version
    try:
        import numpy as np
        env['numpy_version'] = np.__version__
    except ImportError:
        pass

    # Pip freeze (installed packages)
    try:
        result = subprocess.run(
            ['pip', 'freeze'],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            env['pip_freeze'] = result.stdout.strip().split('\n')
    except Exception:
        pass

    return env


def save_manifest(output_dir: str, config: dict, env: dict = None):
    """Save a reproducibility manifest JSON to the experiment directory.

    Parameters
    ----------
    output_dir : str
        Directory to save manifest.json
    config : dict
        Full experiment configuration
    env : dict, optional
        Environment info from capture_environment(). Captured if not provided.
    """
    if env is None:
        env = capture_environment()

    manifest = {
        'config': config,
        'environment': env,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest_path
