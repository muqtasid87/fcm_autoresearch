"""MLflow tracking wrapper with graceful fallback.

All functions are no-ops if MLflow is not installed, so training code
can always call them without try/except.
"""

import os
import warnings

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def is_available():
    """Check if MLflow is installed and available."""
    return MLFLOW_AVAILABLE


def start_run(experiment_name: str, run_name: str = None, tags: dict = None):
    """Start an MLflow run. Returns run object or None if MLflow unavailable."""
    if not MLFLOW_AVAILABLE:
        return None

    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name, tags=tags)
    return run


def log_params(params: dict):
    """Log parameters to active MLflow run."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        # MLflow has a 500-param limit per batch; flatten nested dicts
        flat = {}
        for k, v in params.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}.{kk}"] = vv
            else:
                flat[k] = v
        mlflow.log_params(flat)
    except Exception as e:
        warnings.warn(f"MLflow log_params failed: {e}")


def log_config(config: dict):
    """Log experiment config as MLflow parameters.

    Handles nested dicts and long values gracefully.
    """
    if not MLFLOW_AVAILABLE:
        return
    try:
        flat = {}
        for k, v in config.items():
            if isinstance(v, (dict, list)):
                import json
                flat[k] = json.dumps(v)[:250]  # MLflow 250-char param limit
            else:
                flat[k] = v
        mlflow.log_params(flat)
    except Exception as e:
        warnings.warn(f"MLflow log_config failed: {e}")


def log_metrics(metrics: dict, step: int = None):
    """Log metrics to active MLflow run."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        warnings.warn(f"MLflow log_metrics failed: {e}")


def log_artifact(path: str):
    """Log an artifact file to active MLflow run."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        if os.path.exists(path):
            mlflow.log_artifact(path)
    except Exception as e:
        warnings.warn(f"MLflow log_artifact failed: {e}")


def log_environment(env: dict):
    """Log environment info (from reproducibility.capture_environment) as tags."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        tags = {
            'git_hash': str(env.get('git_hash', 'unknown')),
            'git_status': str(env.get('git_status', 'unknown')),
            'tensorflow_version': str(env.get('tensorflow_version', 'unknown')),
            'gpu_count': str(env.get('gpu_count', 0)),
        }
        mlflow.set_tags(tags)
    except Exception as e:
        warnings.warn(f"MLflow log_environment failed: {e}")


def log_figure(fig, name: str):
    """Save a matplotlib figure as an MLflow artifact.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to log.
    name : str
        Filename for the artifact (e.g., 'loss_curve.png').
    """
    if not MLFLOW_AVAILABLE:
        return
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(tmp.name, artifact_path='figures')
            os.unlink(tmp.name)
    except Exception as e:
        warnings.warn(f"MLflow log_figure failed: {e}")


def end_run():
    """End the active MLflow run."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.end_run()
    except Exception:
        pass


def create_keras_callback(log_interval: int = 1):
    """Create a Keras callback that logs metrics to MLflow per epoch.

    Parameters
    ----------
    log_interval : int
        Log every N epochs (default: every epoch).

    Returns
    -------
    callback or None
        A Keras LambdaCallback, or None if MLflow is unavailable.
    """
    if not MLFLOW_AVAILABLE:
        return None

    try:
        import tensorflow as tf

        class MLflowCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs and (epoch + 1) % log_interval == 0:
                    metrics = {}
                    for k, v in logs.items():
                        metrics[k] = float(v)
                    try:
                        mlflow.log_metrics(metrics, step=epoch)
                    except Exception:
                        pass

        return MLflowCallback()
    except ImportError:
        return None
