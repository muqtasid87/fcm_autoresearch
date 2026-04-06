"""MLflow tracking wrapper with graceful fallback."""

import warnings

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def start_run(experiment_name: str, run_name: str = None, tags: dict = None):
    """Start an MLflow run. Returns run object or None if MLflow unavailable."""
    if not MLFLOW_AVAILABLE:
        warnings.warn("MLflow not installed. Tracking disabled. Install with: pip install mlflow")
        return None

    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name, tags=tags)
    return run


def log_params(params: dict):
    """Log parameters to active MLflow run."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.log_params(params)
    except Exception as e:
        warnings.warn(f"MLflow log_params failed: {e}")


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
        mlflow.log_artifact(path)
    except Exception as e:
        warnings.warn(f"MLflow log_artifact failed: {e}")


def end_run():
    """End the active MLflow run."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.end_run()
    except Exception:
        pass
