#!/usr/bin/env python
"""
Worker script for parallel neural network training on GPUs.

This script trains a single model with a given configuration on a specific GPU.
It's designed to be called by the orchestrator with a GPU ID assigned.

Usage:
    python parallel_train_worker.py --config config.json --output-dir output/ --gpu-id 0
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pickle
from pathlib import Path

from fcm_quadrature.training.losses import create_loss
from fcm_quadrature.utils.reproducibility import set_global_seeds, capture_environment, save_manifest
from fcm_quadrature.utils import tracking

# Set TensorFlow logging level before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def setup_tensorflow(gpu_id=None, num_threads=4):
    """Configure TensorFlow to use a specific GPU.

    Parameters
    ----------
    gpu_id : int or None
        GPU device index to use. If None, use CPU only.
    num_threads : int
        Number of CPU threads for data loading/preprocessing.
    """
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    import tensorflow as tf

    # Set CPU threads for data pipeline
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)

    if gpu_id is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Allow memory growth so multiple models can share a GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # Already configured

    return tf


class ProgressCallback:
    """Custom callback to write training progress to a file."""

    def __init__(self, progress_file, total_epochs, model_name):
        self.progress_file = progress_file
        self.total_epochs = total_epochs
        self.model_name = model_name
        self.start_time = time.time()

    def __call__(self, epoch, logs=None):
        """Called at the end of each epoch."""
        logs = logs or {}
        elapsed = time.time() - self.start_time
        epochs_done = epoch + 1
        eta_seconds = (elapsed / epochs_done) * (self.total_epochs - epochs_done) if epochs_done > 0 else 0
        progress = {
            'model_name': self.model_name,
            'epoch': epochs_done,
            'total_epochs': self.total_epochs,
            'percent': round(100 * epochs_done / self.total_epochs, 1),
            'loss': float(logs.get('loss', 0)),
            'val_loss': float(logs.get('val_loss', 0)),
            'elapsed_seconds': round(elapsed, 1),
            'eta_seconds': round(eta_seconds, 1),
            'status': 'running'
        }
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f)
        except:
            pass  # Ignore write errors


def train_model(config, data_paths, output_dir, gpu_id=None, num_threads=4, verbose=0):
    """
    Train a single model with the given configuration.

    Parameters
    ----------
    config : dict
        Model configuration
    data_paths : dict
        Paths to training and validation data
    output_dir : str
        Directory to save outputs
    gpu_id : int or None
        GPU device index to use
    num_threads : int
        Number of CPU threads for data pipeline

    Returns
    -------
    dict
        Results dictionary
    """
    # Set seeds for reproducibility
    seed = config.get('seed', 42)
    set_global_seeds(seed)

    # Setup TensorFlow with GPU
    tf = setup_tensorflow(gpu_id, num_threads)

    # Import after TensorFlow setup
    from fcm_quadrature.training.data_loading import (
        preprocessData, buildDataset, saveHistory
    )
    from fcm_quadrature.models.fnn import (
        buildSequentialModel
    )
    from tensorflow.keras.layers import Normalization

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_name = config.get('model_name', 'model')

    # Save config
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Capture environment for reproducibility manifest
    env_info = capture_environment()

    # Start MLflow tracking (if available and experiment name is configured)
    mlflow_experiment = config.get('mlflow_experiment_name')
    mlflow_run = None
    if mlflow_experiment:
        mlflow_run = tracking.start_run(
            experiment_name=mlflow_experiment,
            run_name=model_name,
            tags={'gpu_id': str(gpu_id)}
        )
        tracking.log_config(config)
        tracking.log_environment(env_info)

    # Load and preprocess data
    data = preprocessData(
        data_paths['train'],
        data_paths['valid'],
        testSetPath=None,
        dtype=np.float32,
        num_outputs=config.get('num_outputs', 4),
        num_inputs=config.get('num_inputs', 12),
    )

    XTrain, yTrain, XValid, yValid = data

    # Build dataset
    trainSet, validSet, inputDim, stepPerEpoch = buildDataset(
        data,
        config['batch_size_train'],
        config['batch_size_valid']
    )

    # Create normalization layer (optional)
    use_normalization = config.get('use_normalization', True)
    if use_normalization:
        inputNormLayer = Normalization()
        inputNormLayer.adapt(XTrain)
    else:
        inputNormLayer = None
        print("Note: Input normalization disabled")

    # Build model
    layerSizeList = [config['model_width']] * config['model_depth']
    activationList = [config['activation']] * config['model_depth']

    # Support custom loss functions (mse, moment, combined)
    loss_type = config.get('loss_type', config.get('loss_func', 'mse'))
    loss_alpha = config.get('loss_alpha', 1.0)
    loss_beta = config.get('loss_beta', 0.5)
    loss_fn = create_loss(loss_type, alpha=loss_alpha, beta=loss_beta)

    model = buildSequentialModel(
        inputDim=inputDim,
        dtype=np.float32,
        layerSizeList=layerSizeList,
        activationList=activationList,
        weightInitializer=config['weight_initializer'],
        lossFunc=loss_fn,
        modelName=model_name,
        num_outputs=config.get('num_outputs', 4),
        learningRate=None,
        metricFunc=None,
        verbose=0,
        inputNormLayer=inputNormLayer,
        dropout_rate=config.get('dropout_rate', 0.1),
    )

    # Recompile with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['learning_rate'],
        clipnorm=1.0
    )
    model.compile(optimizer=optimizer, loss=loss_fn)

    total_params = model.count_params()

    # Checkpoint path
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.weights.h5')

    # Progress file for monitoring
    progress_file = os.path.join(output_dir, 'progress.json')

    # Write initial progress
    initial_progress = {
        'model_name': model_name,
        'epoch': 0,
        'total_epochs': config['num_epochs'],
        'percent': 0,
        'loss': 0,
        'val_loss': 0,
        'elapsed_seconds': 0,
        'eta_seconds': 0,
        'status': 'starting',
        'gpu_id': gpu_id,
    }
    with open(progress_file, 'w') as f:
        json.dump(initial_progress, f)

    # Create callbacks
    from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping

    # Progress callback
    progress_callback = ProgressCallback(progress_file, config['num_epochs'], model_name)
    lambda_cb = LambdaCallback(on_epoch_end=progress_callback)

    # Model checkpoint callback
    checkpoint_cb = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=0
    )

    # Early stopping: stop if val_loss doesn't improve for 500 epochs
    early_stop_cb = EarlyStopping(
        monitor='val_loss',
        patience=config.get('early_stopping_patience', 500),
        restore_best_weights=True,
        verbose=0
    )

    # MLflow callback (logs per-epoch metrics)
    mlflow_cb = tracking.create_keras_callback(log_interval=1)
    all_callbacks = [lambda_cb, checkpoint_cb, early_stop_cb]
    if mlflow_cb is not None and mlflow_run is not None:
        all_callbacks.append(mlflow_cb)

    # Train
    train_start = time.time()

    history = model.fit(
        trainSet,
        validation_data=validSet,
        epochs=config['num_epochs'],
        verbose=verbose,
        callbacks=all_callbacks
    )

    train_time = time.time() - train_start
    actual_epochs = len(history.history['loss'])
    stopped_early = actual_epochs < config['num_epochs']

    # Update progress to complete
    final_progress = {
        'model_name': model_name,
        'epoch': actual_epochs,
        'total_epochs': config['num_epochs'],
        'percent': 100,
        'loss': float(history.history['loss'][-1]),
        'val_loss': float(history.history['val_loss'][-1]),
        'elapsed_seconds': round(train_time, 1),
        'eta_seconds': 0,
        'status': 'complete',
        'stopped_early': stopped_early,
        'gpu_id': gpu_id,
    }
    with open(progress_file, 'w') as f:
        json.dump(final_progress, f)

    # Save history
    history_path = os.path.join(output_dir, 'history.pickle')
    saveHistory(history.history, history_path)

    # Save loss curve plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(history.history['loss'], label='Training Loss', alpha=0.8)
    ax.semilogy(history.history['val_loss'], label='Validation Loss', alpha=0.8)
    if stopped_early:
        ax.axvline(x=actual_epochs - 1, color='red', linestyle='--', alpha=0.5,
                   label=f'Early Stop (epoch {actual_epochs})')
    best_epoch_idx = int(np.argmin(history.history['val_loss']))
    ax.axvline(x=best_epoch_idx, color='green', linestyle='--', alpha=0.5,
               label=f'Best (epoch {best_epoch_idx + 1})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title(f'Loss Curves - {model_name}\n'
                 f'Best Val Loss: {min(history.history["val_loss"]):.6e} at epoch {best_epoch_idx + 1}'
                 + (f' | Early stopped at epoch {actual_epochs}' if stopped_early else ''))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150)
    plt.close(fig)

    # Load best weights (early stopping already restores them, but load checkpoint if available)
    try:
        model.load_weights(checkpoint_path)
    except:
        pass  # EarlyStopping already restored best weights

    # Get metrics
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    best_val_loss = min(val_loss)
    best_epoch = int(np.argmin(val_loss))

    # Measure inference time
    inference_times = []
    for _ in range(5):
        start = time.time()
        _ = model.predict(validSet, verbose=0)
        inference_times.append(time.time() - start)

    mean_inference = np.mean(inference_times)
    std_inference = np.std(inference_times)

    # Calculate prediction metrics
    yPredValid = model.predict(validSet, verbose=0)

    # Save predictions and ground truth for analysis script
    np.save(os.path.join(output_dir, 'y_pred_valid.npy'), yPredValid)
    np.save(os.path.join(output_dir, 'y_true_valid.npy'), yValid)

    # Relative error
    epsilon = 1e-10
    relative_errors = (yPredValid - yValid) / (np.abs(yValid) + epsilon)
    abs_errors = np.abs(yPredValid - yValid)

    mean_abs_rel_error = float(np.mean(np.abs(relative_errors)))
    median_abs_rel_error = float(np.median(np.abs(relative_errors)))
    max_abs_rel_error = float(np.max(np.abs(relative_errors)))
    mean_abs_error = float(np.mean(abs_errors))
    median_abs_error = float(np.median(abs_errors))

    # Save model
    model_save_path = os.path.join(output_dir, 'saved_model.keras')
    model.save(model_save_path)

    # Compile results
    results = {
        'model_name': model_name,
        'config': config,
        'total_params': int(total_params),
        'input_dim': int(inputDim),
        'training_time_seconds': train_time,
        'actual_epochs': actual_epochs,
        'stopped_early': stopped_early,
        'inference_time_mean': mean_inference,
        'inference_time_std': std_inference,
        'best_val_loss': float(best_val_loss),
        'best_epoch': best_epoch,
        'final_train_loss': float(train_loss[-1]),
        'final_val_loss': float(val_loss[-1]),
        'mean_abs_rel_error': mean_abs_rel_error,
        'median_abs_rel_error': median_abs_rel_error,
        'max_abs_rel_error': max_abs_rel_error,
        'mean_abs_error': mean_abs_error,
        'median_abs_error': median_abs_error,
        'output_dir': output_dir,
        'gpu_id': gpu_id,
        'num_threads': num_threads,
    }

    # Save results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Save reproducibility manifest
    save_manifest(output_dir, config, env_info)

    # Log final metrics and artifacts to MLflow
    if mlflow_run is not None:
        tracking.log_metrics({
            'best_val_loss': float(best_val_loss),
            'mean_abs_rel_error': mean_abs_rel_error,
            'median_abs_rel_error': median_abs_rel_error,
            'training_time_seconds': train_time,
            'actual_epochs': actual_epochs,
        })
        tracking.log_artifact(results_path)
        tracking.log_artifact(config_path)
        loss_curve_path = os.path.join(output_dir, 'loss_curve.png')
        if os.path.exists(loss_curve_path):
            tracking.log_artifact(loss_curve_path)
        tracking.end_run()

    # Clean up
    del model
    tf.keras.backend.clear_session()

    return results


def main():
    parser = argparse.ArgumentParser(description='Train single neural network on GPU')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON file')
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to training CSV')
    parser.add_argument('--valid-data', type=str, required=True,
                       help='Path to validation CSV')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--gpu-id', type=int, default=None,
                       help='GPU device index to use (None for CPU)')
    parser.add_argument('--num-threads', type=int, default=4,
                       help='Number of CPU threads for data pipeline')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    data_paths = {
        'train': args.train_data,
        'valid': args.valid_data,
    }

    # Train
    results = train_model(config, data_paths, args.output_dir,
                         gpu_id=args.gpu_id, num_threads=args.num_threads)

    # Print summary
    print(f"Training complete: {results['model_name']}")
    print(f"  Best val loss: {results['best_val_loss']:.6e}")
    print(f"  Training time: {results['training_time_seconds']:.1f}s")
    print(f"  Mean rel error: {results['mean_abs_rel_error']:.6f}")
    print(f"  Actual epochs: {results['actual_epochs']}"
          + (f" (early stopped)" if results['stopped_early'] else ""))
    print(f"  GPU: {results['gpu_id']}")


if __name__ == '__main__':
    main()
