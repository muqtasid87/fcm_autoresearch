"""
Evaluation and visualization for FT-Transformer models.

Functions:
    evaluate_model - Evaluate a model and generate metrics, plots, and output files.
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from LiangNet_MultiOutput import preprocessData
from fcm_quadrature.models.transformer import FTTransformer, CUSTOM_OBJECTS


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


def evaluate_model(
    model=None,
    model_path=None,
    XEval=None,
    yEval=None,
    eval_data_path=None,
    output_dir=None,
    batch_size=4096,
    num_outputs=4,
    history=None,
    train_config=None,
):
    """
    Evaluate a transformer model and generate metrics, plots, and output files.

    Can be called in two modes:
    1. After training: pass model, XEval, yEval, history, train_config
    2. Standalone:     pass model_path, eval_data_path

    Args:
        model: Keras model object (if already in memory).
        model_path: Path to saved .keras model (loads from disk).
        XEval: Input features array.
        yEval: Target labels array.
        eval_data_path: Path to evaluation CSV (loads from disk).
        output_dir: Output directory for results.
        batch_size: Batch size for inference.
        num_outputs: Number of model outputs.
        history: Keras training history (for loss curves, only after training).
        train_config: Training config dict (for metadata in results).

    Returns:
        results: dict with all evaluation metrics.
        output_dir: path to output directory.
    """

    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    # --- Resolve model ---
    if model is None:
        if model_path is None:
            raise ValueError("Either model or model_path must be provided.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
        print(f"Loaded model from: {model_path}")

    total_params = model.count_params()
    print(f"Total parameters: {total_params:,}")

    # --- Resolve data ---
    if XEval is None or yEval is None:
        if eval_data_path is None:
            raise ValueError("Either (XEval, yEval) or eval_data_path must be provided.")
        if not os.path.exists(eval_data_path):
            raise FileNotFoundError(f"Data not found: {eval_data_path}")
        # Workaround: preprocessData expects both train and valid paths
        data = preprocessData(
            eval_data_path, eval_data_path,
            testSetPath=None, dtype=np.float32, num_outputs=num_outputs,
        )
        _, _, XEval, yEval = data
        print(f"Loaded data from: {eval_data_path}")

    print(f"Samples: {XEval.shape[0]:,}, Features: {XEval.shape[1]}, Outputs: {yEval.shape[1]}")

    # --- Create output directory ---
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_path:
            model_name = Path(model_path).parent.name
            output_dir = f'evaluation_results/{model_name}_{timestamp}'
        else:
            output_dir = f'transformer_results/train_{timestamp}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Create finite dataset for inference ---
    eval_dataset = create_dataset(XEval, yEval, batch_size, shuffle=False, repeat=False)

    # --- Measure inference time ---
    print("\nMeasuring inference time...")
    inference_times = []
    for _ in range(10):
        start = time.time()
        _ = model.predict(eval_dataset, verbose=0)
        inference_times.append(time.time() - start)

    mean_inference = np.mean(inference_times)
    std_inference = np.std(inference_times)
    print(f"Inference time: {mean_inference:.4f} +/- {std_inference:.4f}s")
    print(f"Throughput: {XEval.shape[0]/mean_inference:.1f} samples/s")

    # --- Generate predictions ---
    print("Generating predictions...")
    yPred = model.predict(eval_dataset, verbose=0)

    # --- Calculate metrics ---
    epsilon = 1e-10
    abs_errors = np.abs(yPred - yEval)
    relative_errors = abs_errors / (np.abs(yEval) + epsilon)

    mean_abs_error = float(np.mean(abs_errors))
    median_abs_error = float(np.median(abs_errors))
    mean_rel_error = float(np.mean(relative_errors))
    median_rel_error = float(np.median(relative_errors))
    max_rel_error = float(np.max(relative_errors))
    mse = float(np.mean(np.square(yPred - yEval)))
    mae = float(np.mean(abs_errors))

    # R-squared scores per output
    r2_scores = []
    for idx in range(yEval.shape[1]):
        ss_res = np.sum((yEval[:, idx] - yPred[:, idx]) ** 2)
        ss_tot = np.sum((yEval[:, idx] - np.mean(yEval[:, idx])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2_scores.append(float(r2))

    # Area fraction verification
    pred_weight_sum = np.sum(yPred, axis=1)
    true_weight_sum = np.sum(yEval, axis=1)
    pred_af = pred_weight_sum / 4.0
    true_af = true_weight_sum / 4.0
    af_error = np.abs(pred_af - true_af)

    print(f"\nMetrics:")
    print(f"  MSE: {mse:.6e}")
    print(f"  MAE: {mae:.6e}")
    print(f"  Mean absolute error: {mean_abs_error:.6e}")
    print(f"  Mean relative error: {mean_rel_error:.6f}")
    print(f"  Median relative error: {median_rel_error:.6f}")
    for idx, r2 in enumerate(r2_scores):
        print(f"  R2 (output {idx+1}): {r2:.6f}")
    print(f"  Mean |AF error|: {np.mean(af_error):.6e}")
    print(f"  Max |AF error|: {np.max(af_error):.6e}")

    # --- Save predictions ---
    np.save(os.path.join(output_dir, 'y_pred.npy'), yPred)
    np.save(os.path.join(output_dir, 'y_true.npy'), yEval)
    np.save(os.path.join(output_dir, 'area_fractions.npy'), true_af)

    # --- Visualizations ---
    print("\nCreating visualizations...")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    weight_names = ['w1', 'w2', 'w3', 'w4']

    # Loss curves (only if history is available, i.e. after training)
    if history is not None:
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        best_val_loss = min(val_loss)
        best_epoch = int(np.argmin(val_loss))
        actual_epochs = len(train_loss)
        stopped_early = actual_epochs < (train_config or {}).get('num_epochs', actual_epochs + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('white')

        ax = axes[0]
        ax.semilogy(train_loss, label='Train Loss', alpha=0.8)
        ax.semilogy(val_loss, label='Val Loss', alpha=0.8)
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5,
                   label=f'Best (epoch {best_epoch+1})')
        if stopped_early:
            ax.axvline(x=actual_epochs-1, color='red', linestyle='--', alpha=0.5,
                       label='Early stop')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title(f'Loss Curves\nBest Val Loss: {best_val_loss:.6e}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(history.history['mae'], label='Train MAE', alpha=0.8)
        ax.plot(history.history['val_mae'], label='Val MAE', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title('Mean Absolute Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150)
        plt.close()
        print("  Saved: loss_curves.png")

    # Predictions scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor('white')

    for idx in range(4):
        ax = axes[idx // 2, idx % 2]
        ax.scatter(yEval[:, idx], yPred[:, idx], s=1, alpha=0.2)
        lims = [min(yEval[:, idx].min(), yPred[:, idx].min()),
                max(yEval[:, idx].max(), yPred[:, idx].max())]
        ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect')
        ax.set_xlabel(f'True {weight_names[idx]}')
        ax.set_ylabel(f'Predicted {weight_names[idx]}')
        ax.set_title(f'{weight_names[idx]}: Predicted vs True')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, f'R2 = {r2_scores[idx]:.6f}', transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Predictions vs Ground Truth', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'), dpi=150)
    plt.close()
    print("  Saved: predictions.png")

    # Error distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    for idx in range(4):
        ax = axes[idx // 2, idx % 2]
        errors = yPred[:, idx] - yEval[:, idx]
        ax.hist(errors, bins=200, alpha=0.7, density=True)
        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_xlabel('Error (Pred - True)')
        ax.set_ylabel('Density')
        ax.set_title(f'{weight_names[idx]}: mean={np.mean(errors):.4e}, std={np.std(errors):.4e}')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Error Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distributions.png'), dpi=150)
    plt.close()
    print("  Saved: error_distributions.png")

    # Area fraction analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('white')

    ax = axes[0]
    ax.scatter(true_af, pred_af, s=1, alpha=0.2)
    lims = [min(true_af.min(), pred_af.min()), max(true_af.max(), pred_af.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5)
    ax.set_xlabel('True Area Fraction')
    ax.set_ylabel('Predicted Area Fraction')
    ax.set_title('Area Fraction: Predicted vs True')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(af_error, bins=200, alpha=0.7, density=True)
    ax.set_xlabel('|AF Error|')
    ax.set_ylabel('Density')
    ax.set_title(f'AF Error Distribution\nmean={np.mean(af_error):.4e}')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.scatter(true_af, af_error, s=1, alpha=0.2)
    ax.set_xlabel('True Area Fraction')
    ax.set_ylabel('|AF Error|')
    ax.set_title('AF Error vs True AF')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'area_fraction_analysis.png'), dpi=150)
    plt.close()
    print("  Saved: area_fraction_analysis.png")

    # --- Compile results ---
    results = {
        'mse': mse,
        'mae': mae,
        'mean_abs_error': mean_abs_error,
        'median_abs_error': median_abs_error,
        'mean_rel_error': mean_rel_error,
        'median_rel_error': median_rel_error,
        'max_rel_error': max_rel_error,
        'r2_scores': r2_scores,
        'mean_af_error': float(np.mean(af_error)),
        'median_af_error': float(np.median(af_error)),
        'max_af_error': float(np.max(af_error)),
        'inference_time_mean': mean_inference,
        'inference_time_std': std_inference,
        'total_params': int(total_params),
        'num_samples': int(XEval.shape[0]),
        'output_dir': output_dir,
    }

    # Add training-specific fields if available
    if history is not None:
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        results['best_val_loss'] = float(min(val_loss))
        results['best_epoch'] = int(np.argmin(val_loss))
        results['final_train_loss'] = float(train_loss[-1])
        results['final_val_loss'] = float(val_loss[-1])
    if train_config is not None:
        results['config'] = train_config
        # Promote key training fields to top level for easy access
        results['model_name'] = (
            f"FTTransformer_d{train_config.get('d_model', '?')}"
            f"_h{train_config.get('num_heads', '?')}"
            f"_l{train_config.get('num_layers', '?')}"
        )
        for key in ('training_time_seconds', 'actual_epochs', 'stopped_early'):
            if key in train_config:
                results[key] = train_config[key]
    if model_path is not None:
        results['model_path'] = str(model_path)
    if eval_data_path is not None:
        results['data_path'] = str(eval_data_path)

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print("  Saved: results.json")

    # --- Text summary ---
    with open(os.path.join(output_dir, 'SUMMARY.txt'), 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TRANSFORMER MODEL EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        if model_path:
            f.write(f"Model: {model_path}\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Samples: {XEval.shape[0]:,}\n\n")

        if train_config:
            f.write("Architecture:\n")
            f.write(f"  d_model: {train_config.get('d_model', 'N/A')}\n")
            f.write(f"  num_heads: {train_config.get('num_heads', 'N/A')}\n")
            f.write(f"  num_layers: {train_config.get('num_layers', 'N/A')}\n")
            f.write(f"  d_ff: {train_config.get('d_ff', 'N/A')}\n")
            f.write(f"  dropout: {train_config.get('dropout_rate', 'N/A')}\n\n")

        if 'best_val_loss' in results:
            f.write("Training:\n")
            f.write(f"  Epochs: {results.get('actual_epochs', 'N/A')}")
            if results.get('stopped_early'):
                f.write(" (early stopped)")
            f.write("\n")
            t = results.get('training_time_seconds', 0)
            f.write(f"  Time: {t:.2f}s ({t/60:.2f} min)\n")
            f.write(f"  Best val loss: {results['best_val_loss']:.6e} (epoch {results['best_epoch']+1})\n")
            f.write(f"  Final train loss: {results['final_train_loss']:.6e}\n")
            f.write(f"  Final val loss: {results['final_val_loss']:.6e}\n\n")

        f.write("Performance:\n")
        f.write(f"  MSE: {mse:.6e}\n")
        f.write(f"  MAE: {mae:.6e}\n")
        f.write(f"  Mean absolute error: {mean_abs_error:.6e}\n")
        f.write(f"  Mean relative error: {mean_rel_error:.6f}\n")
        f.write(f"  Median relative error: {median_rel_error:.6f}\n\n")

        f.write("R2 Scores:\n")
        for idx, r2 in enumerate(r2_scores):
            f.write(f"  Output {idx+1}: {r2:.6f}\n")
        f.write("\n")

        f.write("Area Fraction:\n")
        f.write(f"  Mean |AF error|: {np.mean(af_error):.6e}\n")
        f.write(f"  Max |AF error|: {np.max(af_error):.6e}\n\n")

        f.write("Inference:\n")
        f.write(f"  Time: {mean_inference:.4f} +/- {std_inference:.4f}s\n")
        f.write(f"  Throughput: {XEval.shape[0]/mean_inference:.1f} samples/s\n")

    print("  Saved: SUMMARY.txt")

    return results, output_dir
