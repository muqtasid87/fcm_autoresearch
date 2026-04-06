#!/usr/bin/env python
# coding: utf-8
"""
Quick Model Test Script

Train a single small model quickly to verify everything works.
Generates all visualizations and saves results.

Estimated time: 5-15 minutes
"""

import numpy as np
import os
import json
import time
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from moment_loss import create_loss
from LiangNet_MultiOutput import (
    preprocessData, buildDataset, buildSequentialModel,
    trainModel, saveHistory
)
from tensorflow.keras.layers import Normalization


def quick_test_model(
    train_path='Data/Training_1M_NoVertices.csv',
    valid_path='Data/Valid_1M_NoVertices.csv',
    num_epochs=20,
    model_width=256,
    model_depth=4,
    learning_rate=1e-3,
    lossFunc = create_loss('combined', alpha = 1.0, beta = 1.0)
):
    """
    Run a quick model test with visualizations.
    
    Parameters:
    -----------
    train_path : str
        Path to training CSV
    valid_path : str
        Path to validation CSV
    num_epochs : int
        Number of training epochs (default: 200 for quick test)
    model_width : int
        Number of neurons per layer
    model_depth : int
        Number of hidden layers
    learning_rate : float
        Learning rate for training
    """
    
    print("="*80)
    print("QUICK MODEL TEST")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {model_width}x{model_depth}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Training data: {train_path}")
    print(f"  Validation data: {valid_path}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'quick_test_{timestamp}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # ========================================================================
    # 1. LOAD AND PREPROCESS DATA
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 1: Loading Data")
    print("="*80)

    dtype = np.float32
    num_outputs = 4

    data = preprocessData(
        train_path,
        valid_path,
        testSetPath=None,
        dtype=dtype,
        num_outputs=num_outputs,
    )

    XTrain, yTrain, XValid, yValid = data
    print(f"✓ Training samples: {XTrain.shape[0]:,}")
    print(f"✓ Validation samples: {XValid.shape[0]:,}")
    print(f"✓ Input features: {XTrain.shape[1]}")
    print(f"✓ Output targets: {yTrain.shape[1]}")
    
    # ========================================================================
    # 2. BUILD DATASET
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: Building Dataset")
    print("="*80)
    
    batch_size_train = 2**15
    batch_size_valid = 2**16
    
    trainSet, validSet, inputDim, stepPerEpoch = buildDataset(
        data, batch_size_train, batch_size_valid
    )
    
    print(f"✓ Input dimension: {inputDim}")
    print(f"✓ Steps per epoch: {stepPerEpoch}")
    print(f"✓ Batch size (train): {batch_size_train}")
    print(f"✓ Batch size (valid): {batch_size_valid}")
    
    # ========================================================================
    # 3. BUILD MODEL
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: Building Model")
    print("="*80)
    
    # Create input normalization layer (set to False to disable)
    use_normalization = True
    if use_normalization:
        inputNormLayer = Normalization()
        inputNormLayer.adapt(XTrain)
    else:
        inputNormLayer = None
        print("Note: Input normalization disabled")
    
    # Build model
    layerSizeList = [model_width] * model_depth
    activationList = ['relu'] * model_depth
    
    model_name = f'quick_test_{model_width}x{model_depth}_{num_epochs}epochs'
    
    model = buildSequentialModel(
        inputDim=inputDim,
        dtype=dtype,
        layerSizeList=layerSizeList,
        activationList=activationList,
        weightInitializer='glorot_uniform',
        lossFunc='mse',
        modelName=model_name,
        num_outputs=4,
        learningRate=learning_rate,
        metricFunc=None,
        verbose=1,
        inputNormLayer=inputNormLayer,
    )
    
    total_params = model.count_params()
    print(f"\n✓ Model built successfully")
    print(f"✓ Total parameters: {total_params:,}")
    
    # Save model architecture
    with open(f'{output_dir}/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # ========================================================================
    # 4. TRAIN MODEL
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 4: Training Model")
    print("="*80)
    print(f"\nTraining for {num_epochs} epochs...")
    print("This will take approximately:")
    estimated_time = stepPerEpoch * num_epochs * 0.1  # rough estimate
    print(f"  {estimated_time/60:.1f} minutes")
    
    checkpoint_path = f'{output_dir}/best_model.keras'
    history_path = f'{output_dir}/history.pickle'
    
    train_start = time.time()
    
    history = trainModel(
        model,
        trainSet,
        validSet,
        num_epochs,
        lossCheckpointPath=checkpoint_path,
        initalCheckpointLoss=None,
        verbose=1,
        callbackVerbose=1,
    )
    
    train_end = time.time()
    training_time = train_end - train_start
    
    print(f"\n✓ Training completed!")
    print(f"✓ Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Save history
    saveHistory(history.history, history_path)
    
    # ========================================================================
    # 5. LOAD BEST WEIGHTS AND EVALUATE
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 5: Evaluation")
    print("="*80)
    
    # Load best weights
    try:
        model.load_weights(checkpoint_path)
        print("✓ Loaded best weights from checkpoint")
    except:
        print("⚠ Warning: Could not load checkpoint, using final weights")
    
    # Get training metrics
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    best_val_loss = min(val_loss)
    best_epoch = np.argmin(val_loss)
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    
    print(f"\nTraining Metrics:")
    print(f"  Best validation loss: {best_val_loss:.6f} (epoch {best_epoch + 1})")
    print(f"  Final train loss: {final_train_loss:.6f}")
    print(f"  Final validation loss: {final_val_loss:.6f}")
    
    # Measure inference time
    print("\nMeasuring inference time...")
    inference_times = []
    for i in range(10):
        start = time.time()
        _ = model.predict(validSet, verbose=0)
        end = time.time()
        inference_times.append(end - start)
    
    mean_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    print(f"✓ Inference time: {mean_inference_time:.4f} ± {std_inference_time:.4f} seconds")
    
    # Generate predictions
    print("\nGenerating predictions...")
    yPredValid = model.predict(validSet, verbose=0)
    
    # Calculate prediction errors (linear space)
    epsilon = 1e-10
    relative_errors = (yPredValid - yValid) / (np.abs(yValid) + epsilon)
    mean_abs_rel_error = np.mean(np.abs(relative_errors))
    median_abs_rel_error = np.median(np.abs(relative_errors))
    max_abs_rel_error = np.max(np.abs(relative_errors))
    
    print(f"\nPrediction Metrics:")
    print(f"  Mean absolute relative error: {mean_abs_rel_error:.6f}")
    print(f"  Median absolute relative error: {median_abs_rel_error:.6f}")
    print(f"  Max absolute relative error: {max_abs_rel_error:.6f}")
    
    # ========================================================================
    # 6. SAVE MODEL
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 6: Saving Model")
    print("="*80)
    
    model_save_path = f'{output_dir}/saved_model.keras'
    model.save(model_save_path)
    print(f"✓ Model saved to: {model_save_path}")
    
    # ========================================================================
    # 7. CREATE VISUALIZATIONS
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 7: Creating Visualizations")
    print("="*80)
    
    # 7.1 Loss Curves
    print("\n1. Loss curves...")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    ax.plot(train_loss, label='Training Loss', linewidth=2)
    ax.plot(val_loss, label='Validation Loss', linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title(f'Loss Curves\nBest Val Loss: {best_val_loss:.6f} at Epoch {best_epoch+1}', 
                fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved to {output_dir}/loss_curves.png")
    
    # 7.2 Predictions for all 4 outputs
    print("\n2. Prediction plots (all 4 outputs)...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')

    for idx in range(4):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        y_true_plot = yValid[:, idx]
        y_pred_plot = yPredValid[:, idx]

        ax.scatter(y_true_plot, y_pred_plot, alpha=0.3, s=1)
        min_val = min(y_true_plot.min(), y_pred_plot.min())
        max_val = max(y_true_plot.max(), y_pred_plot.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                label='Perfect Prediction')
        ax.set_xlabel('Ground Truth', fontsize=11)
        ax.set_ylabel('Prediction', fontsize=11)
        ax.set_title(f'Output {idx} - Predictions vs Ground Truth', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'All 4 Outputs - Prediction Quality', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/predictions_all_outputs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved to {output_dir}/predictions_all_outputs.png")
    
    # 7.3 Error distributions for all outputs
    print("\n3. Error distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    
    for idx in range(4):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        rel_err = relative_errors[:, idx]
        
        ax.hist(rel_err, bins=100, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(x=np.mean(rel_err), color='g', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(rel_err):.4f}')
        ax.set_xlabel('Relative Error', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Output {idx} - Error Distribution', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Error Distributions - All 4 Outputs', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved to {output_dir}/error_distributions.png")
    
    # 7.4 Detailed view for first output
    print("\n4. Detailed plots for Output 0...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    y_true_plot = yValid[:, 0]
    y_pred_plot = yPredValid[:, 0]
    rel_err = relative_errors[:, 0]

    # Time series
    ax = axes[0, 0]
    indices = np.arange(len(y_true_plot))
    ax.plot(indices, y_true_plot, label='Ground Truth', alpha=0.7, linewidth=1)
    ax.plot(indices, y_pred_plot, label='Prediction', alpha=0.7, linewidth=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Value')
    ax.set_title('Time Series View - Output 0')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter
    ax = axes[0, 1]
    ax.scatter(y_true_plot, y_pred_plot, alpha=0.3, s=1)
    min_val = min(y_true_plot.min(), y_pred_plot.min())
    max_val = max(y_true_plot.max(), y_pred_plot.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Prediction')
    ax.set_title('Scatter Plot - Output 0')
    ax.grid(True, alpha=0.3)

    # Relative error over time
    ax = axes[1, 0]
    ax.plot(indices, rel_err, linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Relative Error')
    ax.set_title('Relative Error - Output 0')
    ax.set_ylim([-1, 1])
    ax.grid(True, alpha=0.3)

    # Error histogram
    ax = axes[1, 1]
    ax.hist(rel_err, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Relative Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Histogram - Output 0')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Detailed Analysis - Output 0', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/detailed_output_0.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved to {output_dir}/detailed_output_0.png")
    
    # ========================================================================
    # 8. SAVE RESULTS SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 8: Saving Results Summary")
    print("="*80)
    
    # Create summary dictionary
    results = {
        'model_name': model_name,
        'timestamp': timestamp,
        'configuration': {
            'model_width': model_width,
            'model_depth': model_depth,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size_train': batch_size_train,
            'batch_size_valid': batch_size_valid,
        },
        'data_info': {
            'training_samples': int(XTrain.shape[0]),
            'validation_samples': int(XValid.shape[0]),
            'input_features': int(XTrain.shape[1]),
            'output_targets': int(yTrain.shape[1]),
        },
        'model_info': {
            'total_parameters': int(total_params),
            'input_dim': int(inputDim),
        },
        'training_metrics': {
            'training_time_seconds': float(training_time),
            'best_val_loss': float(best_val_loss),
            'best_epoch': int(best_epoch),
            'final_train_loss': float(final_train_loss),
            'final_val_loss': float(final_val_loss),
        },
        'prediction_metrics': {
            'mean_abs_rel_error': float(mean_abs_rel_error),
            'median_abs_rel_error': float(median_abs_rel_error),
            'max_abs_rel_error': float(max_abs_rel_error),
        },
        'inference_metrics': {
            'mean_time_seconds': float(mean_inference_time),
            'std_time_seconds': float(std_inference_time),
        },
    }
    
    # Save as JSON
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✓ Results saved to {output_dir}/results.json")
    
    # Save as text summary
    with open(f'{output_dir}/SUMMARY.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("QUICK MODEL TEST SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Width: {model_width}, Depth: {model_depth}\n")
        f.write(f"  Epochs: {num_epochs}\n")
        f.write(f"  Learning Rate: {learning_rate}\n")
        f.write(f"  Batch Size: {batch_size_train}\n\n")
        
        f.write("Model Info:\n")
        f.write(f"  Total Parameters: {total_params:,}\n")
        f.write(f"  Input Dimension: {inputDim}\n")
        f.write(f"  Output Dimension: 4\n\n")
        
        f.write("Training Results:\n")
        f.write(f"  Training Time: {training_time:.2f}s ({training_time/60:.2f} min)\n")
        f.write(f"  Best Val Loss: {best_val_loss:.6f} (epoch {best_epoch+1})\n")
        f.write(f"  Final Train Loss: {final_train_loss:.6f}\n")
        f.write(f"  Final Val Loss: {final_val_loss:.6f}\n\n")
        
        f.write("Prediction Quality:\n")
        f.write(f"  Mean Abs Rel Error: {mean_abs_rel_error:.6f}\n")
        f.write(f"  Median Abs Rel Error: {median_abs_rel_error:.6f}\n")
        f.write(f"  Max Abs Rel Error: {max_abs_rel_error:.6f}\n\n")
        
        f.write("Inference Performance:\n")
        f.write(f"  Mean Time: {mean_inference_time:.4f}s\n")
        f.write(f"  Std Dev: {std_inference_time:.4f}s\n\n")
        
        f.write("Output Files:\n")
        f.write(f"  - saved_model.keras\n")
        f.write(f"  - best_model.keras\n")
        f.write(f"  - history.pickle\n")
        f.write(f"  - loss_curves.png\n")
        f.write(f"  - predictions_all_outputs.png\n")
        f.write(f"  - error_distributions.png\n")
        f.write(f"  - detailed_output_0.png\n")
        f.write(f"  - results.json\n")
        f.write(f"  - model_summary.txt\n")
    
    print(f"✓ Summary saved to {output_dir}/SUMMARY.txt")
    
    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    
    print("\n" + "="*80)
    print("TEST COMPLETE! ✅")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nFiles created:")
    print(f"  📊 loss_curves.png")
    print(f"  📊 predictions_all_outputs.png")
    print(f"  📊 error_distributions.png")
    print(f"  📊 detailed_output_0.png")
    print(f"  📄 SUMMARY.txt")
    print(f"  📄 results.json")
    print(f"  💾 saved_model.keras")
    print(f"  💾 best_model.keras")
    print(f"  💾 history.pickle")
    print(f"  📝 model_summary.txt")
    
    print(f"\nKey Results:")
    print(f"  ✓ Training time: {training_time/60:.1f} minutes")
    print(f"  ✓ Best validation loss: {best_val_loss:.6f}")
    print(f"  ✓ Mean relative error: {mean_abs_rel_error:.6f}")
    print(f"  ✓ Inference time: {mean_inference_time:.4f}s")
    
    print("\nTo view results:")
    print(f"  cat {output_dir}/SUMMARY.txt")
    print(f"  # Or open the PNG files to see plots")
    
    print("\nTo load the model:")
    print(f"  import tensorflow as tf")
    print(f"  model = tf.keras.models.load_model('{output_dir}/saved_model.keras')")
    
    print("="*80 + "\n")
    
    return results, output_dir


if __name__ == "__main__":
    # Run quick test
    results, output_dir = quick_test_model(
        train_path='Data/Training_1M_NoVertices.csv',
        valid_path='Data/Valid_1M_NoVertices.csv',
        num_epochs=30,  # Quick test - change to 500 or 1000 for better results
        model_width=256,
        model_depth=4,
        learning_rate=1e-3,

    )
    
    print("Done!")
