#!/usr/bin/env python
# coding: utf-8
"""
Comprehensive Hyperparameter Search Script for Multi-Output Neural Networks

This script trains multiple neural networks with different hyperparameters,
logs all metrics, saves graphs, measures inference time, and identifies
the best performing model.
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

# Import custom modules
from LiangNet_MultiOutput import (
    preprocessData, buildDataset, buildSequentialModel,
    trainModel, saveHistory, loadHistory
)
from tensorflow.keras.layers import Normalization


class HyperparameterSearch:
    """
    Class to manage hyperparameter search for neural networks.
    """
    
    def __init__(self, base_output_dir='hyperparameter_search'):
        """
        Initialize hyperparameter search.
        
        Parameters:
        -----------
        base_output_dir : str
            Base directory for all outputs
        """
        self.base_output_dir = base_output_dir
        self.results = []
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.search_dir = os.path.join(base_output_dir, f'search_{self.start_time}')
        
        # Create base directory
        Path(self.search_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Hyperparameter search initialized")
        print(f"Output directory: {self.search_dir}")
    
    def define_hyperparameter_grid(self):
        """
        Define the grid of hyperparameters to search.
        
        Returns a list of hyperparameter configurations.
        """
        hyperparameter_grid = []
        
        # Define ranges for each hyperparameter
        model_widths = [256, 512, 1024]
        model_depths = [4, 6, 8]
        learning_rates = [1e-3, 5e-4, 1e-4]
        batch_sizes = [2**14, 2**15, 2**16]
        
        # Generate all combinations
        for width in model_widths:
            for depth in model_depths:
                for lr in learning_rates:
                    for batch_size in batch_sizes:
                        config = {
                            'model_width': width,
                            'model_depth': depth,
                            'learning_rate': lr,
                            'batch_size_train': batch_size,
                            'batch_size_valid': 2**16,
                            'num_epochs': 1000,
                            'weight_initializer': 'glorot_uniform',
                            'activation': 'relu',
                            'loss_func': 'mse',
                        }
                        hyperparameter_grid.append(config)
        
        return hyperparameter_grid
    
    def train_single_model(self, config, data, model_idx, total_models):
        """
        Train a single model with given hyperparameters.
        
        Parameters:
        -----------
        config : dict
            Hyperparameter configuration
        data : tuple
            Preprocessed data (XTrain, yTrain, XValid, yValid)
        model_idx : int
            Index of current model
        total_models : int
            Total number of models to train
        
        Returns:
        --------
        results_dict : dict
            Dictionary containing all results and metrics
        """
        print(f"\n{'='*80}")
        print(f"Training Model {model_idx + 1}/{total_models}")
        print(f"{'='*80}")
        
        # Create model name and directories
        model_name = (f"model_{model_idx:03d}_"
                     f"w{config['model_width']}_"
                     f"d{config['model_depth']}_"
                     f"lr{config['learning_rate']:.0e}_"
                     f"bs{config['batch_size_train']}")
        
        model_dir = os.path.join(self.search_dir, model_name)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint_dir = os.path.join(model_dir, 'checkpoints')
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Paths for saving
        config_path = os.path.join(model_dir, 'config.json')
        history_path = os.path.join(model_dir, 'history.pickle')
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.keras')  # .keras extension required for Keras 3.x
        loss_plot_path = os.path.join(model_dir, 'loss_curves.png')
        prediction_plot_path = os.path.join(model_dir, 'predictions.png')
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Build dataset
        trainSet, validSet, inputDim, stepPerEpoch = buildDataset(
            data,
            config['batch_size_train'],
            config['batch_size_valid']
        )
        
        print(f"Input dimension: {inputDim}")
        print(f"Steps per epoch: {stepPerEpoch}")
        
        # Create input normalization layer (optional)
        XTrain, yTrain, XValid, yValid = data
        use_normalization = config.get('use_normalization', True)
        if use_normalization:
            inputNormLayer = Normalization()
            inputNormLayer.adapt(XTrain)
        else:
            inputNormLayer = None
        
        # Build model
        layerSizeList = [config['model_width']] * config['model_depth']
        activationList = [config['activation']] * config['model_depth']
        
        model = buildSequentialModel(
            inputDim=inputDim,
            dtype=np.float32,
            layerSizeList=layerSizeList,
            activationList=activationList,
            weightInitializer=config['weight_initializer'],
            lossFunc=config['loss_func'],
            modelName=model_name,
            num_outputs=4,
            learningRate=config['learning_rate'],
            metricFunc=None,
            verbose=0,
            inputNormLayer=inputNormLayer,
        )
        
        # Count parameters
        total_params = model.count_params()
        print(f"Total parameters: {total_params:,}")
        
        # Train model
        print("\nTraining started...")
        train_start_time = time.time()
        
        history = trainModel(
            model,
            trainSet,
            validSet,
            config['num_epochs'],
            lossCheckpointPath=checkpoint_path,
            initalCheckpointLoss=None,
            verbose=0,
            callbackVerbose=1,
        )
        
        train_end_time = time.time()
        training_time = train_end_time - train_start_time
        
        print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Save history
        saveHistory(history.history, history_path)
        
        # Load best weights
        try:
            model.load_weights(checkpoint_path)
            print("Loaded best weights from checkpoint")
        except:
            print("Warning: Could not load checkpoint, using final weights")
        
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
        num_inference_runs = 10
        
        for i in range(num_inference_runs):
            start = time.time()
            _ = model.predict(validSet, verbose=0)
            end = time.time()
            inference_times.append(end - start)
        
        mean_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        
        print(f"Inference time: {mean_inference_time:.4f} ± {std_inference_time:.4f} seconds")
        print(f"  (average over {num_inference_runs} runs)")
        
        # Generate predictions for plotting
        yPredValid = model.predict(validSet, verbose=0)
        
        # Calculate additional metrics (linear space)
        epsilon = 1e-10
        relative_errors = (yPredValid - yValid) / (np.abs(yValid) + epsilon)
        mean_abs_rel_error = np.mean(np.abs(relative_errors))
        median_abs_rel_error = np.median(np.abs(relative_errors))
        max_abs_rel_error = np.max(np.abs(relative_errors))
        
        print(f"\nPrediction Metrics:")
        print(f"  Mean absolute relative error: {mean_abs_rel_error:.6f}")
        print(f"  Median absolute relative error: {median_abs_rel_error:.6f}")
        print(f"  Max absolute relative error: {max_abs_rel_error:.6f}")
        
        # Plot loss curves
        self.plot_loss_curves(train_loss, val_loss, loss_plot_path, model_name)
        
        # Plot predictions (for first output only to keep it manageable)
        self.plot_predictions(yValid[:, 0], yPredValid[:, 0], prediction_plot_path, 
                            model_name, output_idx=0)
        
        # Compile results
        results_dict = {
            'model_idx': model_idx,
            'model_name': model_name,
            'config': config,
            'total_params': int(total_params),
            'training_time_seconds': training_time,
            'inference_time_mean': mean_inference_time,
            'inference_time_std': std_inference_time,
            'best_val_loss': float(best_val_loss),
            'best_epoch': int(best_epoch),
            'final_train_loss': float(final_train_loss),
            'final_val_loss': float(final_val_loss),
            'mean_abs_rel_error': float(mean_abs_rel_error),
            'median_abs_rel_error': float(median_abs_rel_error),
            'max_abs_rel_error': float(max_abs_rel_error),
            'model_dir': model_dir,
        }
        
        # Save model
        model_save_path = os.path.join(model_dir, 'saved_model.keras')  # .keras extension required for Keras 3.x
        model.save(model_save_path)
        print(f"\nModel saved to: {model_save_path}")
        
        # Clear model from memory
        del model
        tf.keras.backend.clear_session()
        
        return results_dict
    
    def plot_loss_curves(self, train_loss, val_loss, save_path, model_name):
        """Plot and save training/validation loss curves"""
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        
        ax.plot(train_loss, label='Training Loss', linewidth=2)
        ax.plot(val_loss, label='Validation Loss', linewidth=2)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (log scale)', fontsize=12)
        ax.set_title(f'Loss Curves - {model_name}\nBest Val Loss: {min(val_loss):.6f}', 
                    fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def plot_predictions(self, y_true, y_pred, save_path, model_name, output_idx=0):
        """Plot predictions vs ground truth"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor('white')

        # Linear space - no conversion needed
        epsilon = 1e-10
        relative_error = (y_pred - y_true) / (np.abs(y_true) + epsilon)

        # Plot 1: Predictions vs Ground Truth
        ax = axes[0, 0]
        ax.plot(y_true, label='Ground Truth', alpha=0.7, linewidth=1)
        ax.plot(y_pred, label='Prediction', alpha=0.7, linewidth=1)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')
        ax.set_title(f'Predictions vs Ground Truth (Output {output_idx})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Scatter plot
        ax = axes[0, 1]
        ax.scatter(y_true, y_pred, alpha=0.3, s=1)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Prediction')
        ax.set_title('Prediction Scatter Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Relative Error
        ax = axes[1, 0]
        ax.plot(relative_error, linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Relative Error')
        ax.set_title('Relative Error Distribution')
        ax.set_ylim([-1, 1])
        ax.grid(True, alpha=0.3)

        # Plot 4: Error Histogram
        ax = axes[1, 1]
        ax.hist(relative_error, bins=100, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Relative Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Relative Error Histogram')
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'{model_name} - Output {output_idx}', fontsize=14, y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def run_search(self, data, hyperparameter_grid=None):
        """
        Run the hyperparameter search.
        
        Parameters:
        -----------
        data : tuple
            Preprocessed data
        hyperparameter_grid : list, optional
            List of hyperparameter configurations. If None, uses default grid.
        """
        if hyperparameter_grid is None:
            hyperparameter_grid = self.define_hyperparameter_grid()
        
        total_models = len(hyperparameter_grid)
        print(f"\n{'='*80}")
        print(f"Starting hyperparameter search")
        print(f"Total models to train: {total_models}")
        print(f"{'='*80}\n")
        
        search_start_time = time.time()
        
        for idx, config in enumerate(hyperparameter_grid):
            try:
                results = self.train_single_model(config, data, idx, total_models)
                self.results.append(results)
                
                # Save intermediate results
                self.save_results()
                
            except Exception as e:
                print(f"\n{'!'*80}")
                print(f"ERROR training model {idx + 1}: {str(e)}")
                print(f"{'!'*80}\n")
                continue
        
        search_end_time = time.time()
        total_search_time = search_end_time - search_start_time
        
        print(f"\n{'='*80}")
        print(f"Hyperparameter search completed!")
        print(f"Total time: {total_search_time:.2f} seconds ({total_search_time/3600:.2f} hours)")
        print(f"Successfully trained: {len(self.results)}/{total_models} models")
        print(f"{'='*80}\n")
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def save_results(self):
        """Save all results to JSON file"""
        results_path = os.path.join(self.search_dir, 'all_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to: {results_path}")
    
    def generate_summary(self):
        """Generate and save summary of all experiments"""
        if not self.results:
            print("No results to summarize!")
            return
        
        summary_path = os.path.join(self.search_dir, 'SUMMARY.txt')
        
        # Sort results by validation loss
        sorted_results = sorted(self.results, key=lambda x: x['best_val_loss'])
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HYPERPARAMETER SEARCH SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total models trained: {len(self.results)}\n")
            f.write(f"Search directory: {self.search_dir}\n\n")
            
            f.write("="*80 + "\n")
            f.write("TOP 10 MODELS (by validation loss)\n")
            f.write("="*80 + "\n\n")
            
            for i, result in enumerate(sorted_results[:10]):
                f.write(f"Rank {i+1}: {result['model_name']}\n")
                f.write(f"  Validation Loss: {result['best_val_loss']:.6f}\n")
                f.write(f"  Mean Abs Rel Error: {result['mean_abs_rel_error']:.6f}\n")
                f.write(f"  Training Time: {result['training_time_seconds']:.2f}s\n")
                f.write(f"  Inference Time: {result['inference_time_mean']:.4f}s\n")
                f.write(f"  Parameters: {result['total_params']:,}\n")
                f.write(f"  Config: Width={result['config']['model_width']}, "
                       f"Depth={result['config']['model_depth']}, "
                       f"LR={result['config']['learning_rate']:.0e}, "
                       f"BS={result['config']['batch_size_train']}\n")
                f.write(f"  Model Directory: {result['model_dir']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("BEST MODEL DETAILS\n")
            f.write("="*80 + "\n\n")
            
            best = sorted_results[0]
            f.write(f"Model Name: {best['model_name']}\n")
            f.write(f"Validation Loss: {best['best_val_loss']:.6f}\n")
            f.write(f"Train Loss: {best['final_train_loss']:.6f}\n")
            f.write(f"Mean Abs Rel Error: {best['mean_abs_rel_error']:.6f}\n")
            f.write(f"Median Abs Rel Error: {best['median_abs_rel_error']:.6f}\n")
            f.write(f"Training Time: {best['training_time_seconds']:.2f}s\n")
            f.write(f"Inference Time: {best['inference_time_mean']:.4f} ± {best['inference_time_std']:.4f}s\n")
            f.write(f"Total Parameters: {best['total_params']:,}\n")
            f.write(f"Best Epoch: {best['best_epoch'] + 1}\n\n")
            
            f.write("Configuration:\n")
            for key, value in best['config'].items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nModel Directory: {best['model_dir']}\n")
        
        print(f"\nSummary saved to: {summary_path}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("BEST MODEL")
        print("="*80)
        print(f"Model: {best['model_name']}")
        print(f"Validation Loss: {best['best_val_loss']:.6f}")
        print(f"Mean Abs Rel Error: {best['mean_abs_rel_error']:.6f}")
        print(f"Model Directory: {best['model_dir']}")
        print("="*80 + "\n")
        
        # Create comparison plots
        self.create_comparison_plots()
    
    def create_comparison_plots(self):
        """Create plots comparing all models"""
        if not self.results:
            return
        
        # Extract data for plotting
        model_names = [r['model_name'] for r in self.results]
        val_losses = [r['best_val_loss'] for r in self.results]
        train_times = [r['training_time_seconds'] for r in self.results]
        inference_times = [r['inference_time_mean'] for r in self.results]
        param_counts = [r['total_params'] for r in self.results]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('white')
        
        # Sort by validation loss for better visualization
        sorted_indices = np.argsort(val_losses)
        
        # Plot 1: Validation Loss
        ax = axes[0, 0]
        ax.bar(range(len(val_losses)), np.array(val_losses)[sorted_indices])
        ax.set_xlabel('Model Index (sorted by val loss)')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss Comparison')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Training Time vs Validation Loss
        ax = axes[0, 1]
        scatter = ax.scatter(train_times, val_losses, c=param_counts, 
                           s=100, alpha=0.6, cmap='viridis')
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Training Time vs Validation Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Parameter Count')
        
        # Plot 3: Inference Time vs Validation Loss
        ax = axes[1, 0]
        scatter = ax.scatter(inference_times, val_losses, c=param_counts,
                           s=100, alpha=0.6, cmap='viridis')
        ax.set_xlabel('Inference Time (seconds)')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Inference Time vs Validation Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Parameter Count')
        
        # Plot 4: Parameter Count vs Validation Loss
        ax = axes[1, 1]
        ax.scatter(param_counts, val_losses, s=100, alpha=0.6)
        ax.set_xlabel('Total Parameters')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Model Size vs Validation Loss')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Model Comparison Across All Experiments', fontsize=16, y=0.995)
        plt.tight_layout()
        
        comparison_plot_path = os.path.join(self.search_dir, 'model_comparison.png')
        plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Comparison plots saved to: {comparison_plot_path}")


def main():
    """
    Main function to run the hyperparameter search.
    """
    
    # ========================================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ========================================================================
    
    trainingSetPath = 'Data/Training_MultiOutput.csv'  # UPDATE THIS
    validSetPath = 'Data/Valid_MultiOutput.csv'        # UPDATE THIS
    
    # Data preprocessing parameters
    dtype = np.float32
    num_outputs = 4

    # ========================================================================
    # LOAD AND PREPROCESS DATA
    # ========================================================================

    print("Loading and preprocessing data...")
    data = preprocessData(
        trainingSetPath,
        validSetPath,
        testSetPath=None,
        dtype=dtype,
        num_outputs=num_outputs,
    )

    XTrain, yTrain, XValid, yValid = data
    print(f"Training set: {XTrain.shape[0]} samples, {XTrain.shape[1]} features")
    print(f"Validation set: {XValid.shape[0]} samples")
    print(f"Number of outputs: {yTrain.shape[1]}")
    
    # ========================================================================
    # DEFINE HYPERPARAMETER SEARCH
    # ========================================================================
    
    # Initialize search
    search = HyperparameterSearch(base_output_dir='hyperparameter_search')
    
    # Option 1: Use default grid (automatically generated)
    # hyperparameter_grid = search.define_hyperparameter_grid()
    
    # Option 2: Define custom hyperparameter grid (RECOMMENDED FOR TESTING)
    hyperparameter_grid = [
        # Small models for quick testing
        {'model_width': 256, 'model_depth': 4, 'learning_rate': 1e-3,
         'batch_size_train': 2**14, 'batch_size_valid': 2**16,
         'num_epochs': 500, 'weight_initializer': 'glorot_uniform',
         'activation': 'relu', 'loss_func': 'mse'},
        
        {'model_width': 512, 'model_depth': 6, 'learning_rate': 5e-4,
         'batch_size_train': 2**15, 'batch_size_valid': 2**16,
         'num_epochs': 500, 'weight_initializer': 'glorot_uniform',
         'activation': 'relu', 'loss_func': 'mse'},
        
        # Larger model
        {'model_width': 1024, 'model_depth': 6, 'learning_rate': 1e-4,
         'batch_size_train': 2**16, 'batch_size_valid': 2**16,
         'num_epochs': 1000, 'weight_initializer': 'glorot_uniform',
         'activation': 'relu', 'loss_func': 'mse'},
    ]
    
    # ========================================================================
    # RUN HYPERPARAMETER SEARCH
    # ========================================================================
    
    print(f"\nStarting hyperparameter search with {len(hyperparameter_grid)} configurations...")
    
    results = search.run_search(data, hyperparameter_grid)
    
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH COMPLETE!")
    print("="*80)
    print(f"Results directory: {search.search_dir}")
    print(f"Total models trained: {len(results)}")
    
    if results:
        best_model = min(results, key=lambda x: x['best_val_loss'])
        print(f"\nBest model: {best_model['model_name']}")
        print(f"Best validation loss: {best_model['best_val_loss']:.6f}")
        print(f"Location: {best_model['model_dir']}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
