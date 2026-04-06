"""
Quick Test Script

Test that everything is set up correctly before running full hyperparameter search.
This script:
1. Verifies data files exist and are readable
2. Tests data preprocessing
3. Trains a small model quickly to verify the pipeline works
4. Tests inference speed
5. Tests moment loss integration

Run this before starting the full hyperparameter search!
"""

import numpy as np
import time
from pathlib import Path
import sys

try:
    from fcm_quadrature.training.data_loading import (
        preprocessData, buildDataset, trainModel
    )
    from fcm_quadrature.models.fnn import buildSequentialModel
    from tensorflow.keras.layers import Normalization
    from fcm_quadrature.training.losses import create_loss
    print("✓ Successfully imported all modules")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    sys.exit(1)


def test_data_loading(train_path, valid_path):
    """Test that data files can be loaded"""
    print("\n" + "="*80)
    print("TEST 1: Data Loading")
    print("="*80)

    # Check files exist
    if not Path(train_path).exists():
        print(f"✗ Training file not found: {train_path}")
        return None
    else:
        print(f"✓ Training file found: {train_path}")

    if not Path(valid_path).exists():
        print(f"✗ Validation file not found: {valid_path}")
        return None
    else:
        print(f"✓ Validation file found: {valid_path}")

    # Try loading data
    try:
        print("\nLoading and preprocessing data...")
        data = preprocessData(
            train_path,
            valid_path,
            testSetPath=None,
            dtype=np.float32,
            num_outputs=4,
        )

        XTrain, yTrain, XValid, yValid = data

        print(f"✓ Data loaded successfully")
        print(f"\nData Shape Information:")
        print(f"  Training samples: {XTrain.shape[0]:,}")
        print(f"  Validation samples: {XValid.shape[0]:,}")
        print(f"  Input features: {XTrain.shape[1]}")
        print(f"  Output targets: {yTrain.shape[1]}")

        # Check for NaN or Inf
        if np.any(np.isnan(XTrain)) or np.any(np.isnan(yTrain)):
            print("✗ Warning: NaN values detected in training data!")
            return None

        if np.any(np.isinf(XTrain)) or np.any(np.isinf(yTrain)):
            print("✗ Warning: Inf values detected in training data!")
            return None

        # Check validation data too
        if np.any(np.isnan(XValid)) or np.any(np.isnan(yValid)):
            print("✗ Warning: NaN values detected in validation data!")
            return None

        if np.any(np.isinf(XValid)) or np.any(np.isinf(yValid)):
            print("✗ Warning: Inf values detected in validation data!")
            return None

        print("✓ No NaN or Inf values detected")

        return data

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None


def test_model_building(data):
    """Test that model can be built"""
    print("\n" + "="*80)
    print("TEST 2: Model Building")
    print("="*80)

    try:
        XTrain = data[0]
        inputDim = XTrain.shape[1]

        print(f"\nBuilding test model (small configuration)...")
        print(f"  Input dimension: {inputDim}")
        print(f"  Hidden layers: 2")
        print(f"  Layer width: 64")
        print(f"  Output neurons: 4")

        # Create normalization layer
        inputNormLayer = Normalization()
        inputNormLayer.adapt(XTrain)

        # Build small model (no dropout)
        model = buildSequentialModel(
            inputDim=inputDim,
            dtype=np.float32,
            layerSizeList=[64, 64],
            activationList=['relu', 'relu'],
            weightInitializer='glorot_uniform',
            lossFunc='mse',
            modelName='test_model',
            num_outputs=4,
            learningRate=1e-3,
            inputNormLayer=inputNormLayer,
            verbose=0,
        )

        total_params = model.count_params()
        print(f"✓ Model built successfully (no dropout)")
        print(f"  Total parameters: {total_params:,}")

        # Build model with dropout
        model_dropout = buildSequentialModel(
            inputDim=inputDim,
            dtype=np.float32,
            layerSizeList=[64, 64],
            activationList=['relu', 'relu'],
            weightInitializer='glorot_uniform',
            lossFunc='mse',
            modelName='test_model_dropout',
            num_outputs=4,
            learningRate=1e-3,
            inputNormLayer=inputNormLayer,
            verbose=0,
            dropout_rate=0.1,
        )
        print(f"✓ Model built successfully (dropout_rate=0.1)")
        print(f"  Total parameters: {model_dropout.count_params():,}")

        return model

    except Exception as e:
        print(f"✗ Error building model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_training(data, model):
    """Test that model can be trained"""
    print("\n" + "="*80)
    print("TEST 3: Model Training")
    print("="*80)

    try:
        print("\nPreparing datasets...")
        trainSet, validSet, inputDim, stepPerEpoch = buildDataset(
            data,
            batchSizeTrain=2**12,
            batchSizeValid=2**14,
        )

        print(f"✓ Datasets created")
        print(f"  Steps per epoch: {stepPerEpoch}")

        print("\nTraining for 10 epochs (this may take a minute)...")
        start_time = time.time()

        history = trainModel(
            model,
            trainSet,
            validSet,
            numEpochs=10,
            verbose=0,
            callbackVerbose=0,
        )

        training_time = time.time() - start_time

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        print(f"✓ Training completed")
        print(f"\nTraining Results:")
        print(f"  Time for 10 epochs: {training_time:.2f} seconds")
        print(f"  Time per epoch: {training_time/10:.2f} seconds")
        print(f"  Initial train loss: {train_loss[0]:.6f}")
        print(f"  Final train loss: {train_loss[-1]:.6f}")
        print(f"  Initial val loss: {val_loss[0]:.6f}")
        print(f"  Final val loss: {val_loss[-1]:.6f}")

        # Check if loss decreased
        if train_loss[-1] < train_loss[0]:
            print("✓ Training loss decreased (model is learning!)")
        else:
            print("✗ Warning: Training loss did not decrease")

        return training_time

    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_inference(data, model):
    """Test inference speed"""
    print("\n" + "="*80)
    print("TEST 4: Inference Speed")
    print("="*80)

    try:
        _, validSet, _, _ = buildDataset(data, 2**12, 2**14)

        print("\nMeasuring inference time (5 runs)...")
        inference_times = []

        for i in range(5):
            start = time.time()
            predictions = model.predict(validSet, verbose=0)
            end = time.time()
            inference_times.append(end - start)
            print(f"  Run {i+1}: {inference_times[-1]:.4f} seconds")

        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)

        print(f"\n✓ Inference test completed")
        print(f"  Mean time: {mean_time:.4f} ± {std_time:.4f} seconds")
        print(f"  Prediction shape: {predictions.shape}")

        return mean_time

    except Exception as e:
        print(f"✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_moment_loss(data):
    """Test that moment loss works in the training pipeline"""
    print("\n" + "="*80)
    print("TEST 5: Moment Loss")
    print("="*80)

    try:
        XTrain = data[0]
        inputDim = XTrain.shape[1]

        inputNormLayer = Normalization()
        inputNormLayer.adapt(XTrain)

        for loss_type in ['moment', 'combined']:
            loss_fn = create_loss(loss_type)
            print(f"\n  Testing '{loss_type}' loss...")

            model = buildSequentialModel(
                inputDim=inputDim,
                dtype=np.float32,
                layerSizeList=[32, 32],
                activationList=['relu', 'relu'],
                weightInitializer='glorot_uniform',
                lossFunc=loss_fn,
                modelName=f'test_{loss_type}',
                num_outputs=4,
                learningRate=1e-3,
                inputNormLayer=inputNormLayer,
                verbose=0,
            )

            trainSet, validSet, _, _ = buildDataset(
                data, batchSizeTrain=2**12, batchSizeValid=2**14
            )

            history = trainModel(
                model, trainSet, validSet,
                numEpochs=3, verbose=0, callbackVerbose=0,
            )

            train_loss = history.history['loss']
            print(f"  ✓ '{loss_type}' loss works (loss: {train_loss[0]:.6f} → {train_loss[-1]:.6f})")

        print("\n✓ All moment loss types work correctly")
        return True

    except Exception as e:
        print(f"✗ Error testing moment loss: {e}")
        import traceback
        traceback.print_exc()
        return None


def estimate_search_time(training_time_per_epoch, num_configs, epochs_per_model):
    """Estimate total time for hyperparameter search"""
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH TIME ESTIMATE")
    print("="*80)

    estimated_time_per_model = training_time_per_epoch * epochs_per_model
    total_time = estimated_time_per_model * num_configs

    print(f"\nEstimations (based on test model, actual models will be larger):")
    print(f"  Test model time per epoch: {training_time_per_epoch:.2f}s")
    print(f"  Number of configurations: {num_configs}")
    print(f"  Epochs per model: {epochs_per_model}")
    print(f"\nMinimum estimated time: {total_time/3600:.1f} hours")
    print(f"  ({total_time/60:.0f} minutes)")
    print(f"  Note: Actual time will be higher with larger models")

    if total_time > 86400:  # More than 24 hours
        print(f"\n⚠ Warning: Estimated time is over 24 hours!")
        print("  Consider reducing:")
        print("    - Number of configurations")
        print("    - Epochs per model")
        print("    - Model sizes")


def main():
    """Main test function"""

    print("\n" + "="*80)
    print("NEURAL NETWORK PIPELINE TEST")
    print("="*80)
    print("\nThis script will verify that everything is set up correctly")
    print("before running the full hyperparameter search.\n")

    # Configuration
    train_path = 'Data/Training_1M_NoVertices.csv'
    valid_path = 'Data/Valid_1M_NoVertices.csv'

    print(f"Configuration:")
    print(f"  Training data: {train_path}")
    print(f"  Validation data: {valid_path}")

    # Test 1: Data loading
    data = test_data_loading(train_path, valid_path)
    if data is None:
        print("\n✗ Data loading test failed. Please fix the issues above.")
        return

    # Test 2: Model building
    model = test_model_building(data)
    if model is None:
        print("\n✗ Model building test failed. Please fix the issues above.")
        return

    # Test 3: Training
    training_time = test_training(data, model)
    if training_time is None:
        print("\n✗ Training test failed. Please fix the issues above.")
        return
    training_time_per_epoch = training_time / 10

    # Test 4: Inference
    inference_time = test_inference(data, model)
    if inference_time is None:
        print("\n✗ Inference test failed. Please fix the issues above.")
        return

    # Test 5: Moment loss
    moment_result = test_moment_loss(data)
    if moment_result is None:
        print("\n✗ Moment loss test failed. Please fix the issues above.")
        return

    # All tests passed
    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✓")
    print("="*80)

    # Estimate hyperparameter search time
    estimate_search_time(
        training_time_per_epoch,
        num_configs=30,
        epochs_per_model=500
    )

    print("\n" + "="*80)
    print("READY FOR TRAINING!")
    print("="*80)
    print("\nYou can now run:")
    print("  python train.py --template > config.json    # generate config")
    print("  python train.py config.json                  # train 1 model")
    print("  python train.py configs.json --num-gpus 8    # train many on GPUs")
    print("  python train.py configs.json --cpu-only      # train many on CPU")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()