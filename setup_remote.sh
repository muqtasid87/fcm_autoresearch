#!/bin/bash
# One-time setup script for the remote GPU machine.
# Run this after cloning the repo on the remote.
set -euo pipefail

echo "Setting up fcm-quadrature on remote machine..."

# Install package in editable mode
pip install -e .

# Install MLflow for experiment tracking (optional)
pip install mlflow || echo "Warning: MLflow install failed (optional dependency)"

echo "Setup complete. You can now run training scripts."
echo "Example: python scripts/train.py configs/examples/fnn_single.json"
