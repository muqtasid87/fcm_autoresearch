"""
Moment Fitting Loss Functions for FCM Quadrature Weight Prediction

Custom Keras loss functions that compare predicted and true quadrature weights
based on their computed moments rather than direct weight values.

Mathematical Background:
- 4 Gauss points at +/- 1/sqrt(3) in 2D
- Basis functions: {1, x, y, xy}
- Moment equations: m = weights @ V where V is the Vandermonde matrix

Usage:
    from fcm_quadrature.training.losses import create_loss

    # MSE only (current behavior)
    loss_fn = create_loss('mse')

    # Moment loss only
    loss_fn = create_loss('moment')

    # Combined loss (recommended)
    loss_fn = create_loss('combined')  # Uses defaults: alpha=1.0, beta=0.5
    loss_fn = create_loss('combined', alpha=1.0, beta=1.0)  # Custom values
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


# Handle both TensorFlow 2.x and Keras 3.x
try:
    keras_serializable = tf.keras.saving.register_keras_serializable
except AttributeError:
    try:
        from keras.saving import register_keras_serializable
        keras_serializable = register_keras_serializable
    except:
        def keras_serializable(*args, **kwargs):
            def decorator(cls):
                return cls
            return decorator


# =============================================================================
# Constants: Precomputed Vandermonde Matrix
# =============================================================================

# Standard 2D Gauss quadrature point: 1/sqrt(3)
_XI = 1.0 / np.sqrt(3.0)
_XI_SQ = 1.0 / 3.0

# Vandermonde matrix V[i,j] = phi_j(point_i)
# Points: [(-xi,-xi), (xi,-xi), (-xi,xi), (xi,xi)]
# Basis:  [1, x, y, xy]
_VANDERMONDE_NP = np.array([
    [1.0, -_XI, -_XI,  _XI_SQ],  # Point 0: (-xi, -xi)
    [1.0,  _XI, -_XI, -_XI_SQ],  # Point 1: ( xi, -xi)
    [1.0, -_XI,  _XI, -_XI_SQ],  # Point 2: (-xi,  xi)
    [1.0,  _XI,  _XI,  _XI_SQ],  # Point 3: ( xi,  xi)
], dtype=np.float32)


# =============================================================================
# Core Moment Computation Function
# =============================================================================

def compute_moments(weights):
    """
    Compute moments from quadrature weights.

    The moments are computed as: m = weights @ V
    where V is the Vandermonde matrix with V[i,j] = phi_j(point_i)

    Args:
        weights: Tensor of shape (batch_size, 4) containing quadrature weights
                 [w0, w1, w2, w3] for points [(-xi,-xi), (xi,-xi), (-xi,xi), (xi,xi)]

    Returns:
        moments: Tensor of shape (batch_size, 4) containing [m_0, m_1, m_2, m_3]
                 m_0 = sum of weights (integrates basis 1)
                 m_1 = weighted sum for x basis
                 m_2 = weighted sum for y basis
                 m_3 = weighted sum for xy basis
    """
    V = tf.constant(_VANDERMONDE_NP, dtype=tf.float32)
    # moments[b,:] = weights[b,:] @ V
    # Shape: (B, 4) @ (4, 4) = (B, 4)
    moments = tf.matmul(weights, V)
    return moments


# =============================================================================
# Keras Loss Classes
# =============================================================================

@keras_serializable(package="MomentLoss")
class MomentSquaredError(keras.losses.Loss):
    """
    Custom Keras loss that computes MSE in moment space.

    Instead of comparing weights directly, this loss compares the moments
    (integrated basis functions) that the weights would produce.

    This is useful because different weight combinations can produce the same
    integration results - what matters is that the moments match.

    Args:
        reduction: Type of reduction to apply to loss
        name: Name for the loss instance
    """

    def __init__(
        self,
        reduction='sum_over_batch_size',
        name='moment_squared_error',
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)

    def call(self, y_true, y_pred):
        """
        Compute the moment-based loss.

        Args:
            y_true: True quadrature weights, shape (batch_size, 4)
            y_pred: Predicted quadrature weights, shape (batch_size, 4)

        Returns:
            Per-sample loss values, shape (batch_size,)
        """
        # Ensure float32 for numerical stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        moments_true = compute_moments(y_true)
        moments_pred = compute_moments(y_pred)

        moment_errors = moments_pred - moments_true

        # Return per-sample loss (reduction handled by parent class)
        squared_errors = tf.square(moment_errors)
        return tf.reduce_sum(squared_errors, axis=-1)

    def get_config(self):
        config = super().get_config()
        return config


@keras_serializable(package="MomentLoss")
class CombinedWeightMomentLoss(keras.losses.Loss):
    """
    Combined loss: alpha * MSE(weights) + beta * MSE(moments)

    This loss provides a balance between directly matching weights and
    ensuring the physical moment-fitting properties are satisfied.

    The moment loss ensures that even if individual weights differ slightly,
    the overall integration (moment computation) remains accurate.

    Args:
        alpha: Weight for direct MSE loss on weights (default: 1.0)
        beta: Weight for MSE loss in moment space (default: 0.5)
        reduction: Type of reduction to apply
        name: Name for the loss instance

    Example:
        >>> loss_fn = CombinedWeightMomentLoss(alpha=1.0, beta=0.5)
        >>> model.compile(optimizer='adam', loss=loss_fn)
    """

    def __init__(
        self,
        alpha=1.0,
        beta=0.5,
        reduction='sum_over_batch_size',
        name='combined_weight_moment_loss',
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        """
        Compute the combined loss.

        Args:
            y_true: True quadrature weights, shape (batch_size, 4)
            y_pred: Predicted quadrature weights, shape (batch_size, 4)

        Returns:
            Per-sample combined loss values, shape (batch_size,)
        """
        # Ensure float32 for numerical stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Direct MSE on weights (per sample)
        weight_mse = tf.reduce_sum(tf.square(y_pred - y_true), axis=-1)

        # Moment MSE (per sample)
        moments_true = compute_moments(y_true)
        moments_pred = compute_moments(y_pred)
        moment_errors = moments_pred - moments_true
        moment_mse = tf.reduce_sum(tf.square(moment_errors), axis=-1)

        # Combined loss (per sample)
        return self.alpha * weight_mse + self.beta * moment_mse

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta,
        })
        return config


# =============================================================================
# Factory Function for Easy Switching
# =============================================================================

def create_loss(
    loss_type='mse',
    alpha=1.0,
    beta=0.5
):
    """
    Factory function to create loss instances.

    This provides an easy way to switch between different loss functions
    without changing the rest of the training code.

    Args:
        loss_type: One of 'mse', 'moment', 'combined'
            - 'mse': Standard mean squared error on weights (built-in, fastest)
            - 'moment': MSE computed in moment space only
            - 'combined': alpha * MSE(weights) + beta * MSE(moments)
        alpha: Weight for MSE loss (used in 'combined', default: 1.0)
        beta: Weight for moment loss (used in 'combined', default: 0.5)

    Returns:
        Keras Loss instance or string for built-in losses

    Examples:
        >>> # Standard MSE (current behavior)
        >>> loss_fn = create_loss('mse')

        >>> # Moment-only loss
        >>> loss_fn = create_loss('moment')

        >>> # Combined with default weights
        >>> loss_fn = create_loss('combined')

        >>> # Combined with custom weights
        >>> loss_fn = create_loss('combined', alpha=1.0, beta=1.0)
    """
    if loss_type == 'mse':
        return 'mse'  # Use built-in for efficiency

    elif loss_type == 'moment':
        return MomentSquaredError()

    elif loss_type == 'combined':
        return CombinedWeightMomentLoss(alpha=alpha, beta=beta)

    else:
        raise ValueError(
            f"Unknown loss_type: '{loss_type}'. "
            f"Valid options are: 'mse', 'moment', 'combined'"
        )


# =============================================================================
# Utility Functions
# =============================================================================

def verify_moment_loss_gradient():
    """
    Verify that moment loss has well-defined gradients.

    This is useful for debugging and ensuring the loss function
    is properly differentiable for backpropagation.

    Returns:
        bool: True if gradients exist and are valid
    """
    # Create test data
    y_true = tf.constant([[0.5, 0.3, 0.2, 0.4]], dtype=tf.float32)
    y_pred = tf.Variable([[0.45, 0.35, 0.25, 0.35]], dtype=tf.float32)

    # Test MomentSquaredError
    loss_fn = MomentSquaredError()
    with tf.GradientTape() as tape:
        loss = loss_fn(y_true, y_pred)
    gradients = tape.gradient(loss, y_pred)

    print("Moment Loss Gradient Verification:")
    print(f"  y_true: {y_true.numpy()}")
    print(f"  y_pred: {y_pred.numpy()}")
    print(f"  loss: {loss.numpy():.6f}")
    print(f"  gradients: {gradients.numpy()}")
    print(f"  gradients exist: {gradients is not None}")

    # Test CombinedWeightMomentLoss
    combined_loss_fn = CombinedWeightMomentLoss(alpha=1.0, beta=0.5)
    with tf.GradientTape() as tape:
        combined_loss = combined_loss_fn(y_true, y_pred)
    combined_gradients = tape.gradient(combined_loss, y_pred)

    print("\nCombined Loss Gradient Verification:")
    print(f"  combined loss: {combined_loss.numpy():.6f}")
    print(f"  gradients: {combined_gradients.numpy()}")
    print(f"  gradients exist: {combined_gradients is not None}")

    return gradients is not None and combined_gradients is not None


def print_vandermonde_info():
    """Print information about the Vandermonde matrix for debugging."""
    print("Vandermonde Matrix Information:")
    print(f"  xi = 1/sqrt(3) = {_XI:.10f}")
    print(f"  xi^2 = 1/3 = {_XI_SQ:.10f}")
    print("\nVandermonde Matrix V[point, basis]:")
    print("  Points: [(-xi,-xi), (xi,-xi), (-xi,xi), (xi,xi)]")
    print("  Basis:  [1, x, y, xy]")
    print(_VANDERMONDE_NP)
    print("\nFor weights = [1, 1, 1, 1] (full cell):")
    test_weights = np.array([[1.0, 1.0, 1.0, 1.0]])
    moments = compute_moments(tf.constant(test_weights, dtype=tf.float32))
    print(f"  Moments: {moments.numpy()[0]}")
    print(f"  Expected: [4, 0, 0, 0] (area=4, symmetric moments=0)")


# =============================================================================
# Main (Self-Test)
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Moment Loss Module - Self Test")
    print("=" * 60)

    print_vandermonde_info()
    print("\n" + "=" * 60)
    verify_moment_loss_gradient()

    print("\n" + "=" * 60)
    print("Factory Function Test:")
    print("=" * 60)

    for loss_type in ['mse', 'moment', 'combined']:
        loss_fn = create_loss(loss_type)
        print(f"  create_loss('{loss_type}'): {loss_fn}")
