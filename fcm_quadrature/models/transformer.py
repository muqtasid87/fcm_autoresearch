"""
FT-Transformer model components for tabular regression.

Classes:
    FeatureTokenizer - Converts each numerical feature into a d_model embedding.
    CLSToken - Learnable CLS token prepended to the sequence.
    TransformerEncoderBlock - Standard Transformer encoder block.
    FTTransformer - Full Feature Tokenizer Transformer for tabular regression.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
# Transformer Components
# =============================================================================

class FeatureTokenizer(layers.Layer):
    """
    Converts each numerical feature into a d_model dimensional embedding.

    For tabular data, each feature becomes a "token" that can attend to other features.
    This is the key insight from FT-Transformer paper.
    """

    def __init__(self, num_features, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.d_model = d_model

    def build(self, input_shape):
        # Each feature gets its own embedding weights
        # Shape: (num_features, d_model) - one embedding vector per feature
        self.feature_embeddings = self.add_weight(
            name='feature_embeddings',
            shape=(self.num_features, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        # Bias per feature
        self.feature_biases = self.add_weight(
            name='feature_biases',
            shape=(self.num_features, self.d_model),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, num_features)
        # Expand to (batch, num_features, 1)
        x = tf.expand_dims(inputs, axis=-1)
        # Multiply each feature by its embedding: (batch, num_features, d_model)
        # This creates a unique embedding direction for each feature
        tokens = x * self.feature_embeddings + self.feature_biases
        return tokens

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_features': self.num_features,
            'd_model': self.d_model
        })
        return config


class CLSToken(layers.Layer):
    """
    Learnable CLS token prepended to the sequence.
    Used for aggregating information from all features.
    """

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Broadcast CLS token to batch: (batch, 1, d_model)
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.d_model])
        # Prepend to sequence: (batch, 1 + num_features, d_model)
        return tf.concat([cls_tokens, inputs], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config


class TransformerEncoderBlock(layers.Layer):
    """
    Standard Transformer encoder block with:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization (pre-norm for stability)
    - Residual connections
    - Dropout for regularization
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Multi-head attention
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )

        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(d_ff, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),
            layers.Dropout(dropout_rate)
        ])

        # Layer normalization (pre-norm style)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)

        # Dropout for attention residual
        self.dropout1 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # Pre-norm multi-head attention with residual
        x_norm = self.ln1(inputs)
        attn_output = self.mha(x_norm, x_norm, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = inputs + attn_output

        # Pre-norm FFN with residual
        x_norm = self.ln2(x)
        ffn_output = self.ffn(x_norm, training=training)
        x = x + ffn_output

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate
        })
        return config


class FTTransformer(keras.Model):
    """
    Feature Tokenizer Transformer for Tabular Regression.

    Architecture:
    1. Input normalization (optional)
    2. Feature tokenization (each feature -> d_model embedding)
    3. CLS token prepended
    4. N transformer encoder blocks
    5. Extract CLS token output
    6. Regression head -> num_outputs

    Parameters:
    -----------
    num_features : int
        Number of input features (12 for this dataset)
    num_outputs : int
        Number of regression outputs (4 quadrature weights)
    d_model : int
        Embedding dimension
    num_heads : int
        Number of attention heads
    num_layers : int
        Number of transformer encoder blocks
    d_ff : int
        Feed-forward hidden dimension (typically 4 * d_model)
    dropout_rate : float
        Dropout rate for regularization
    """

    def __init__(
        self,
        num_features=12,
        num_outputs=4,
        d_model=64,
        num_heads=4,
        num_layers=3,
        d_ff=None,
        dropout_rate=0.1,
        use_normalization=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_features = num_features
        self.num_outputs = num_outputs
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.dropout_rate = dropout_rate
        self.use_normalization = use_normalization

        # Input normalization (optional)
        if use_normalization:
            self.input_norm = layers.Normalization()
        else:
            self.input_norm = None

        # Feature tokenizer
        self.feature_tokenizer = FeatureTokenizer(num_features, d_model)

        # CLS token
        self.cls_token = CLSToken(d_model)

        # Transformer encoder blocks
        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, num_heads, self.d_ff, dropout_rate)
            for _ in range(num_layers)
        ]

        # Final layer norm
        self.final_ln = layers.LayerNormalization(epsilon=1e-6)

        # Regression head
        self.regression_head = keras.Sequential([
            layers.Dense(d_model, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_outputs)
        ])

    def adapt_normalization(self, data):
        """Adapt the input normalization layer to training data."""
        if self.input_norm is not None:
            self.input_norm.adapt(data)

    def call(self, inputs, training=False):
        # Normalize inputs (if enabled)
        if self.input_norm is not None:
            x = self.input_norm(inputs)
        else:
            x = inputs

        # Tokenize features: (batch, num_features, d_model)
        x = self.feature_tokenizer(x)

        # Add CLS token: (batch, 1 + num_features, d_model)
        x = self.cls_token(x)

        # Pass through transformer blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, training=training)

        # Final layer norm
        x = self.final_ln(x)

        # Extract CLS token output: (batch, d_model)
        cls_output = x[:, 0, :]

        # Regression head: (batch, num_outputs)
        outputs = self.regression_head(cls_output, training=training)

        return outputs

    def get_config(self):
        return {
            'num_features': self.num_features,
            'num_outputs': self.num_outputs,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'use_normalization': self.use_normalization
        }


# =============================================================================
# Custom Objects (for model loading)
# =============================================================================

CUSTOM_OBJECTS = {
    'FTTransformer': FTTransformer,
    'FeatureTokenizer': FeatureTokenizer,
    'CLSToken': CLSToken,
    'TransformerEncoderBlock': TransformerEncoderBlock,
}
