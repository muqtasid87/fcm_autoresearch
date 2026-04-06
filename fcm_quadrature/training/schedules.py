"""
Learning rate schedules for transformer training.

Classes:
    WarmupSchedule - Linear warmup followed by another schedule (e.g. CosineDecay).
"""

import tensorflow as tf
from tensorflow import keras


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


@keras_serializable(package="TransformerSchedules")
class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with linear warmup followed by another schedule.

    During warmup (first warmup_steps), LR increases linearly from 0 to base_lr.
    After warmup, follows the provided schedule (e.g., CosineDecay).

    Args:
        base_lr: Base learning rate (reached at end of warmup)
        warmup_steps: Number of warmup steps
        schedule: Learning rate schedule to use after warmup
    """

    def __init__(self, base_lr, warmup_steps, schedule):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.schedule = schedule

    def __call__(self, step):
        """Compute learning rate for given step."""
        warmup_lr = self.base_lr * (tf.cast(step, tf.float32) /
                                    tf.cast(self.warmup_steps, tf.float32))
        scheduled_lr = self.schedule(step - self.warmup_steps)
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: scheduled_lr
        )

    def get_config(self):
        """Return configuration for serialization."""
        try:
            from keras.saving import serialize_keras_object
        except ImportError:
            from tensorflow.keras.saving import serialize_keras_object

        return {
            'base_lr': self.base_lr,
            'warmup_steps': self.warmup_steps,
            'schedule': serialize_keras_object(self.schedule),
        }

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration (required for deserialization)."""
        try:
            from keras.saving import deserialize_keras_object
        except ImportError:
            from tensorflow.keras.saving import deserialize_keras_object

        schedule = deserialize_keras_object(config['schedule'])

        return cls(
            base_lr=config['base_lr'],
            warmup_steps=config['warmup_steps'],
            schedule=schedule
        )
