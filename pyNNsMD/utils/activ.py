import tensorflow as tf
import tensorflow.keras as ks


@tf.keras.utils.register_keras_serializable(package='pyNNsMD', name='leaky_softplus')
class leaky_softplus(tf.keras.layers.Layer):
    r"""Leaky soft-plus activation function similar to :obj:`tf.nn.leaky_relu` but smooth. """

    def __init__(self, alpha=0.05, **kwargs):
        """Initialize with optionally learnable parameter.

        Args:
            alpha (float, optional): Leak parameter alpha. Default is 0.05.
        """
        super(leaky_softplus, self).__init__(**kwargs)
        self.alpha = float(alpha)

    def call(self, inputs, **kwargs):
        """Compute leaky_softplus activation from inputs."""
        x = inputs
        return ks.activations.softplus(x) * (1 - self.alpha) + self.alpha * x

    def get_config(self):
        config = super(leaky_softplus, self).get_config()
        config.update({"alpha": self.alpha})
        return config


@tf.keras.utils.register_keras_serializable(package='pyNNsMD', name='shifted_softplus')
def shifted_softplus(x):
    """Soft-plus function from tf.keras shifted downwards.

    Args:
        x (tf.tensor): Activation input.

    Returns:
        tf.tensor: Activation.

    """
    return ks.activations.softplus(x) - ks.backend.log(2.0)
