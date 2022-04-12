import numpy as np
import tensorflow.keras as ks
import tensorflow as tf

from kgcnn.literature.Schnet import make_model
from kgcnn.utils.data import ragged_tensor_from_nested_numpy


class SchnetEnergy(ks.Model):
    """Subclassed SchNet which outputs energies from coordinates.

    The model is supposed to be saved and exported.
    """

    def __init__(self,
                 model_module="schnet_e",
                 schnet_kwargs=None,
                 **kwargs):
        super(SchnetEnergy, self).__init__(**kwargs)
        self.schnet_kwargs = schnet_kwargs
        self.model_module = model_module

        self._schnet_model = make_model(**schnet_kwargs)

        # Build the model with example data.
        self.predict([tf.ragged.constant([[0]]),
                    tf.ragged.constant([[[0.0, 0.0, 0.0]]], ragged_rank=1, inner_shape=(3,)),
                    tf.ragged.constant([[[0, 0]]], ragged_rank=1, inner_shape=(2,))
        ])

    def call(self, data, training=False, **kwargs):
        """Call the model output, forward pass.

        Args:
            data (list): Atoms, coordinates, indices.
            training (bool, optional): Training Mode. Defaults to False.

        Returns:
            y (tf.tensor): predicted Energy.
        """
        x = data
        out = self._schnet_model(x)
        return out

    def get_config(self):
        # conf = super(NACModel2, self).get_config()
        conf = {}
        conf.update({
            "model_module": self.model_module,
            "schnet_kwargs": self.schnet_kwargs,
        })
        return conf

    def predict_to_tensor_input(self, x):
        atoms = ragged_tensor_from_nested_numpy(x[0])
        coords = ragged_tensor_from_nested_numpy(x[1])
        edge_idx = ragged_tensor_from_nested_numpy(x[2])
        return [atoms, coords, edge_idx]

    def call_to_tensor_input(self, x):
        return self.predict_to_tensor_input(x)

    def call_to_numpy_output(self, y):
        if isinstance(y, np.ndarray):
            return y
        return y.numpy()