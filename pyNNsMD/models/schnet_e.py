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
                 predict_gradient: bool = False,
                 schnet_kwargs=None,
                 **kwargs):
        super(SchnetEnergy, self).__init__(**kwargs)
        self.schnet_kwargs = schnet_kwargs
        self.model_module = model_module
        self.predict_gradient = predict_gradient
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
        if not self.predict_gradient:
            out = self._schnet_model(x)
        else:
            geos = x[1]
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(geos.values)
                temp_e = self._schnet_model(x)
            temp_g = tape2.jacobian(temp_e, geos.values)
            temp_g = tf.transpose(temp_g, [0, 2, 1, 3])
            temp_g = tf.map_fn(lambda l_arg: tf.gather(l_arg[0], tf.range(l_arg[1], l_arg[1] + l_arg[2]), axis=0),
                               [temp_g, geos.row_starts(), geos.row_lengths()],
                               fn_output_signature=tf.RaggedTensorSpec(shape=[None, temp_e.shape[1], 3],
                                                                       ragged_rank=0,
                                                                       dtype=tf.float32))
            out = [temp_e, temp_g]

        return out

    def get_config(self):
        # conf = super(NACModel2, self).get_config()
        conf = {}
        conf.update({
            "model_module": self.model_module,
            "schnet_kwargs": self.schnet_kwargs,
            "predict_gradient": self.predict_gradient
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
        if not self.predict_gradient:
            out = y if isinstance(y, np.ndarray) else y.numpy()
        else:
            y0 = y[0] if isinstance(y[0], np.ndarray) else y[0].numpy()
            y1 = [g.numpy() for g in y[1]]
            out = [y0, y1]
        return out
