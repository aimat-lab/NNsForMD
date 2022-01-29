import tensorflow.keras as ks
import tensorflow as tf

from kgcnn.literature.Schnet import make_model


class SchnetEnergy(ks.Model):
    """Subclassed tf.keras.model for NACs which outputs NACs from coordinates.

    The model is supposed to be saved and exported.
    """

    def __init__(self,
                 cutoff_radius=4,
                 model_module="schnet_e",
                 schnet_kwargs=None,
                 **kwargs):
        super(SchnetEnergy, self).__init__(**kwargs)
        self.schnet_kwargs = schnet_kwargs
        self.model_module = model_module

        self._schnet_model = make_model(**schnet_kwargs)
        self.predict([tf.ragged.constant([[0]]),
                    tf.ragged.constant([[[0.0, 0.0, 0.0]]], ragged_rank=1, inner_shape=(3,)),
                    tf.ragged.constant([[[0, 0]]], ragged_rank=1, inner_shape=(2,))
        ])

    def call(self, data, training=False, **kwargs):
        """Call the model output, forward pass.

        Args:
            data (list): Atoms, coordinates.
            training (bool, optional): Training Mode. Defaults to False.

        Returns:
            y_pred (tf.tensor): predicted Energy.
        """
        x = data
        out = self._schnet_model(x)
        return out

    def get_config(self):
        # conf = super(NACModel2, self).get_config()
        conf = {}
        conf.update({
            "model_module": self.model_module,
            "schnet_kwargs": self.schnet_kwargs
        })
        return conf
