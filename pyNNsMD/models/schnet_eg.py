import tensorflow as tf
from kgcnn.layers.modules import OptionalInputEmbedding
from pyNNsMD.layers.schnet import SchNetInteraction, NodeDistance, DenseMasked, ApplyMask, PoolingNodes
from kgcnn.layers.geom import GaussBasisLayer
from kgcnn.utils.adj import define_adjacency_from_distance, coordinates_to_distancematrix
from kgcnn.layers.mlp import MLP
import numpy as np

ks = tf.keras
# test_n = tf.constant([[0, 1], [1, 2]])
# test_x = tf.constant([[[0.0,0.0,0.0], [1.0,1.0,1.0]], [[2.0,2.0,2.0],[3.0,3.0,3.0]]])
# test_edi = tf.constant([[[1, 1], [0, 1]],
#                    [[0, 1], [0, 0]]])
# test_mask_e = tf.constant([[[True], [True]], [[True], [False]]])
# test_mask_n = tf.constant([[[True], [True]], [[True], [False]]])


class SchNetEnergy(ks.Model):
    """Subclassed SchNet which outputs energies from coordinates.

    The model is supposed to be saved and exported.
    """

    def __init__(self,
                 name="SchNetEnergy",
                 model_module="schnet_eg",
                 energy_only: bool = True,
                 output_as_dict: bool = True,
                 inputs: list = None,
                 input_embedding: dict = None,
                 gauss_args: dict = None,
                 interaction_args: dict = None,
                 node_pooling_args: dict = None,
                 depth: int = None,
                 verbose: int = None,
                 last_mlp: dict = None,
                 output_embedding: str = None,
                 use_output_mlp: bool = None,
                 output_mlp: dict = None,
                 max_neighbours: int = None,
                 **kwargs):
        super(SchNetEnergy, self).__init__(**kwargs)
        local_input = locals()
        kwargs_list = ["name", "model_module", "energy_only", "output_as_dict", "inputs", "input_embedding",
                       "gauss_args", "interaction_args", "node_pooling_args", "depth", "verbose", "last_mlp",
                       "output_embedding", "use_output_mlp", "output_mlp", "max_neighbours"]
        self._model_kwargs = {x: local_input[x] for x in kwargs_list}
        self.depth = depth
        self.energy_only = energy_only
        self.output_as_dict = output_as_dict
        self.max_neighbours = max_neighbours
        self.range_dist = gauss_args["distance"]
        # layers
        self.lay_embed = OptionalInputEmbedding(**input_embedding['node'], use_embedding=len(inputs[1]['shape']) < 2)
        self.lay_dist = NodeDistance()
        self.lay_gauss = GaussBasisLayer(**gauss_args)
        self.lay_linear = DenseMasked(interaction_args["units"], activation='linear')
        self.lay_int = [SchNetInteraction(**interaction_args) for _ in range(0, depth)]
        self.lay_pool = PoolingNodes(**node_pooling_args)
        self.lay_mask = ApplyMask()
        self.lay_mlp_last = MLP(**last_mlp)
        self.lay_mlp_output = MLP(**output_mlp)

        input_shape = [tf.TensorShape([None] + list(x["shape"])) for x in inputs]
        self.build(input_shape)
        self.compute_output_shape(input_shape)

    @tf.function
    def call_energy(self, inputs, **kwargs):
        # Make input
        x, n, edi, mask_n, mask_e = inputs
        edi = tf.cast(edi, dtype="int64")

        # embedding, if no feature dimension
        n = self.lay_embed(n)
        n = self.lay_mask(n, mask=mask_n)
        ed = self.lay_dist([x, edi], mask=[mask_n, mask_e])
        ed = self.lay_gauss(ed)
        ed = self.lay_mask(ed, mask=mask_e)
        n = self.lay_linear(n, mask=mask_n)
        for i in range(0, self.depth):
            n = self.lay_int[i]([n, ed, edi], mask=[mask_n, mask_e, mask_e])

        n = self.lay_mlp_last(n)
        n = self.lay_mask(n, mask=mask_n)
        out = self.lay_pool(n, mask=mask_n)
        out = self.lay_mlp_output(out)
        return out

    def call(self, data, training=False, **kwargs):
        """Call the model output, forward pass.

        Args:
            data (list): Atoms, coordinates, indices.
            training (bool, optional): Training Mode. Defaults to False.

        Returns:
            y (tf.tensor): predicted Energy.
        """
        if self.energy_only:
            out = self.call_energy(data, training=training, **kwargs)
            if self.output_as_dict:
                out = {'energy': out}
        else:
            x, n, edi, mask_n, mask_e = data
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                temp_e = self.call_energy([x, n, edi, mask_n, mask_e], training=training, **kwargs)
            temp_g = tape2.batch_jacobian(temp_e, x)
            if self.output_as_dict:
                out = {'energy': temp_e, 'force': temp_g}
            else:
                out = [temp_e, temp_g]
        return out

    def get_config(self):
        conf = {}
        conf.update(self._model_kwargs)
        return conf

    def predict_to_tensor_input(self, inputs):
        atoms, coords = inputs
        dist_mat = [coordinates_to_distancematrix(x) for x in coords]
        index_mat = [np.argsort(x, axis=1) for x in dist_mat]
        dist_okay = [np.take_along_axis(x, i, axis=1) < self.range_dist for x, i in zip(dist_mat, index_mat)]
        index_mat = [x[:, 1:self.max_neighbours + 1] for x in index_mat]
        dist_okay = [x[:, 1:self.max_neighbours + 1] for x in dist_okay]
        n, mask_n = self.padd_batch_dim(atoms)
        pos, _ = self.padd_batch_dim(coords)
        edi, _ = self.padd_batch_dim(index_mat)
        mask_e, _ = self.padd_batch_dim(dist_okay)
        X = [pos, n, edi, np.expand_dims(mask_n, axis=-1), np.expand_dims(mask_e, axis=-1)]
        return X

    def call_to_tensor_input(self, inputs):
        return self.predict_to_tensor_input(inputs)

    def call_to_numpy_output(self, y):
        if self.energy_only:
            if self.output_as_dict:
                out = {'energy': y["energy"].numpy()}
            else:
                out = [y.numpy()]
        else:
            if self.output_as_dict:
                out = {'energy': y['energy'].numpy(), 'force': y['force'].numpy()}
            else:
                out = [y[0].numpy(), y[1].numpy()]
        return out

    @staticmethod
    def padd_batch_dim(values):
        max_shape = np.amax([x.shape for x in values], axis=0)
        final_shape = np.concatenate([np.array([len(values)]), max_shape])
        padded = np.zeros(final_shape, dtype=values[0].dtype)
        mask = np.zeros(final_shape, dtype="bool")
        for i, x in enumerate(values):
            index = [i] + [slice(0, int(j)) for j in x.shape]
            padded[tuple(index)] = x
            mask[tuple(index)] = True
        return padded, mask

# from pyNNsMD.hypers.hyper_schnet_e import DEFAULT_HYPER_PARAM_SCHNET_E
# schnet = SchNetEnergy(**DEFAULT_HYPER_PARAM_SCHNET_E["model"]["config"])
# out = schnet([test_x, test_n, test_edi, test_mask_n, test_mask_e])