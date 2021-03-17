"""
Tensorflow keras model definitions for energy and gradient.

There are two definitions: the subclassed EnergyGradientModel and a precomputed model to 
multiply with the feature derivative for training, which overwrites training/predict step.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks


from pyNNsMD.nn_pes_src.hypers.hyper_mlp_eg import DEFAULT_HYPER_PARAM_ENERGY_GRADS as hyper_model_energy_gradient
from pyNNsMD.layers.features import FeatureGeometric
from pyNNsMD.layers.mlp import MLP
from pyNNsMD.layers.normalize import ConstLayerNormalization
from pyNNsMD.layers.gradients import EmptyGradient
from pyNNsMD.utils.loss import get_lr_metric,r2_metric


class EnergyGradientModel(ks.Model):
    """
    Subclassed tf.keras.model for energy/gradient which outputs both energy and gradient from coordinates.
    
    This is not used for fitting, only for prediction as for fitting a feature-precomputed model is used instead.
    The model is supposed to be saved and exported for MD code.
    """

    def __init__(self,
                states = 1,
                atoms =  2 ,
                invd_index = None,
                angle_index = None,
                dihyd_index = None,
                nn_size = 100,
                depth = 3,
                activ = 'selu',
                use_reg_activ = None,
                use_reg_weight = None,
                use_reg_bias = None,
                use_dropout = False,
                dropout = 0.01,
                **kwargs):
        """
        Initialize Layer

        Args:
            states:
            atoms:
            invd_index:
            angle_index:
            dihyd_index:
            nn_size:
            depth:
            activ:
            use_reg_activ:
            use_reg_weight:
            use_reg_bias:
            use_dropout:
            dropout:
            **kwargs:
        """

        super(EnergyGradientModel, self).__init__(**kwargs)

        out_dim = int(states)
        indim = int(atoms)

        use_invd_index = len(invd_index)>0 if isinstance(invd_index,list) or isinstance(invd_index,np.ndarray) else False
        use_angle_index = len(angle_index)>0 if isinstance(angle_index,list) or isinstance(angle_index,np.ndarray) else False
        use_dihyd_index = len(dihyd_index)>0 if isinstance(dihyd_index,list) or isinstance(dihyd_index,np.ndarray) else False
        
        invd_index = np.array(invd_index,dtype = np.int64) if use_invd_index else None
        angle_index = np.array(angle_index ,dtype = np.int64) if use_angle_index else None
        dihyd_index = np.array(dihyd_index,dtype = np.int64) if use_dihyd_index else None
        
        invd_shape = invd_index.shape if use_invd_index else None
        angle_shape = angle_index.shape if use_angle_index else None
        dihyd_shape = dihyd_index.shape if use_dihyd_index else None
    
        self.feat_layer = FeatureGeometric(invd_shape = invd_shape,
                                           angle_shape = angle_shape,
                                           dihyd_shape = dihyd_shape,
                                           )
        self.feat_layer.set_mol_index(invd_index, angle_index , dihyd_index)
        
        self.std_layer = ConstLayerNormalization(axis=-1,name='feat_std')
        self.mlp_layer = MLP( nn_size,
                 dense_depth = depth,
                 dense_bias = True,
                 dense_bias_last = True,
                 dense_activ = activ,
                 dense_activ_last = activ,
                 dense_activity_regularizer = use_reg_activ,
                 dense_kernel_regularizer = use_reg_weight,
                 dense_bias_regularizer = use_reg_bias,
                 dropout_use = use_dropout,
                 dropout_dropout = dropout,
                 name = 'mlp'
                 )
        self.energy_layer =  ks.layers.Dense(out_dim,name='energy',use_bias=True,activation='linear')
        self.force = EmptyGradient(name='force')  # Will be differentiated in fit/predict/evaluate

        # Control the properties
        self.energy_only = False
        self.precomputed_features = False
        self.output_as_dict = False

        # Will remove later
        self.eg_atoms = atoms
        self.eg_states = states
        self.build((None,indim,3))
        
    def call(self, data, training=False, **kwargs):
        """
        Call the model output, forward pass.

        Args:
            data (tf.tensor): Coordinates.
            training (bool, optional): Training Mode. Defaults to False.

        Returns:
            y_pred (list): List of tf.tensor for predicted [energy,gradient]

        """
        # Unpack the data
        x = data

        if self.energy_only and not self.precomputed_features:
            feat_flat = self.feat_layer(x)
            feat_flat_std = self.std_layer(feat_flat)
            temp_hidden = self.mlp_layer(feat_flat_std, training=training)
            temp_e = self.energy_layer(temp_hidden)
            temp_g = self.force(x)
            y_pred = [temp_e, temp_g]
        elif not self.energy_only and not self.precomputed_features:
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                feat_flat = self.feat_layer(x)
                feat_flat_std = self.std_layer(feat_flat)
                temp_hidden = self.mlp_layer(feat_flat_std,training=training)
                temp_e = self.energy_layer(temp_hidden)
            temp_g = tape2.batch_jacobian(temp_e, x)
            _ = self.force(x)
            y_pred = [temp_e,temp_g]
        elif self.precomputed_features:
            x1 = x[0]
            x2 = x[1]
            with tf.GradientTape() as tape2:
                tape2.watch(x1)
                feat_flat_std = self.std_layer(x1)
                temp_hidden = self.mlp_layer(feat_flat_std, training=training)
                atpot = self.energy_layer(temp_hidden)
            grad = tape2.batch_jacobian(atpot, x1)
            grad = ks.backend.batch_dot(grad, x2, axes=(2, 1))
            y_pred = [atpot, grad]

        if self.output_as_dict:
            out = {'energy': y_pred[0], 'force' : y_pred[1]}
        else:
            out = y_pred
        return out


    @tf.function
    def predict_chunk_feature(self, tf_x):
        with tf.GradientTape() as tape2:
            tape2.watch(tf_x)
            feat_pred = self.feat_layer(tf_x, training=False)  # Forward pass
        grad = tape2.batch_jacobian(feat_pred, tf_x)
        return feat_pred, grad

    def precompute_feature_in_chunks(self, x, batch_size):
        np_x = []
        np_grad = []
        for j in range(int(np.ceil(len(x) / batch_size))):
            a = int(batch_size * j)
            b = int(batch_size * j + batch_size)
            tf_x = tf.convert_to_tensor(x[a:b], dtype=tf.float32)
            feat_pred, grad = self.predict_chunk_feature(tf_x)
            np_x.append(np.array(feat_pred.numpy()))
            np_grad.append(np.array(grad.numpy()))

        np_x = np.concatenate(np_x, axis=0)
        np_grad = np.concatenate(np_grad, axis=0)
        return np_x, np_grad