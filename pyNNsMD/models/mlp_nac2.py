# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:57:07 2020

@author: Patrick
"""


import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from pyNNsMD.nn_pes_src.hypers.hyper_mlp_nac import DEFAULT_HYPER_PARAM_NAC as hyper_create_model_nac
from pyNNsMD.layers.features import FeatureGeometric
from pyNNsMD.layers.mlp import MLP
from pyNNsMD.layers.normalize import ConstLayerNormalization
from pyNNsMD.layers.gradients import PropagateNACGradient2
from pyNNsMD.utils.loss import get_lr_metric,r2_metric,NACphaselessLoss,ScaledMeanAbsoluteError



class NACModel2(ks.Model):
    """
    Subclassed tf.keras.model for NACs which outputs NACs from coordinates.
    
    This is not used for fitting, only for prediction as for fitting a feature-precomputed model is used instead.
    The model is supposed to be saved and exported.
    """
    
    def __init__(self,
                atoms,
                states,
                invd_index = [],
                angle_index = [],
                dihed_index = [],
                nn_size = 100,
                depth = 3,
                activ = "selu",
                use_reg_activ = None,
                use_reg_weight = None,
                use_reg_bias = None ,
                use_dropout = False,
                dropout = 0.01,
                **kwargs):
        """
        Initialize a NACModel with hyperparameters.

        Args:
            hyper (dict): Hyperparamters.
            **kwargs (dict): Additional keras.model parameters.

        Returns:
            tf.keras.model.
            
        """
        super(NACModel2, self).__init__(**kwargs)
        out_dim = int(states*(states-1)/2)
        indim = int(atoms)

        self.y_atoms = indim

        # Allow for all distances, backward compatible
        if isinstance(invd_index,bool):
            if invd_index:
                invd_index = [[i,j] for i in range(0,int(atoms)) for j in range(0,i)]

        use_invd_index = len(invd_index)>0 if isinstance(invd_index,list) or isinstance(invd_index,np.ndarray) else False
        use_angle_index = len(angle_index)>0 if isinstance(angle_index,list) or isinstance(angle_index,np.ndarray) else False
        use_dihed_index = len(dihed_index)>0 if isinstance(dihed_index,list) or isinstance(dihed_index,np.ndarray) else False
        
        invd_index = np.array(invd_index,dtype = np.int64) if use_invd_index else None
        angle_index = np.array(angle_index ,dtype = np.int64) if use_angle_index else None
        dihed_index = np.array(dihed_index,dtype = np.int64) if use_dihed_index else None
        
        invd_shape = invd_index.shape if use_invd_index else None
        angle_shape = angle_index.shape if use_angle_index else None
        dihed_shape = dihed_index.shape if use_dihed_index else None

        in_model_dim = 0
        if(use_invd_index==True):
            in_model_dim += len(invd_index)
        if(use_angle_index == True):
            in_model_dim += len(angle_index) 
        if(use_dihed_index == True):
            in_model_dim += len(dihed_index)

    
        self.feat_layer = FeatureGeometric(invd_shape = invd_shape,
                                           angle_shape = angle_shape,
                                           dihed_shape = dihed_shape,
                                           name="feat_geo"
                                           )
        self.feat_layer.set_mol_index(invd_index, angle_index , dihed_index)
        
        
        self.std_layer = ConstLayerNormalization(name='feat_std')
        self.mlp_layer = MLP(   nn_size,
                                dense_depth = depth,
                                dense_bias = True,
                                dense_bias_last = False,
                                dense_activ = activ,
                                dense_activ_last = activ,
                                dense_activity_regularizer = use_reg_activ,
                                dense_kernel_regularizer = use_reg_weight,
                                dense_bias_regularizer = use_reg_bias,
                                dropout_use = use_dropout,
                                dropout_dropout = dropout,
                                name = 'mlp'
                                )
        self.virt_layer =  ks.layers.Dense(out_dim*in_model_dim,name='virt',use_bias=False,activation='linear')
        self.resh_layer = tf.keras.layers.Reshape((out_dim,in_model_dim))
        self.prop_grad_layer = PropagateNACGradient2(axis=(2, 1))

        self.precomputed_features = False

        self.build((None,indim,3))

    def call(self, data, training=False):
        """
        Call the model output, forward pass.

        Args:
            data (tf.tensor): Coordinates.
            training (bool, optional): Training Mode. Defaults to False.

        Returns:
            y_pred (tf.tensor): predicted NACs.

        """
        x = data
        # Compute predictions
        if not self.precomputed_features:
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                feat_flat = self.feat_layer(x)
            temp_grad = tape2.batch_jacobian(feat_flat, x)

            feat_flat_std = self.std_layer(feat_flat)
            temp_hidden = self.mlp_layer(feat_flat_std,training=training)

            temp_v = self.virt_layer(temp_hidden)
            temp_va = self.resh_layer(temp_v)
            # y_pred = ks.backend.batch_dot(temp_va,temp_grad ,axes=(2,1))
            y_pred = self.prop_grad_layer([temp_va,temp_grad])
        else:
            x1 = x[0]
            x2 = x[1]
            feat_flat_std = self.std_layer(x1)
            temp_hidden = self.mlp_layer(feat_flat_std, training=training)
            temp_v = self.virt_layer(temp_hidden)
            temp_va = self.resh_layer(temp_v)
            # y_pred = ks.backend.batch_dot(temp_va, x2, axes=(2, 1))
            y_pred = self.prop_grad_layer([temp_va, x2])

        return y_pred

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