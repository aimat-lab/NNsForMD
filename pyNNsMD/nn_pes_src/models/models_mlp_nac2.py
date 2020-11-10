# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:57:07 2020

@author: Patrick
"""


import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from pyNNsMD.nn_pes_src.hypers.hyper_mlp_nac import DEFAULT_HYPER_PARAM_NAC as hyper_create_model_nac
from pyNNsMD.nn_pes_src.keras_utils.layers import MLP,ConstLayerNormalization,PropagateNACGradient2,FeatureGeometric
from pyNNsMD.nn_pes_src.keras_utils.loss import get_lr_metric,r2_metric,NACphaselessLoss,ScaledMeanAbsoluteError



class NACModel2(ks.Model):
    """
    Subclassed tf.keras.model for NACs which outputs NACs from coordinates.
    
    This is not used for fitting, only for prediction as for fitting a feature-precomputed model is used instead.
    The model is supposed to be saved and exported.
    """
    
    def __init__(self,hyper, **kwargs):
        """
        Initialize a NACModel with hyperparameters.

        Args:
            hyper (dict): Hyperparamters.
            **kwargs (dict): Additional keras.model parameters.

        Returns:
            tf.keras.model.
            
        """
        super(NACModel2, self).__init__(**kwargs)
        out_dim = int( hyper['states']*(hyper['states']-1)/2)
        indim = int( hyper['atoms'])
        use_invdist = hyper['invd_index'] != []
        use_bond_angles = hyper['angle_index'] != []
        angle_index = hyper['angle_index'] 
        use_dihyd_angles = hyper['dihyd_index'] != []
        dihyd_index = hyper['dihyd_index']
        nn_size = hyper['nn_size']
        depth = hyper['depth']
        activ = hyper['activ']
        use_reg_activ = hyper['use_reg_activ']
        use_reg_weight = hyper['use_reg_weight']
        use_reg_bias = hyper['use_reg_bias'] 
        use_dropout = hyper['use_dropout']
        dropout = hyper['dropout']
        
        in_model_dim = 0
        if(use_invdist==True):
            in_model_dim += int(indim*(indim-1)/2)
        if(use_bond_angles == True):
            in_model_dim += len(angle_index) 
        if(use_dihyd_angles == True):
            in_model_dim += len(dihyd_index) 

        self.y_atoms = indim
        self.feat_layer = FeatureGeometric(invd_index = use_invdist,
                                angle_index = angle_index ,
                                dihyd_index = dihyd_index,
                                )
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
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            feat_flat = self.feat_layer(x)
        temp_grad = tape2.batch_jacobian(feat_flat, x)
        
        feat_flat_std = self.std_layer(feat_flat)
        temp_hidden = self.mlp_layer(feat_flat_std,training=training)
        
        temp_v = self.virt_layer(temp_hidden)
        temp_va = self.resh_layer(temp_v)
        
        y_pred = ks.backend.batch_dot(temp_va,temp_grad ,axes=(2,1)) # (batch,states,atoms,atoms,3)
        return y_pred




def create_model_nac_precomputed(hyper=hyper_create_model_nac['model'],
                                 learning_rate_start = 1e-3,
                                 make_phase_loss = False):
    """
    Get precomputed withmodel y = model(feat) with feat=[f,df/dx] features and its derivative to coordinates x.

    Args:
        hyper (dict, optional): Hyper model dictionary. The default is hyper_create_model_nac['model'].
        learning_rate_start (float, optional): Initial learning rate. Default is 1e-3.
        make_phase_loss (bool, optional): Use normal loss MSE regardless of hyper. The default is False.

    Returns:
        model (tf.keras.model): tf.keras model.

    """
    num_outstates = int(hyper['states'])
    indim = int( hyper['atoms'])
    use_invdist = hyper['invd_index'] != []
    use_bond_angles = hyper['angle_index'] != []
    angle_index = hyper['angle_index'] 
    use_dihyd_angles = hyper['dihyd_index'] != []
    dihyd_index = hyper['dihyd_index']
    nn_size = hyper['nn_size']
    depth = hyper['depth']
    activ = hyper['activ']
    use_reg_activ = hyper['use_reg_activ']
    use_reg_weight = hyper['use_reg_weight']
    use_reg_bias = hyper['use_reg_bias'] 
    use_dropout = hyper['use_dropout']
    dropout = hyper['dropout']
    
    out_dim = int(num_outstates*(num_outstates-1)/2)
    
    in_model_dim = 0
    if(use_invdist==True):
        in_model_dim += int(indim*(indim-1)/2)
    if(use_bond_angles == True):
        in_model_dim += len(angle_index) 
    if(use_dihyd_angles == True):
        in_model_dim += len(dihyd_index) 

    geo_input = ks.Input(shape=(in_model_dim,), dtype='float32' ,name='geo_input')
    grad_input = ks.Input(shape=(in_model_dim,indim,3), dtype='float32' ,name='grad_input')
    full = ks.layers.Flatten(name='feat_flat')(geo_input)
    full = ConstLayerNormalization(name='feat_std')(full)
    full = MLP(  nn_size,
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
         )(full)
    nac =  ks.layers.Dense(out_dim*in_model_dim,name='virt',use_bias=False,activation='linear')(full)
    nac = tf.keras.layers.Reshape((out_dim,in_model_dim))(nac)
    nac = PropagateNACGradient2()([nac,grad_input])
   
    model = ks.Model(inputs=[geo_input,grad_input], outputs=nac)
    
    optimizer = tf.keras.optimizers.Adam(lr= learning_rate_start)
    lr_metric = get_lr_metric(optimizer)
    smae = ScaledMeanAbsoluteError(scaling_shape=(1,out_dim,indim,1))
    
    if(make_phase_loss == False):
        model.compile(loss='mean_squared_error',optimizer=optimizer,
              metrics=[smae ,lr_metric,r2_metric])
    else:
        model.compile(loss=NACphaselessLoss(number_state = num_outstates, shape_nac = (indim,3),name="phaseless_loss"),optimizer=optimizer,
              metrics=[smae ,lr_metric,r2_metric])   
    
    return model,smae
