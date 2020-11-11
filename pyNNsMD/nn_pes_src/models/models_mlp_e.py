"""
Tensorflow keras model definitions for energy and gradient.

There are two definitions: the subclassed EnergyModel and a precomputed model to 
train energies. The subclassed Model will also predict gradients.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks


from pyNNsMD.nn_pes_src.hypers.hyper_mlp_e import DEFAULT_HYPER_PARAM_ENERGY as hyper_model_energy
from pyNNsMD.nn_pes_src.keras_utils.layers import MLP,ConstLayerNormalization,FeatureGeometric
from pyNNsMD.nn_pes_src.keras_utils.loss import get_lr_metric,r2_metric,ScaledMeanAbsoluteError


class EnergyModel(ks.Model):
    """
    Subclassed tf.keras.model for energy/gradient which outputs both energy and gradient from coordinates.
    
    This is not used for fitting, only for prediction as for fitting a feature-precomputed model is used instead.
    The model is supposed to be saved and exported for MD code.
    """
    
    def __init__(self,hyper, **kwargs):
        """
        Initialize an EnergyModel with hyperparameters.

        Args:
            hyper (dict): Hyperparamters.
            **kwargs (dict): Additional keras.model parameters.

        Returns:
            tf.keras.model.
            
        """
        super(EnergyModel, self).__init__(**kwargs)
        out_dim = int( hyper['states'])
        indim = int( hyper['atoms'])
        #invd_index = np.array(hyper['invd_index'])
        angle_index = np.array(hyper['angle_index'])
        dihyd_index = np.array(hyper['dihyd_index'])
        nn_size = hyper['nn_size']
        depth = hyper['depth']
        activ = hyper['activ']
        use_reg_activ = hyper['use_reg_activ']
        use_reg_weight = hyper['use_reg_weight']
        use_reg_bias = hyper['use_reg_bias'] 
        use_dropout = hyper['use_dropout']
        dropout = hyper['dropout']
        
        use_bond_angles = angle_index.shape if len(angle_index)>0 else None
        use_dihyd_angles = dihyd_index.shape if len(dihyd_index)>0 else None
        self.feat_layer = FeatureGeometric(invd_shape = None,
                                           angle_shape = use_bond_angles,
                                           dihyd_shape = use_dihyd_angles,
                                           )
        self.feat_layer.set_mol_index(None, angle_index , dihyd_index)
        
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
        

        self.build((None,indim,3))
        
    def call(self, data, training=False):
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
        # Compute predictions
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            feat_flat = self.feat_layer(x)
            feat_flat_std = self.std_layer(feat_flat)
            temp_hidden = self.mlp_layer(feat_flat_std,training=training)
            temp_e = self.energy_layer(temp_hidden)
        temp_g = tape2.batch_jacobian(temp_e, x)
        y_pred = [temp_e,temp_g]
        return y_pred


def create_model_energy_precomputed(hyper=hyper_model_energy['model'],
                                             learning_rate_start = 1e-3):
    """
    Full Model y = model(feat) with feat=[f,df/dx] features and its derivative to coordinates x.

    Args:
        hyper (dict, optional): Hyper dictionary. The default is hyper_model_energy_gradient['model'].
        learning_rate_start (float, optional): Initial Learning rate in compile. Defaults to 1e-3.
        loss_weights (list, optional): Weights between energy and gradient. defualt is [1,1]

    Returns:
        model (TYPE): DESCRIPTION.

    """
    out_dim = hyper['states']
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

    geo_input = ks.Input(shape=(in_model_dim,), dtype='float32' ,name='geo_input')
    #grad_input = ks.Input(shape=(in_model_dim,indim,3), dtype='float32' ,name='grad_input')
    
    full = ks.layers.Flatten(name='feat_flat')(geo_input)
    full = ConstLayerNormalization(name='feat_std')(full)
    full = MLP( nn_size,
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
             )(full)
    
    energy =  ks.layers.Dense(out_dim,name='energy',use_bias=True,activation='linear')(full)

    
    model = tf.keras.Model(inputs=geo_input, outputs=energy)
    
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate_start)
    lr_metric = get_lr_metric(optimizer)
    scmae = ScaledMeanAbsoluteError(scaling_shape=(1,out_dim))
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=[scmae  ,lr_metric,r2_metric])
    
    return model,scmae 
