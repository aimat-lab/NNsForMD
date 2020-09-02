"""
Tensorflow keras model definitions for NAC.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from pyNNsMD.nn_pes_src.hyper import DEFAULT_HYPER_PARAM_NAC as hyper_create_model_nac
from pyNNsMD.nn_pes_src.reg import identify_regularizer
from pyNNsMD.nn_pes_src.layers import InverseDistance,Angles,Dihydral,MLP,EmptyGradient,RevertStandardize
from pyNNsMD.nn_pes_src.activ import identify_keras_activation,leaky_softplus,shifted_sofplus
from pyNNsMD.nn_pes_src.loss import get_lr_metric,r2_metric,nac_loss



class NACModel(ks.Model):
    """
    Subclassed tf.keras.model for NACs which outputs NACs from coordinates.
    This is not used for fitting, only for prediction as for fitting a feature-precomputed model is used instead.
    The model is supposed to be saved and exported.
    """
    def __init__(self,hyper, **kwargs):
        super(NACModel, self).__init__(**kwargs)
        out_dim = int( hyper['states'])
        indim = int( hyper['atoms'])
        use_invdist = hyper['use_invdist']
        invd_mean = hyper['invd_mean']
        invd_std = hyper['invd_std']
        use_bond_angles = hyper['use_bond_angles']
        angle_index = hyper['angle_index']
        angle_mean = hyper['angle_mean']
        angle_std = hyper['angle_std']
        use_dihyd_angles = hyper['use_dihyd_angles']
        dihyd_index = hyper['dihyd_index']
        dihyd_mean = hyper['dihyd_mean']
        dihyd_std = hyper['dihyd_std']
        nn_size = hyper['nn_size']
        Depth = hyper['Depth']
        activ = hyper['activ']
        activ_alpha = hyper['activ_alpha']
        use_reg_activ = hyper['use_reg_activ']
        use_reg_weight = hyper['use_reg_weight']
        use_reg_bias = hyper['use_reg_bias'] 
        reg_l1 = hyper['reg_l1']
        reg_l2 = hyper['reg_l2']
        use_dropout = hyper['use_dropout']
        dropout = hyper['dropout']
        y_nac_unit_conv = hyper['y_nac_unit_conv']
        y_nac_std = hyper['y_nac_std'] 
        y_nac_mean = hyper['y_nac_mean']

        self.y_atoms = indim
        self.use_dihyd_angles = use_dihyd_angles
        self.use_invdist = use_invdist
        self.use_bond_angles = use_bond_angles
        #geo_input = ks.Input(shape=(indim,3), dtype='float32' ,name='geo_input')
        if(self.use_invdist==True):        
            self.invd_layer = InverseDistance(dinv_mean=invd_mean,dinv_std=invd_std)
        if(self.use_bond_angles==True):
            self.ang_layer = Angles(angle_list=angle_index,angle_offset=angle_mean,angle_std =angle_std)
            self.concat_ang = ks.layers.Concatenate(axis=-1)
        if(self.use_dihyd_angles==True):
            self.dih_layer = Dihydral(angle_list=dihyd_index,angle_offset=dihyd_mean,angle_std = dihyd_std)
            self.concat_dih = ks.layers.Concatenate(axis=-1)
        self.flat_layer = ks.layers.Flatten(name='feat_flat')
        self.mlp_layer = MLP(   nn_size,
                                dense_depth = Depth,
                                dense_bias = True,
                                dense_bias_last = False,
                                dense_activ = identify_keras_activation(activ,alpha=activ_alpha),
                                dense_activ_last = identify_keras_activation(activ,alpha=activ_alpha),
                                dense_activity_regularizer = identify_regularizer(use_reg_activ,reg_l1,reg_l2),
                                dense_kernel_regularizer = identify_regularizer(use_reg_weight,reg_l1,reg_l2),
                                dense_bias_regularizer = identify_regularizer(use_reg_bias,reg_l1,reg_l2),
                                dropout_use = use_dropout,
                                dropout_dropout = dropout,
                                name = 'mlp'
                                )
        self.virt_layer =  ks.layers.Dense(out_dim*indim,name='virt',use_bias=False,activation='linear')
        self.resh_layer = tf.keras.layers.Reshape((out_dim,indim))
        self.rev_std_nac = RevertStandardize(name='rev_std_nac',val_offset = y_nac_mean,val_scale = y_nac_std/y_nac_unit_conv) 
        
        self.build((None,indim,3))
    def call(self, data, training=False):
        x = data
        # Compute predictions
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            if(self.use_invdist==True):
                feat = self.invd_layer(x)
            if(self.use_bond_angles==True):
                if(self.use_invdist==False):
                    feat = self.ang_layer(x)
                else:
                    angs = self.ang_layer(x)
                    feat = self.concat_ang([feat,angs],axis=-1)
            if(self.use_dihyd_angles==True):
                if(self.use_invdist==False and self.use_bond_angles==False):
                    feat = self.dih_layer(x)
                else:
                    dih = self.dih_layer(x)
                    feat = self.concat_dih([feat,dih],axis=-1)

            feat_flat = self.flat_layer(feat)
            temp_hidden = self.mlp_layer(feat_flat)
            temp_v = self.virt_layer(temp_hidden)
            temp_va = self.resh_layer(temp_v)
        temp_grad = tape2.batch_jacobian(temp_va, x)
        grad = ks.backend.concatenate([ks.backend.expand_dims(temp_grad[:,:,i,i,:],axis=2) for i in range(self.y_atoms)],axis=2)
        y_pred = self.rev_std_nac(grad)
        return y_pred


class NACModelPrecomputed(ks.Model):
    def __init__(self,nac_atoms ,nac_states, **kwargs):
        super(NACModelPrecomputed, self).__init__(**kwargs)
        self.nac_atoms = nac_atoms 
        self.nac_states = nac_states
 
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        
        x1 = x[0]
        x2 = x[1]
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                tape2.watch(x1)
                atpot = self(x1, training=True)  # Forward pass      
            grad = tape2.batch_jacobian(atpot, x1)
            grad = ks.backend.reshape(grad,(ks.backend.shape(x1)[0],self.nac_states,self.nac_atoms,ks.backend.shape(grad)[2]))
            grad = ks.backend.batch_dot(grad,x2,axes=(3,1)) # (batch,states,atoms,atoms,3)
            y_pred = ks.backend.concatenate([ks.backend.expand_dims(grad[:,:,i,i,:],axis=2) for i in range(self.nac_atoms)],axis=2)
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        # Unpack the data
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Compute predictions
        x1 = x[0]
        x2 = x[1]
        with tf.GradientTape() as tape2:
            tape2.watch(x1)
            atpot = self(x1, training=False)  # Forward pass 
        grad = tape2.batch_jacobian(atpot, x1)
        grad = ks.backend.reshape(grad,(ks.backend.shape(x1)[0],self.nac_states,self.nac_atoms,ks.backend.shape(grad)[2]))
        grad = ks.backend.batch_dot(grad,x2,axes=(3,1))            
        y_pred = ks.backend.concatenate([ks.backend.expand_dims(grad[:,:,i,i,:],axis=2) for i in range(self.nac_atoms)],axis=2)

        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        # Unpack the data
        x,_,_ = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Compute predictions
        x1 = x[0]
        x2 = x[1]
        with tf.GradientTape() as tape2:
            tape2.watch(x1)
            atpot = self(x1, training=False)  # Forward pass 
        grad = tape2.batch_jacobian(atpot, x1)
        grad = ks.backend.reshape(grad,(ks.backend.shape(x1)[0],self.nac_states,self.nac_atoms,ks.backend.shape(grad)[2]))
        grad = ks.backend.batch_dot(grad,x2,axes=(3,1))            
        y_pred = ks.backend.concatenate([ks.backend.expand_dims(grad[:,:,i,i,:],axis=2) for i in range(self.nac_atoms)],axis=2)  
        return y_pred




def create_model_nac_precomputed(hyper=hyper_create_model_nac, force_phase_loss = False):
    """
    Get precomputed withmodel y = model(feat) with feat=[f,df/dx] features 
    and its derivative to coordinates x.

    Parameters
    ----------
    hyper : dict, optional
        Hyper dictionary. The default is hyper_create_model_nac.
    force_phase_loss : bool, optional
        Use normal loss MSE regardless of hyper. The default is False.
        
    Returns
    -------
    model : tf.keras.model
        tf.keras model.

    """
    out_dim = int(hyper['states'])
    indim = int( hyper['atoms'])
    use_invdist = hyper['use_invdist']
    use_bond_angles = hyper['use_bond_angles']
    use_dihyd_angles = hyper['use_dihyd_angles']
    angle_index = hyper['angle_index']
    dihyd_index = hyper['dihyd_index']
    nn_size = hyper['nn_size']
    Depth = hyper['Depth']
    activ = hyper['activ']
    activ_alpha = hyper['activ_alpha']
    use_reg_activ = hyper['use_reg_activ']
    use_reg_weight = hyper['use_reg_weight']
    use_reg_bias = hyper['use_reg_bias'] 
    reg_l1 = hyper['reg_l1']
    reg_l2 = hyper['reg_l2']
    use_dropout = hyper['use_dropout']
    dropout = hyper['dropout']
    learning_rate_start = hyper['learning_rate_compile']
    phase_less_loss = hyper['phase_less_loss']
    
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
    full = MLP(  nn_size,
         dense_depth = Depth,
         dense_bias = True,
         dense_bias_last = False,
         dense_activ = identify_keras_activation(activ,alpha=activ_alpha),
         dense_activ_last = identify_keras_activation(activ,alpha=activ_alpha),
         dense_activity_regularizer = identify_regularizer(use_reg_activ,reg_l1,reg_l2),
         dense_kernel_regularizer = identify_regularizer(use_reg_weight,reg_l1,reg_l2),
         dense_bias_regularizer = identify_regularizer(use_reg_bias,reg_l1,reg_l2),
         dropout_use = use_dropout,
         dropout_dropout = dropout,
         name = 'mlp'
         )(full)
    nac =  ks.layers.Dense(out_dim*indim,name='virt',use_bias=False,activation='linear')(full)
    #nac = NACGradient(mult_states=out_dim,atoms=indim)([nac ,geo_input])
    #nac = RevertStandardize(val_offset=hyper['y_nac_mean'],val_scale=hyper['y_nac_std']/hyper['y_nac_unit_conv'])(nac)

   
    model = NACModelPrecomputed(inputs=geo_input, outputs=nac,
                     nac_atoms = indim,
                     nac_states = out_dim)
    
    optimizer = tf.keras.optimizers.Adam(lr= learning_rate_start)
    lr_metric = get_lr_metric(optimizer)
    
    if(phase_less_loss == False or force_phase_loss == True):
        model.compile(loss='mean_squared_error',optimizer=optimizer,
              metrics=['mean_absolute_error' ,lr_metric,r2_metric])
    else:
        model.compile(loss=nac_loss,optimizer=optimizer,
              metrics=['mean_absolute_error' ,lr_metric,r2_metric])   
    
    return model


def create_model_nac(hyper=hyper_create_model_nac):
    """
    Get full model with potential = model(x) with coordinates x. The gradients are computed 
    by predict_step() or seperately in tf.GradientTape(). See subclassed NACModel().

    Parameters
    ----------
    hyper : dict, optional
        Hyper dictionary. The default is hyper_create_model_nac.
        
    Returns
    -------
    model : tf.keras.model
        tf.keras model.

    """
    out_dim = int( hyper['states'])
    indim = int( hyper['atoms'])
    use_invdist = hyper['use_invdist']
    invd_mean = hyper['invd_mean']
    invd_std = hyper['invd_std']
    use_bond_angles = hyper['use_bond_angles']
    angle_index = hyper['angle_index']
    angle_mean = hyper['angle_mean']
    angle_std = hyper['angle_std']
    use_dihyd_angles = hyper['use_dihyd_angles']
    dihyd_index = hyper['dihyd_index']
    dihyd_mean = hyper['dihyd_mean']
    dihyd_std = hyper['dihyd_std']
    nn_size = hyper['nn_size']
    Depth = hyper['Depth']
    activ = hyper['activ']
    activ_alpha = hyper['activ_alpha']
    use_reg_activ = hyper['use_reg_activ']
    use_reg_weight = hyper['use_reg_weight']
    use_reg_bias = hyper['use_reg_bias'] 
    reg_l1 = hyper['reg_l1']
    reg_l2 = hyper['reg_l2']
    use_dropout = hyper['use_dropout']
    dropout = hyper['dropout']
    learning_rate_start = hyper['learning_rate_compile']
    
    # Input Coordinates
    geo_input = ks.Input(shape=(indim,3), dtype='float32' ,name='geo_input')
   
    #Features precompute layer        
    if(use_invdist==True):
        invdlayer = InverseDistance(dinv_mean=invd_mean,dinv_std=invd_std)
        feat = invdlayer(geo_input)
    if(use_bond_angles==True):
        if(use_invdist==False):
            feat = Angles(angle_list=angle_index,angle_offset=angle_mean,angle_std =angle_std)(geo_input)
        else:
            angs = Angles(angle_list=angle_index,angle_offset=angle_mean,angle_std =angle_std)(geo_input)
            feat = ks.layers.concatenate([feat,angs], axis=-1)
    if(use_dihyd_angles==True):
        if(use_invdist==False and use_bond_angles==False):
            feat = Dihydral(angle_list=dihyd_index,angle_offset=dihyd_mean,angle_std = dihyd_std)(geo_input)
        else:
            dih = Dihydral(angle_list=dihyd_index,angle_offset=dihyd_mean,angle_std = dihyd_std)(geo_input)
            feat = ks.layers.concatenate([feat,dih], axis=-1)
    
    feat = ks.layers.Flatten(name='feat_flat')(feat)

    # Actual Model with NN as ff-NN
    full = feat
    full = MLP(  nn_size,
         dense_depth = Depth,
         dense_bias = True,
         dense_bias_last = False,
         dense_activ = identify_keras_activation(activ,alpha=activ_alpha),
         dense_activ_last = identify_keras_activation(activ,alpha=activ_alpha),
         dense_activity_regularizer = identify_regularizer(use_reg_activ,reg_l1,reg_l2),
         dense_kernel_regularizer = identify_regularizer(use_reg_weight,reg_l1,reg_l2),
         dense_bias_regularizer = identify_regularizer(use_reg_bias,reg_l1,reg_l2),
         dropout_use = use_dropout,
         dropout_dropout = dropout,
         name = 'mlp'
         )(full)
    nac =  ks.layers.Dense(out_dim*indim,name='virt',use_bias=False,activation='linear')(full)    
   
    model = NACModel(inputs=geo_input, outputs=nac,
                     nac_atoms = indim,
                     nac_states = out_dim)
    
    #Compile model
    optimizer = tf.keras.optimizers.Adam(lr= learning_rate_start)
    model.compile(loss='mean_squared_error',optimizer=optimizer,
          metrics=['mean_absolute_error' ])

    
    return model



    