"""
Tensorflow keras model definitions for NAC.

There are two definitions: the subclassed NACModel and a precomputed model to 
multiply with the feature derivative for training, which overwrites training/predict step.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from pyNNsMD.nn_pes_src.hypers.hyper_mlp_nac import DEFAULT_HYPER_PARAM_NAC as hyper_create_model_nac
from pyNNsMD.nn_pes_src.layers import InverseDistance,Angles,Dihydral,MLP,ConstLayerNormalization
from pyNNsMD.nn_pes_src.loss import get_lr_metric,r2_metric,nac_loss



class NACModel(ks.Model):
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
        super(NACModel, self).__init__(**kwargs)
        out_dim = int( hyper['states'])
        indim = int( hyper['atoms'])
        use_invdist = hyper['invd_index'] != []
        use_bond_angles = hyper['angle_index'] != []
        angle_index = hyper['angle_index'] 
        use_dihyd_angles = hyper['dihyd_index'] != []
        dihyd_index = hyper['dihyd_index']
        nn_size = hyper['nn_size']
        Depth = hyper['Depth']
        activ = hyper['activ']
        use_reg_activ = hyper['use_reg_activ']
        use_reg_weight = hyper['use_reg_weight']
        use_reg_bias = hyper['use_reg_bias'] 
        use_dropout = hyper['use_dropout']
        dropout = hyper['dropout']

        self.y_atoms = indim
        self.use_dihyd_angles = use_dihyd_angles
        self.use_invdist = use_invdist
        self.use_bond_angles = use_bond_angles
        #geo_input = ks.Input(shape=(indim,3), dtype='float32' ,name='geo_input')
        if(self.use_invdist==True):        
            self.invd_layer = InverseDistance()
        if(self.use_bond_angles==True):
            self.ang_layer = Angles(angle_list=angle_index)
            self.concat_ang = ks.layers.Concatenate(axis=-1)
        if(self.use_dihyd_angles==True):
            self.dih_layer = Dihydral(angle_list=dihyd_index)
            self.concat_dih = ks.layers.Concatenate(axis=-1)
        self.flat_layer = ks.layers.Flatten(name='feat_flat')
        self.std_layer = ConstLayerNormalization(name='feat_std')
        self.mlp_layer = MLP(   nn_size,
                                dense_depth = Depth,
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
        self.virt_layer =  ks.layers.Dense(out_dim*indim,name='virt',use_bias=False,activation='linear')
        self.resh_layer = tf.keras.layers.Reshape((out_dim,indim))
        
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
            feat_flat_std = self.std_layer(feat_flat)
            temp_hidden = self.mlp_layer(feat_flat_std,training=training)
            temp_v = self.virt_layer(temp_hidden)
            temp_va = self.resh_layer(temp_v)
        temp_grad = tape2.batch_jacobian(temp_va, x)
        grad = ks.backend.concatenate([ks.backend.expand_dims(temp_grad[:,:,i,i,:],axis=2) for i in range(self.y_atoms)],axis=2)
        y_pred = grad
        return y_pred


class NACModelPrecomputed(ks.Model):
    def __init__(self,nac_atoms ,nac_states, **kwargs):
        super(NACModelPrecomputed, self).__init__(**kwargs)
        self.nac_atoms = nac_atoms 
        self.nac_states = nac_states
        self.metrics_y_nac_std = tf.constant(np.ones((1,1,1,1)),dtype=tf.float32)
        
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
        self.compiled_metrics.update_state(y*self.metrics_y_nac_std , y_pred*self.metrics_y_nac_std , sample_weight=sample_weight)

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
        self.compiled_metrics.update_state(y*self.metrics_y_nac_std , y_pred*self.metrics_y_nac_std )
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
    out_dim = int(hyper['states'])
    indim = int( hyper['atoms'])
    use_invdist = hyper['invd_index'] != []
    use_bond_angles = hyper['angle_index'] != []
    angle_index = hyper['angle_index'] 
    use_dihyd_angles = hyper['dihyd_index'] != []
    dihyd_index = hyper['dihyd_index']
    nn_size = hyper['nn_size']
    Depth = hyper['Depth']
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
    full = MLP(  nn_size,
         dense_depth = Depth,
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
    nac =  ks.layers.Dense(out_dim*indim,name='virt',use_bias=False,activation='linear')(full)
    #nac = NACGradient(mult_states=out_dim,atoms=indim)([nac ,geo_input])
    #nac = RevertStandardize(val_offset=hyper['y_nac_mean'],val_scale=hyper['y_nac_std']/hyper['y_nac_unit_conv'])(nac)

   
    model = NACModelPrecomputed(inputs=geo_input, outputs=nac,
                     nac_atoms = indim,
                     nac_states = out_dim)
    
    optimizer = tf.keras.optimizers.Adam(lr= learning_rate_start)
    lr_metric = get_lr_metric(optimizer)
    
    if(make_phase_loss == False):
        model.compile(loss='mean_squared_error',optimizer=optimizer,
              metrics=['mean_absolute_error' ,lr_metric,r2_metric])
    else:
        model.compile(loss=nac_loss,optimizer=optimizer,
              metrics=['mean_absolute_error' ,lr_metric,r2_metric])   
    
    return model


class FeatureNacModel(ks.Model):
    def __init__(self, **kwargs):
        super(FeatureNacModel, self).__init__(**kwargs)
    #No training here possible
        
    def predict_step(self, data):
        #Precompute features with gradient:
        x,_,_ = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Compute predictions
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            feat_pred = self(x, training=False)  # Forward pass 
        grad = tape2.batch_jacobian(feat_pred, x)
        return [feat_pred,grad]

    def predict_in_chunks(self,x,batch_size):
        np_x = []
        np_grad = []
        for j in range(int(np.ceil(len(x)/batch_size))):
            a = int(batch_size*j)
            b = int(batch_size*j + batch_size)
            invd,grad = self.predict(x[a:b])
            np_x.append(np.array(invd)) 
            np_grad.append(np.array(grad))
            
        np_x = np.concatenate(np_x,axis=0) 
        np_grad = np.concatenate(np_grad,axis=0)
        return np_x,np_grad


def create_feature_models(hyper,model_name="feat",run_eagerly=False):
    """
    Model to precompute features feat = model(x).

    Args:
        hyper (dict): Hyper dictionary.
        model_name (str, optional): Name of the Model. Defaults to "feat".
        run_eagerly (bool, optional): Whether to run eagerly. Defaults to False.

    Returns:
        model (keras.model): tf.keras model with coordinate input.

    """
    out_dim = int( hyper['states'])
    indim = int( hyper['atoms'])
    use_invdist = hyper['invd_index'] != []
    use_bond_angles = hyper['angle_index'] != []
    angle_index = hyper['angle_index'] 
    use_dihyd_angles = hyper['dihyd_index'] != []
    dihyd_index = hyper['dihyd_index']
    
    geo_input = ks.Input(shape=(indim,3), dtype='float32' ,name='geo_input')
    #Features precompute layer        
    if(use_invdist==True):
        invdlayer = InverseDistance()
        feat = invdlayer(geo_input)
    if(use_bond_angles==True):
        if(use_invdist==False):
            feat = Angles(angle_list=angle_index)(geo_input)
        else:
            angs = Angles(angle_list=angle_index)(geo_input)
            feat = ks.layers.concatenate([feat,angs], axis=-1)
    if(use_dihyd_angles==True):
        if(use_invdist==False and use_bond_angles==False):
            feat = Dihydral(angle_list=dihyd_index)(geo_input)
        else:
            dih = Dihydral(angle_list=dihyd_index)(geo_input)
            feat = ks.layers.concatenate([feat,dih], axis=-1)
    
    feat = ks.layers.Flatten(name='feat_flat')(feat)
    model = FeatureNacModel(inputs=geo_input, outputs=feat,name=model_name)
    
    model.compile(run_eagerly=run_eagerly) 
    #Strange bug with tensorflow.python.framework.errors_impl.InvalidArgumentError: assertion failed: [0] [Op:Assert] name: EagerVariableNameReuse

    
    return model

    