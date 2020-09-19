"""
Tensorflow keras model definitions for features and descriptors.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from pyNNsMD.nn_pes_src.layers import InverseDistance,Angles,Dihydral,MLP,EmptyGradient
from pyNNsMD.nn_pes_src.loss import get_lr_metric,r2_metric,nac_loss


class FeatureModel(ks.Model):
    def __init__(self, **kwargs):
        super(FeatureModel, self).__init__(**kwargs)
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
    model = FeatureModel(inputs=geo_input, outputs=feat,name=model_name)
    
    model.compile(run_eagerly=run_eagerly) 
    #Strange bug with tensorflow.python.framework.errors_impl.InvalidArgumentError: assertion failed: [0] [Op:Assert] name: EagerVariableNameReuse

    
    return model