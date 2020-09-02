"""
Old unused functions. May not be working anymore.

@author: Patrick
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from pyNNsMD.nn_pes_src.layers import InverseDistance,Angles,Dihydral


def compute_feature_derivative(x,hyper,batch_size):
    
    #Features precompute layer
    invdlayer = InverseDistance(dinv_mean=hyper['invd_mean'],dinv_std=hyper['invd_std'])
    anglelayer = Angles(angle_list=hyper['angle_index'],angle_offset=hyper['angle_mean'],angle_std = hyper['angle_std'])
    dihydlayer = Dihydral(angle_list=hyper['dihyd_index'],angle_offset=hyper['dihyd_mean'],angle_std = hyper['dihyd_std'])
    
    #Precompute features with gradient:
    #batch_size = hyper['batch_size']
    
    @tf.function
    def tf_comp_feat_dev(tfx):
        with tf.GradientTape() as tape:
            tape.watch(tfx)
            invd = invdlayer(tfx)
            if(hyper['use_bond_angles']==True):
                angs = anglelayer(tfx)
                invd = tf.concat([invd,angs], axis=-1)
            if(hyper['use_dihyd_angles']==True):
                dihy = dihydlayer(tfx)
                invd = tf.concat([invd,dihy], axis=-1)
        grad= tape.batch_jacobian(invd,tfx)
        return invd,grad
    
    #Precompute features with gradient:
    #batch_size = hyper['batch_size']
    np_x = []
    np_grad = []
    for j in range(int(np.ceil(len(x)/batch_size))):
        a = batch_size*j
        b = batch_size*j + batch_size
        tab = tf.constant(x[a:b],dtype=tf.float32) 
        invd,grad = tf_comp_feat_dev(tab)
        np_x.append(invd) 
        np_grad.append(grad)
        
    np_x = tf.concat(np_x,axis=0) 
    np_grad = tf.concat(np_grad,axis=0)
    np_x = np_x.numpy()
    np_grad = np_grad.numpy()
    return np_x,np_grad