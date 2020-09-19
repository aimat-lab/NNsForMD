"""
Old unused functions. May not be working anymore.
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


def predict_model_energy_gradient_precomputed(ml,mf,x,hyper):

    batch_size_predict = hyper['predict']['batch_size_predict']
    y_energy_unit_conv = hyper['model']['y_energy_unit_conv']
    y_gradient_unit_conv = hyper['model']['y_gradient_unit_conv']
    y_energy_std = hyper['model']['y_energy_std']
    y_energy_mean = hyper['model']['y_energy_mean']
    
    #Prediction
    feat,feat_grad = mf.predict(x,batch_size = batch_size_predict)
    temp_e,temp_g = ml.predict([feat,feat_grad],batch_size = batch_size_predict) 
    #Scaling
    temp_e = temp_e / y_energy_unit_conv * y_energy_std + y_energy_mean #/27.21138624598853-156.22214381375588
    temp_g = temp_g / y_gradient_unit_conv * y_energy_std
    return temp_e,temp_g





def predict_model_nac_precomputed(ml,mf,x,hyper):

    batch_size_predict = hyper['predict']['batch_size_predict']
    y_nac_unit_conv = hyper['model']['y_nac_unit_conv']
    y_nac_std = hyper['model']['y_nac_std'] 
    y_nac_mean = hyper['model']['y_nac_mean']
        
    #Predcition 
    feat,feat_grad = mf.predict(x,batch_size = batch_size_predict)
    temp = ml.predict([feat,feat_grad],batch_size = batch_size_predict)    
    #Scaling
    temp = temp/y_nac_unit_conv* y_nac_std + y_nac_mean
    return temp





def predict_model_energy_gradient(ml,x,batch_size =32,scaler = {}):

    y_energy_std = scaler['energy_std']
    y_energy_mean = scaler['energy_mean']
    y_gradient_std = scaler['gradient_std']
    y_gradient_mean = scaler['gradient_mean']
    x_mean = scaler['x_mean']
    x_std = scaler['x_std']
    
    #Prediction
    x_res = (x -x_mean)/x_std
    temp_e,temp_g  = ml.predict(x_res,batch_size = batch_size)
    #Scaling
    temp_e = temp_e * y_energy_std + y_energy_mean 
    temp_g = temp_g  * y_gradient_std

    return temp_e,temp_g


def predict_model_nac(ml,x,batch_size=32,scaler = {}):
         
    y_nac_std = scaler['nac_std']
    y_nac_mean = scaler['nac_mean']
    x_mean = scaler['x_mean']
    x_std = scaler['x_std']
    
    
    #Predcition 
    x_res = x
    temp = ml.predict(x_res,batch_size = batch_size)    
    #Scaling
    temp = temp* y_nac_std + y_nac_mean
    return temp


@tf.function
def call_model_energy_gradient(m,x,scaler={}):

    #Note tf boradcasting should work the same here
    y_energy_std = tf.constant(scaler['energy_std'])
    y_energy_mean = tf.constant(scaler['energy_mean'])
    y_gradient_std = tf.constant(scaler['gradient_std'])
    y_gradient_mean = tf.constant(scaler['gradient_mean'])
    x_mean = tf.constant(scaler['x_mean'])
    x_std = tf.constant(scaler['x_std'])
    
    #Xscaling
    x_rescale = (x-x_mean) / (x_std)
      
    #Model prediction, feature normalization is within model
    temp_e,temp_g = m(x_rescale,training=False)

    #Scaling
    temp_e = temp_e * y_energy_std + y_energy_mean 
    temp_g = temp_g * y_gradient_std  + y_gradient_mean

    
    return temp_e,temp_g


@tf.function
def call_model_nac(m,x,scaler={}):    

    #Note tf boradcasting should work the same here
    y_nac_std = tf.constant(scaler['nac_std'])
    y_nac_mean = tf.constant(scaler['nac_mean'])
    x_mean = tf.constant(scaler['x_mean'])
    x_std = tf.constant(scaler['x_std'])
    
    #Xscaling
    #x_rescale = (x-x_mean) / (x_std)
    x_rescale = x #Not use x-sacling
    
    #Model prediction, feature normalization is within model
    temp = m(x_rescale,training=False)

    #Scaling
    temp = temp * y_nac_std + y_nac_mean

    
    return temp