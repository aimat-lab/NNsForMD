"""
Functions for Model prediction that handles scaling for precomputed models.
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from pyNNsMD.nn_pes_src.hyper import DEFAULT_HYPER_PARAM_ENERGY_GRADS as hyper_predict_model_energy_gradient
from pyNNsMD.nn_pes_src.hyper import DEFAULT_HYPER_PARAM_NAC as hyper_predict_model_nac
#from nn_pes_legacy import compute_feature_derivative



def predict_model_energy_gradient_precomputed(ml,mf,x,hyper=hyper_predict_model_energy_gradient):
    """
    Predict energy plus gradient from separate FeatureModel and EnergyModelPrecomputed. Scales to original y.

    Parameters
    ----------
    ml : tf.keras.model
        Model that accepts features and returns energy.
    mf : tf.keras.model
        Model for features.
    x : np.array
        Coordinates in shape (batch,Atoms,3).
    hyper : dict, optional
        Hyperparameter dictionary. The default is hyper_predict_model_energy_gradient.

    Returns
    -------
    temp_e : np.array
        Calculated and scaled energy.
    temp_g : np.arry
        Calculated and scaled gradient.

    """
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


def predict_model_energy_gradient(ml,x,batch_size_predict =32):
    """
    Predict energy plus gradient from EnergyModel. Scales to original y.

    Parameters
    ----------
    ml : tf.keras.model
        Model for energy.
    x : np.array
        Coordinates in shape (batch,Atoms,3).
    batch_size_predict : int, optional
        The batch size used in prediction.

    Returns
    -------
    temp_e : np.array
        Calculated and rescaled energy.
    temp_g : np.arry
        Calculated and rescaled gradient.

    """
    #Prediction
    temp_e,temp_g  = ml.predict(x,batch_size = batch_size_predict)
    #Scaling
    #temp_e = temp_e / y_energy_unit_conv * y_energy_std + y_energy_mean #/27.21138624598853-156.22214381375588
    #temp_g = temp_g / y_gradient_unit_conv * y_energy_std

    return temp_e,temp_g


def predict_model_nac_precomputed(ml,mf,x,hyper=hyper_predict_model_nac):
    """
    Predict NAC from separate FeatureModel and NACModelPrecomputed. Scales to original y.

    Parameters
    ----------
    ml : tf.keras.model
        Model that accepts features and returns NAC virutal potential.
    mf : tf.keras.model
        Model for features.
    x : np.array
        Coordinates in shape (batch,Atoms,3).
    hyper : dict, optional
        Hyperparameter dictionary. The default is hyper_predict_model_nac.

    Returns
    -------
    temp : np.array
        Calculated and rescaled NAC.

    """    
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


def predict_model_nac(ml,x,batch_size_predict=32):
    """
    Predict NAC. Scales to original y.

    Parameters
    ----------
    ml : tf.keras.model
        Model for NAC virutal potential.
    x : np.array
        Coordinates in shape (batch,Atoms,3).
    batch_size_predict : int, optional
        The batch size used in prediction.

    Returns
    -------
    temp : np.array
        Calculated and rescaled NAC.

    """           
    #feat,feat_grad = compute_feature_derivative(x,hyper[0])    
    #Predcition 
    temp = ml.predict(x,batch_size = batch_size_predict)    
    #Scaling
    #temp = temp/y_nac_unit_conv* y_nac_std + y_nac_mean
    return temp


@tf.function
def call_model_energy_gradient_precomputed(ml,mf,x,hyper=hyper_predict_model_energy_gradient):
    """
    Call energy plus gradient from separate FeatureModel and EnergyModelPrecomputed. Scales to original y.

    Parameters
    ----------
    ml : tf.keras.model
        Model for energy.
    mf : tf.keras.model
        Model for features.
    x : tf.tensor
        Coordinates in shape (batch,Atoms,3).
    hyper : dict, optional
        Hyperparameter dictionary. The default is hyper_predict_model_energy_gradient.

    Returns
    -------
    temp_e : tf.tensor
        Calculated and rescaled energy.
    temp_g : tf.tensor
        Calculated and rescaled gradient.

    """
    y_energy_unit_conv = hyper['model']['y_energy_unit_conv']
    y_gradient_unit_conv = hyper['model']['y_gradient_unit_conv']
    y_energy_std = hyper['model']['y_energy_std']
    y_energy_mean = hyper['model']['y_energy_mean']
    
    #feat,feat_grad = compute_feature_derivative(x,hyper)    
      
    #Prediction
    
    with tf.GradientTape() as tape2:
        tape2.watch(x)
        feat = mf(x,training=False)
        temp_e,_ = ml(feat,training=False) 
    temp_g = tape2.batch_jacobian(temp_e, x)
    
    
    #Scaling
    temp_e = temp_e / y_energy_unit_conv * y_energy_std + y_energy_mean #/27.21138624598853-156.22214381375588
    temp_g = temp_g / y_gradient_unit_conv * y_energy_std

    
    return temp_e,temp_g


@tf.function
def call_model_nac_precomputed(ml,mf,x,hyper=hyper_predict_model_nac):    
    """
    Call NAC from separate FeatureModel and NACModelPrecomputed. Scales to original y.

    Parameters
    ----------
    ml : tf.keras.model
        Model for NAC virutal potential.
    mf : tf.keras.model
        Model for features.
    x : tf.tensor
        Coordinates in shape (batch,Atoms,3).
    hyper : dict, optional
        Hyperparameter dictionary. The default is hyper_predict_model_nac.

    Returns
    -------
    temp : tf.tensor
        Calculated and rescaled NAC.

    """  
    y_nac_unit_conv = hyper['model']['y_nac_unit_conv']
    y_nac_std = hyper['model']['y_nac_std'] 
    y_nac_mean = hyper['model']['y_nac_mean']
    y_states = hyper['model']['states']
    y_atoms = hyper['model']['atoms']
    
    #feat,feat_grad = compute_feature_derivative(x,hyper[0])
    with tf.GradientTape() as tape2:
        tape2.watch(x)
        feat = mf(x,training=False)
        atpot = ml(feat,training=False) 
        atpot = tf.reshape(atpot, (tf.shape(atpot)[0],y_states,y_atoms))
    grad = tape2.batch_jacobian(atpot, x)   
    temp = ks.backend.concatenate([ks.backend.expand_dims(grad[:,:,i,i,:],axis=2) for i in range(y_atoms)],axis=2)


    #Scaling
    temp = temp/y_nac_unit_conv* y_nac_std + y_nac_mean

    
    return temp
    