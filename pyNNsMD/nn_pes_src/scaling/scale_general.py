"""
Created on Tue Nov 10 11:47:49 2020

@author: Patrick
"""
import numpy as np

def scale_x(x,scaler = {'x_mean' : np.zeros((1,1,1)),'x_std' : np.ones((1,1,1))}):
    """
    Scale coordinates.

    Args:
        x (np.array): Coordinates.
        scaler (dict, optional): X-scale to apply. The default is {'x_mean' : np.zeros((1,1,1)),'x_std' : np.ones((1,1,1))}.

    Returns:
        x_res (np.array): Rescaled coordinates.

    """
    x_mean = scaler['x_mean']
    x_std = scaler['x_std']
    
    #Prediction
    x_res = (x -x_mean)/x_std 
    return x_res


def scale_feature(feats,hyper):
    """
    Scale features.
    
    This rquires knowledge on how the featue vector is composed. 
    Must be changed with the feautre description.

    Args:
        feats (np.array): DESCRIPTION.
        hyp (dict): DESCRIPTION.

    Returns:
        out_mean (np.array): DESCRIPTION.
        out_scale (np.array): DESCRIPTION.

    """
    indim = int( hyper['atoms'])
    use_invdist = hyper['invd_index'] != []
    use_bond_angles = hyper['angle_index'] != []
    angle_index = hyper['angle_index'] 
    use_dihyd_angles = hyper['dihyd_index'] != []
    dihyd_index = hyper['dihyd_index']

    in_model_dim = 0
    out_scale = []
    out_mean = []
    if(use_invdist==True):
        in_model_dim += int(indim*(indim-1)/2)
        invd_mean = np.mean(feats[:,0:in_model_dim])
        invd_std  = np.std(feats[:,0:in_model_dim])
        out_scale.append(np.tile(np.expand_dims(invd_std,axis=-1),(1,int(indim*(indim-1)/2))))
        out_mean.append(np.tile(np.expand_dims(invd_mean,axis=-1),(1,int(indim*(indim-1)/2))))
    if(use_bond_angles == True):
        in_model_dim += len(angle_index) 
    if(use_dihyd_angles == True):
        in_model_dim += len(dihyd_index) 
    
    
    out_scale = np.concatenate(out_scale,axis=-1)
    out_mean = np.concatenate(out_mean,axis=-1)
    return out_mean,out_scale