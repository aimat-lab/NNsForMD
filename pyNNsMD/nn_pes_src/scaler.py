"""
Standard scaler functions.

Note: Since energy and gradients are connected and must be physical meaningful, a completely freely scaling can not be applied.
For now it is just a dict with corresponding scalings.
"""

import numpy as np
import json
#import os



DEFAULT_STD_SCALER_ENERGY_GRADS = {'x_mean' : np.zeros((1,1,1)),
                                   'x_std' : np.ones((1,1,1)),
                                   'energy_mean' : np.zeros((1,1)),
                                   'energy_std' : np.ones((1,1)),
                                   'gradient_mean' : np.zeros((1,1,1,1)),
                                   'gradient_std' : np.ones((1,1,1,1))
                                   }

DEFAULT_STD_SCALER_NAC = {         'x_mean' : np.zeros((1,1,1)),
                                   'x_std' : np.ones((1,1,1)),
                                   'nac_mean' : np.zeros((1,1,1,1)),
                                   'nac_std' : np.ones((1,1,1,1))
                                   }


def save_std_scaler_dict(indict,filepath):
    """
    Save std_scaler to directory.

    Args:
        indict (dict): Dictionary to save.
        filepath (dict): Filepath.

    Returns:
        outdict (dict): Dict saved to file.

    """
    outdict = {key: x.tolist() for key,x in indict.items()}
    with open(filepath, 'w') as f:
        json.dump(outdict, f)
    return outdict


def load_std_scaler_dict(filepath):
    """
    Loadstd_scaler to directory.

    Args:
        filepath (str): Filepath.

    Returns:
        outdict (dict): Loaded std dict.

    """
    with open(filepath, 'r') as f:
        indict = json.load(f)
    outdict = {key: np.array(x) for key,x in indict.items()}
    return outdict


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


def rescale_eg(eng,grad, scaler = DEFAULT_STD_SCALER_ENERGY_GRADS ):
    """
    Rescale Energy and gradients.

    Args:
        eng (np.array): Energy.
        grad (np.array): Gradients.
        scaler (dict, optional): Scale to revert. The default is DEFAULT_STD_SCALER_ENERGY_GRADS.

    Returns:
        out_e (np.array): Rescaled energy.
        out_g (np.array): gradient.

    """
    y_energy_std = scaler['energy_std']
    y_energy_mean = scaler['energy_mean']
    y_gradient_std = scaler['gradient_std']
    #y_gradient_mean = scaler['gradient_mean']
    
    #Scaling
    out_e = eng * y_energy_std + y_energy_mean 
    out_g = grad  * y_gradient_std

    return out_e,out_g


def rescale_nac(nac, scaler = DEFAULT_STD_SCALER_NAC ):
    """
    Rescale NACs.

    Args:
        nac (np.array): NACs of shape (batch,states,atoms,3).
        scaler (dict, optional): Scale to revert. The default is DEFAULT_STD_SCALER_NAC.

    Returns:
        out_nac (np.array): DESCRIPTION.

    """
    y_nac_std = scaler['nac_std']
    y_nac_mean = scaler['nac_mean']
    
    #Scaling
    out_nac = nac * y_nac_std + y_nac_mean

    return out_nac