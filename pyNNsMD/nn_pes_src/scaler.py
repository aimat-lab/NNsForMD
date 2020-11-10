"""
Standard scaler functions.

Note: Since energy and gradients are connected and must be physical meaningful, a completely freely scaling can not be applied.
For now it is just a dict with corresponding scalings.
"""

import numpy as np
import json
#import os

from pyNNsMD.nn_pes_src.scaling.scale_general import scale_x
from pyNNsMD.nn_pes_src.scaling.scale_mlp_nac import DEFAULT_STD_SCALER_NAC, rescale_nac
from pyNNsMD.nn_pes_src.scaling.scale_mlp_eg import DEFAULT_STD_SCALER_ENERGY_GRADS, rescale_eg



def _get_default_scaler_dict(model_type):
    """
    Get default values for scaling in and output for each model.

    Args:
        model_type (str): Model identifier.

    Returns:
        Dict: Scaling dictionary.

    """
    if(model_type == 'mlp_eg'):
        return DEFAULT_STD_SCALER_ENERGY_GRADS
    elif(model_type == 'mlp_nac'):
        return DEFAULT_STD_SCALER_NAC
    elif(model_type == 'mlp_nac2'):
        return DEFAULT_STD_SCALER_NAC
    elif(model_type == 'mlp_e'):
        return DEFAULT_STD_SCALER_ENERGY_GRADS
    else:
        print("Error: Unknown model type",model_type)
        raise TypeError(f"Error: Unknown model type for default scaling {model_type}")


def _scale_x(model_type,x,scaler = {'x_mean' : np.zeros((1,1,1)),'x_std' : np.ones((1,1,1))}):
    if(model_type == 'mlp_eg'):
        return scale_x(x,scaler)
    elif(model_type == 'mlp_nac'):
        return scale_x(x,scaler)
    elif(model_type == 'mlp_nac2'):
        return scale_x(x,scaler)
    elif(model_type == 'mlp_e'):
        return scale_x(x,scaler)
    else:
        print("Error: Unknown model type",model_type)
        raise TypeError(f"Error: Unknown model type for x-scaling {model_type}")



def _rescale_output(model_type,temp, scaler):
    if(model_type == 'mlp_eg'):
        return rescale_eg(temp,scaler)
    elif(model_type == 'mlp_nac'):
        return rescale_nac(temp,scaler)
    elif(model_type == 'mlp_nac2'):
        return rescale_nac(temp,scaler)
    elif(model_type == 'mlp_e'):
        return rescale_eg(temp,scaler)
    else:
        print("Error: Unknown model type",model_type)
        raise TypeError(f"Error: Unknown model type for rescaling {model_type}")




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
    Load std_scaler to directory.

    Args:
        filepath (str): Filepath.

    Returns:
        outdict (dict): Loaded std dict.

    """
    with open(filepath, 'r') as f:
        indict = json.load(f)
    outdict = {key: np.array(x) for key,x in indict.items()}
    return outdict








