"""
Scaling of in and output

@author: Patrick
"""

import numpy as np

DEFAULT_STD_SCALER_ENERGY_GRADS = {'x_mean' : np.zeros((1,1,1)),
                                   'x_std' : np.ones((1,1,1)),
                                   'energy_mean' : np.zeros((1,1)),
                                   'energy_std' : np.ones((1,1)),
                                   'gradient_mean' : np.zeros((1,1,1,1)),
                                   'gradient_std' : np.ones((1,1,1,1))
                                   }


def rescale_eg(output, scaler = DEFAULT_STD_SCALER_ENERGY_GRADS ):
    """
    Rescale Energy and gradients.

    Args:
        output (np.array): [Energy,Gradients]
        scaler (dict, optional): Scale to revert. The default is DEFAULT_STD_SCALER_ENERGY_GRADS.

    Returns:
        out_e (np.array): Rescaled energy.
        out_g (np.array): gradient.

    """
    eng = output[0]
    grad = output[1]
    y_energy_std = scaler['energy_std']
    y_energy_mean = scaler['energy_mean']
    y_gradient_std = scaler['gradient_std']
    #y_gradient_mean = scaler['gradient_mean']
    
    #Scaling
    out_e = eng * y_energy_std + y_energy_mean 
    out_g = grad  * y_gradient_std

    return out_e,out_g

