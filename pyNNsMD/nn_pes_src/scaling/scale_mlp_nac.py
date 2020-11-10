"""
Scaling of in and output.

@author: Patrick
"""

import numpy as np

DEFAULT_STD_SCALER_NAC = {         'x_mean' : np.zeros((1,1,1)),
                                   'x_std' : np.ones((1,1,1)),
                                   'nac_mean' : np.zeros((1,1,1,1)),
                                   'nac_std' : np.ones((1,1,1,1))
                                   }



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

