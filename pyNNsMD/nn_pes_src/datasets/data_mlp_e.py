"""
Prepare Training data for mlp_e.

@author: Patrick
"""

import numpy as np
import pickle
import os
from sklearn.utils import shuffle


from pyNNsMD.nn_pes_src.datasets.data_general import make_random_shuffle,merge_np_arrays_in_chunks,save_data_to_folder


def mlp_e_save_data_to_folder(x,y,target_model,mod_dir,random_shuffle):
    """
    Save all training data for model mlp_eg to folder.

    Args:
        x (np.array): Coordinates as x-data.
        y (np.array): np.array for y-values. Energy.
        target_model (str): Name of the Model to save data for.
        mod_dir (str): Path of model directory.
        random_shuffle (bool, optional): Whether to shuffle data before save. The default is False.

    Returns:
        None.

    """
    save_data_to_folder(x,y,target_model,mod_dir,random_shuffle)
        



def mlp_e_merge_data_in_chunks(mx1,my1,mx2,my2,val_split):
    """
    Merge Data in chunks.

    Args:
        mx1 (list,np.array): Coordinates as x-data.
        my1 (list,np.array): A possible list of np.arrays for y-values. Energy, Gradients.
        mx2 (list,np.array): Coordinates as x-data.
        my2 (list,np.array): A list of np.arrays for y-values. Energy, Gradients.
        val_split (float, optional): Validation split. Defaults to 0.1.

    Returns:
        x: Merged x data. Depending on model.
        y: Merged y data. Depending on model.
            
    """
    x_merge = merge_np_arrays_in_chunks(mx1,mx2,val_split)
    y_merge = merge_np_arrays_in_chunks(my1,my2,val_split)    
    return x_merge,y_merge
