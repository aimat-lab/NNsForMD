"""
Prepare training data for model mlp_nac.

@author: Patrick
"""
import numpy as np
import pickle
import os


from pyNNsMD.nn_pes_src.datasets.data_general import make_random_shuffle,merge_np_arrays_in_chunks,save_data_to_folder



def mlp_nac_save_data_to_folder(x,y,target_model,mod_dir,random_shuffle):
    """
    Save all training data for model mlp_nac to folder.

    Args:
        x (np.array): Coordinates as x-data.
        y (list): A possible list of np.arrays for y-values. Energy, Gradients, NAC etc.
        target_model (str): Name of the Model to save data for.
        mod_dir (str): Path of model directory.
        random_shuffle (bool, optional): Whether to shuffle data before save. The default is False.

    Returns:
        None.

    """
    save_data_to_folder(x,y,target_model,mod_dir,random_shuffle)


def mlp_nac_merge_data_in_chunks(mx1,my1,mx2,my2,val_split):
    """
    Merge Data in chunks.

    Args:
        mx1 (list,np.array): Coordinates as x-data.
        my1 (,np.array): np.array for y-values. NAC
        mx2 (list,np.array): Coordinates as x-data.
        my2 (np.array): np.array for y-values. NAC
        val_split (float, optional): Validation split. Defaults to 0.1.

    Raises:
        TypeError: Unkown model type.

    Returns:
        x: Merged x data. Depending on model.
        y: Merged y data. Depending on model.
            
    """
    x_merge = merge_np_arrays_in_chunks(mx1,mx2,val_split)
    y_merge = merge_np_arrays_in_chunks(my1,my2,val_split)
    return x_merge,y_merge

