"""
Tools for handling and storing data. For example to save to folder etc.
"""

import numpy as np
import pickle
from sklearn.utils import shuffle
import os



def index_make_random_shuffle(x):
    """
    Shuffle indexarray.

    Parameters
    ----------
    x : np.array
        Index to shuffle.

    Returns
    -------
    np.array
        Shuffled index.

    """
    return shuffle(x)


def datalist_make_random_shuffle(datalist):
    """
    Shuffle a list od data

    Parameters
    ----------
    datalist : list
        List of numpy arrays of same length (axis=0).

    Returns
    -------
    allind : np.array
        Index assignment of the shuffle.
    outlist : list
        List of the shuffled data.

    """
    datalen = len(datalist[0])
    for x in datalist:
        if(len(x) != datalen):
            print("Error: Data has inconsisten length")
      
    allind = np.arange(0,datalen)
    allind = shuffle(allind)
    
    outlist = []
    for x in datalist:
        outlist.append(x[allind])
    return allind, outlist
    

def model_save_data_to_folder(x,y,
               target_model,
               mod_dir,
               random_shuffle = False):
    """
    Save Data to model folder. Always dumps data_x and data_y as pickle.

    Parameters
    ----------
    x : np.array
        Coordinates as x-data.
    y : list,np.array
        A possible list of np.arrays for y-values. Energy, Gradients, NAC etc.
    target_model : str
        Name of the Model to save data for.
    mod_dir : str
        Path of model directory.
    random_shuffle : bool, optional
        Whether to shuffle data before save. The default is False.

    Returns
    -------
    None.

    """
    #Save data:
    if(random_shuffle == False):
        with open(os.path.join(mod_dir,'data_x'),'wb') as f: pickle.dump(x, f)
        with open(os.path.join(mod_dir,'data_y'),'wb') as f: pickle.dump(y, f)
    else:
        if(isinstance(y,list)):
            shuffle_list = [x] + y
        else:
            shuffle_list = [x] + [y]
        #Make random shuffle
        ind_shuffle, datalist  = datalist_make_random_shuffle(shuffle_list)
        x_out = datalist[0]
        if(len(datalist)>2):
            y_out = datalist[1:] 
        else:
            y_out = datalist[1] 
        np.save(os.path.join(mod_dir,'shuffle_index.npy'),ind_shuffle)
        with open(os.path.join(mod_dir,'data_x'),'wb') as f: pickle.dump(x_out, f)
        with open(os.path.join(mod_dir,'data_y'),'wb') as f: pickle.dump(y_out, f)    
        
        
  
def merge_data_in_chunks(data1,data2,split_size):
    """
    Merge data in chunks of split-size. Goal is to keep validation k-splits for fit.
    
    Idea: [a+a+a] + [b+b+b] = [(a+b)+(a+b)+(a+b)] and NOT [a+a+a+b+b+b].

    Parameters
    ----------
    data1 : np.array
        Data to merge.
    data2 : np.array
        Data to merge.
    split_size : float
        Relative size of junks 0 < split_size < 1.

    Returns
    -------
    np.array
        Merged data.

    """
    pacs1 = int(len(data1)*split_size)
    pacs2 = int(len(data2)*split_size)
    
    data1frac = [data1[i*pacs1:(i+1)*pacs1] for i in range(int(np.ceil(1/split_size)))]
    data2frac = [data2[i*pacs2:(i+1)*pacs2] for i in range(int(np.ceil(1/split_size)))]
    
    for i in range(len(data1frac)):
        data1frac[i] = np.concatenate([data1frac[i],data2frac[i]],axis=0)
        
    return np.concatenate(data1frac,axis=0)    
        


def split_validation_training_index(allind,splitsize,do_offset,offset_steps):
    """
    Make a train-validation split for indexarray. Validation set is taken from beginning with possible offset.

    Parameters
    ----------
    allind : np.array
        Indexlist for full dataset of same length.
    splitsize : int
        Total number of validation samples to take.
    do_offset : bool
        Whether to take validation set not from beginnig but with offset.
    offset_steps : int
        Number of validation sizes offseted from the beginning to start to take validation set.

    Returns
    -------
    i_train : np.array
        Training indices.
    i_val : np.arry
        Validation indices.

    """
    i = offset_steps
    lval = splitsize
    if(do_offset == False):
        i_val = allind[:lval]
        i_train = allind[lval:]
    else:
        i_val = allind[i*lval:(i+1)*lval]
        i_train = np.concatenate([allind[0:i*lval],allind[(i+1)*lval:]],axis=0)
    if(len(i_val) <= 0):
        print("Warning: #Validation data is 0, take 1 training sample instead")
        i_val = i_train[:1]
    
    return i_train,i_val



def index_data_in_y_dict(y,ind):
    """
    Index np.arrays as array[index] in the nested y_dict used in pes.

    Parameters
    ----------
    y : dict
        Dcitionary of y-values as y={'energy_gradients' : [np.array,np.array], 'NAC' : np.array}.
    ind : np.array
        Index array.

    Returns
    -------
    y_out : dict
        Same y_dict with its data as data[index].

    """
    y_out = {}
    for key, value in y.items():
        if(isinstance(value,list)):
            y_out[key] = [x[ind] for x in value]
        else:
            y_out[key] = value[ind]
    return y_out
    

          