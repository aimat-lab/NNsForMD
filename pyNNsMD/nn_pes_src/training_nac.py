"""
The main training script for NAC model. Called with ArgumentParse.
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
#from sklearn.utils import shuffle
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import pickle
import sys

import argparse

parser = argparse.ArgumentParser(description='Train a nac model from data, parameters given in a folder')

parser.add_argument("-i", "--index", required=True, help="Index of the NN to train")
parser.add_argument("-f", "--filepath", required=True, help="Filepath to weights, hyperparameter, data etc. ")
parser.add_argument("-g", "--gpus",default=-1 ,required=True, help="Index of gpu to use")
parser.add_argument("-m", "--mode",default="training" ,required=True, help="Which mode to use train or retrain")
args = vars(parser.parse_args())
#args = {"filepath":"E:/Benutzer/Patrick/PostDoc/Projects ML/NeuralNet4/NNfit0/nac_0",'index' : 0,"gpus":0}


fstdout =  open(os.path.join(args['filepath'],"fitlog_"+str(args['index'])+".txt"), 'w')
sys.stderr = fstdout
sys.stdout = fstdout
    
print("Input argpars:",args)

from pyNNsMD.nn_pes_src.device import set_gpu

set_gpu([int(args['gpus'])])
print("Logic Devices:",tf.config.experimental.list_logical_devices('GPU'))


from pyNNsMD.nn_pes_src.callbacks import EarlyStopping,lr_lin_reduction,lr_exp_reduction,lr_step_reduction
from pyNNsMD.nn_pes_src.plot import plot_nac_fit_result
from pyNNsMD.nn_pes_src.models_feat import create_feature_models
from pyNNsMD.nn_pes_src.models_nac import create_model_nac_precomputed
from pyNNsMD.nn_pes_src.legacy import compute_feature_derivative
from pyNNsMD.nn_pes_src.hyper import _load_hyp
from pyNNsMD.nn_pes_src.data import split_validation_training_index


def train_model_nac(i=0, outdir=None, mode = 'training'):
    """
    Train NAC model. Uses precomputed feature and model representation.

    Parameters
    ----------
    i : int, optional
        Model index. The default is 0.
    outdir : str, optional
        Direcotry for fit output. The default is None.
    mode : str, optional
        Fitmode to take from hyperparameters. The default is 'training'.

    Raises
    ------
    ValueError
        Wrong input shape.

    Returns
    -------
    error_val : list
        Validation error for NAC.

    """
    i = int(i)
    #Load everything from folder
    try:
        with open(os.path.join(outdir,'data_y'),'rb') as f: y_in = pickle.load(f)
        with open(os.path.join(outdir,'data_x'),'rb') as f: x = pickle.load(f)
        hyperall = _load_hyp(os.path.join(outdir,'hyper'+'_v%i'%i+".json"))
    except:
        print("Error: Can not load data and info for fit",outdir)
    
    #Model
    hypermodel = hyperall['model']
    y_nac_unit_conv = hypermodel['y_nac_unit_conv']
    y_nac_std = hypermodel['y_nac_std'] 
    y_nac_mean = hypermodel['y_nac_mean']
    #plots
    unit_label_nac = hyperall['plots']['unit_nac']
    #Fit
    hyper = hyperall[mode]
    epo = hyper['epo']
    batch_size = hyper['batch_size']
    epostep = hyper['epostep']
    learning_rate_start = hyper['learning_rate_start']
    learning_rate_stop = hyper['learning_rate_stop']
    use_early_callback = hyper['use_early_callback'] 
    use_linear_callback = hyper['use_linear_callback']
    use_exp_callback = hyper['use_exp_callback']     
    use_step_callback = hyper['use_step_callback'] 
    epomin = hyper['epomin']
    factor_lr = hyper['factor_lr']
    learning_rate_step = hyper['learning_rate_step']
    epoch_step_reduction = hyper['epoch_step_reduction']
    patience =  hyper['patience']
    max_time = hyper['max_time']
    delta_loss = hyper['delta_loss']
    loss_monitor = hyper['loss_monitor']
    pre_epo = hyper['pre_epo']
    val_disjoint = hyper['val_disjoint']
    val_split = hyper['val_split'] 
    reinit_weights = hyper['reinit_weights']
    
    #Data Check here:
    if(len(x.shape) != 3):
        raise ValueError("Input x-shape must be (batch,atoms,3)")
    else:
        print("Found x-shape of",x.shape)
    if(len(y_in.shape) != 4):
        raise ValueError("Input nac-shape must be (batch,states,atoms,3)")
    else:
        print("Found nac-shape of",y_in.shape)

    #print(hyper)    
    y = (y_in - y_nac_mean) / y_nac_std * y_nac_unit_conv
    
    #Set stat dir    
    dir_save = os.path.join(outdir,"fit_stats")
    os.makedirs(dir_save,exist_ok=True)
    
    #Features precompute layer
    temp_model_feat = create_feature_models(hypermodel)
    np_x, np_grad = temp_model_feat.predict_in_chunks(x,batch_size=batch_size)


    #Learning rate schedule        
    lr_sched = lr_lin_reduction(learning_rate_start,learning_rate_stop,epo,epomin)
    lr_exp = lr_exp_reduction(learning_rate_start,epomin,epostep,factor_lr)
    lr_step = lr_step_reduction(learning_rate_step,epoch_step_reduction)
    #cbks
    lr_cbk = tf.keras.callbacks.LearningRateScheduler(lr_sched) 
    step_cbk = tf.keras.callbacks.LearningRateScheduler(lr_step)
    exp_cbk = tf.keras.callbacks.LearningRateScheduler(lr_exp)
    es_cbk = EarlyStopping(patience = patience,
                           minutes=max_time,
                           epochs=epo,
                           learning_rate=learning_rate_start,
                           min_delta=delta_loss,
                           epostep=epostep,
                           min_lr=learning_rate_stop,
                           monitor=loss_monitor,
                           factor=factor_lr,
                           minEpoch=epomin ) 
    cbks = []
    if(use_early_callback == True):
        cbks.append(es_cbk)
    if(use_linear_callback == True):
        cbks.append(lr_cbk)
    if(use_exp_callback == True):
        cbks.append(exp_cbk)
    if(use_step_callback == True):
        cbks.append(step_cbk)
    
    #Data selection
    lval = int(len(x)*val_split)
    allind = np.arange(0,len(x))
    i_train,i_val = split_validation_training_index(allind,lval,val_disjoint,i)
    print("Train-Test split at Train:",len(i_train),"Test",len(i_val),"Total",len(x))
    
    xtrain = [np_x[i_train],np_grad[i_train]]
    ytrain = y[i_train]
    xval = [np_x[i_val],np_grad[i_val]]
    yval = y[i_val]
    

    #Actutal Fitting
    temp_model = create_model_nac_precomputed(hypermodel )
    if(reinit_weights==False):
        try:
            temp_model.load_weights(os.path.join(outdir,"weights"+'_v%i'%i+'.h5'))
            print("Info: Load old weights at:",os.path.join(outdir,"weights"+'_v%i'%i+'.h5'))
        except:
            print("Error: Can't load old weights...")
    temp_model.summary()
    #Prefit if necessary
    if(pre_epo>0):
        temp_model_prefit = create_model_nac_precomputed(hypermodel,True)
        temp_model_prefit.set_weights(temp_model.get_weights())
        temp_model_prefit.fit(x=xtrain, y=ytrain, epochs=pre_epo,batch_size=batch_size,verbose=2)
        temp_model.set_weights(temp_model_prefit.get_weights())
        
    hist = temp_model.fit(x=xtrain, y=ytrain, epochs=epo,batch_size=batch_size,callbacks=cbks,validation_freq=epostep,validation_data=(xval,yval),verbose=2)
            
    try:
        #Save Weights
        temp_model.save_weights(os.path.join(outdir,'weights'+'_v%i'%i+'.h5'))
        outname = os.path.join(dir_save,"history_"+".json")
        outhist = {a: np.array(b,dtype=np.float64).tolist() for a,b in hist.history.items()}
        with open(outname, 'w') as f:
            json.dump(outhist, f)
    except:
        print("Warning: Cant save weights or history")
        
   
    try:
        #Plot stats
        yval_plot = y_in[i_val] * y_nac_unit_conv
        ytrain_plot  = y_in[i_train] * y_nac_unit_conv
        #Revert standard but keep unit conversion
        pval = temp_model.predict(x=xval)
        ptrain = temp_model.predict(x=xtrain)
        pval = pval * y_nac_std + y_nac_mean * y_nac_unit_conv
        ptrain = ptrain * y_nac_std + y_nac_mean * y_nac_unit_conv
        
        print("")
        print("Predicted NAC shape:",ptrain.shape)
        print("")
        print("Plot fit stats...")        
        
        plot_nac_fit_result(i,xval,xtrain,
                            yval_plot,ytrain_plot,
                            pval,ptrain,
                            hist,
                            epostep = epostep,
                            dir_save= dir_save,
                            unit_nac=unit_label_nac)   
    except:
        print("Warning: Could not plot fitting stats")
    
    #error out
    error_val = None
    try:
        testlossall_nac = hist.history['val_mean_absolute_error']
        trainlossall_nac = hist.history['mean_absolute_error']
        error_val = testlossall_nac[-1]
        error_train = trainlossall_nac[-1]
        #Have to do sacline here too
        error_val *= y_nac_std/y_nac_unit_conv
        error_train *= y_nac_std/y_nac_unit_conv
    except:
        print("Error: Can not evaluate fiterror from history")
        

    try:
        #Safe fitting Error MAE
        # pval = temp_model.predict(x=xval)
        # ptrain = temp_model.predict(x=xtrain)
        # pval = pval /y_nac_unit_conv* y_nac_std + y_nac_mean
        # ptrain = ptrain /y_nac_unit_conv * y_nac_std + y_nac_mean
        # error_val = np.mean(np.abs(pval-y_in[i_val]))
        # error_train = np.mean(np.abs(pval-y_in[i_train]))
        np.save(os.path.join(outdir,"fiterr_valid" +'_v%i'%i+ ".npy"),error_val)
        np.save(os.path.join(outdir,"fiterr_train" +'_v%i'%i+".npy"),error_train)
    except:
        print("Error: Can not save fiterror")
        
    return error_val
        



if __name__ == "__main__":

    print("Training Model: ", args['filepath'])
    print("Network instance: ", args['index'])
    train_model_nac(args['index'],args['filepath'],args['mode'])
    
fstdout.close()