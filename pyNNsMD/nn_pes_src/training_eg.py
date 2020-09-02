"""
The main training script for energy gradient model. Called with ArgumentParse.
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

parser = argparse.ArgumentParser(description='Train a energy-gradient model from data, parameters given in a folder')

parser.add_argument("-i", "--index", required=True, help="Index of the NN to train")
parser.add_argument("-f", "--filepath", required=True, help="Filepath to weights, hyperparameter, data etc. ")
parser.add_argument("-g", "--gpus",default=-1 ,required=True, help="Index of gpu to use")
parser.add_argument("-m", "--mode",default="training" ,required=True, help="Which mode to use train or retrain")
args = vars(parser.parse_args())
#args = {"filepath":"E:/Benutzer/Patrick/PostDoc/Projects ML/NeuralNet4/NNfit0/energy_gradient_0",'index' : 0,"gpus":0}


fstdout =  open(os.path.join(args['filepath'],"fitlog_"+str(args['index'])+".txt"), 'w')
sys.stderr = fstdout
sys.stdout = fstdout

print("Input argpars:",args)

from pyNNsMD.nn_pes_src.device import set_gpu

set_gpu([int(args['gpus'])])
print("Logic Devices:",tf.config.experimental.list_logical_devices('GPU'))

from pyNNsMD.nn_pes_src.callbacks import EarlyStopping,lr_lin_reduction,lr_exp_reduction,lr_step_reduction
from pyNNsMD.nn_pes_src.plot import plot_energy_gradient_fit_result
from pyNNsMD.nn_pes_src.models_feat import create_feature_models
from pyNNsMD.nn_pes_src.models_eg import create_model_energy_gradient_precomputed
from pyNNsMD.nn_pes_src.legacy import compute_feature_derivative
from pyNNsMD.nn_pes_src.hyper import _load_hyp
from pyNNsMD.nn_pes_src.data import split_validation_training_index


def train_model_energy_gradient(i = 0, outdir=None,  mode='training'): 
    """
    Train an energy plus gradient model. Uses precomputed feature and model representation.

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
        Validation error for (energy,gradient).

    """
    i = int(i)
    #Load everything from folder
    try:
        with open(os.path.join(outdir,'data_y'),'rb') as f: y = pickle.load(f)
        with open(os.path.join(outdir,'data_x'),'rb') as f: x = pickle.load(f)
        hyperall = _load_hyp(os.path.join(outdir,'hyper'+'_v%i'%i+".json"))
    except:
        print("Error: Can not load data and info for fit",outdir)
        return
    
    #Model
    hypermodel  = hyperall['model']
    y_energy_unit_conv = hypermodel['y_energy_unit_conv']
    y_gradient_unit_conv = hypermodel['y_gradient_unit_conv']
    y_energy_std = hypermodel['y_energy_std']
    y_energy_mean = hypermodel['y_energy_mean']
    #plots
    unit_label_energy = hyperall['plots']['unit_energy']
    unit_label_grad = hyperall['plots']['unit_gradient']
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
    val_disjoint = hyper['val_disjoint']
    val_split = hyper['val_split'] 
    reinit_weights = hyper['reinit_weights']

    #Data Check here:
    if(len(x.shape) != 3):
        raise ValueError("Input x-shape must be (batch,atoms,3)")
    else:
        print("Found x-shape of",x.shape)
    if(isinstance(y,list)==False):
        raise ValueError("Input y must be list of [energy,gradient]")
    if(len(y[0].shape) != 2):
        raise ValueError("Input energy-shape must be (batch,states)")
    else:
        print("Found energy-shape of",y[0].shape)
    if(len(y[1].shape) != 4):
        raise ValueError("Input gradient-shape must be (batch,states,atoms,3)")
    else:
        print("Found gradient-shape of",y[1].shape)

    #Fit stats dir
    dir_save = os.path.join(outdir,"fit_stats")
    os.makedirs(dir_save,exist_ok=True)    
    
    #scaling Changing coordinates  + some offset for derivative to be correct
    y1 = (y[0]-y_energy_mean)/y_energy_std*y_energy_unit_conv
    y2 = y[1]/y_energy_std*y_gradient_unit_conv
        
    #Features precompute layer
    temp_model_feat = create_feature_models(hypermodel)
    np_x, np_grad = temp_model_feat.predict_in_chunks(x,batch_size=batch_size)

    #Learning rate schedule        
    lr_sched = lr_lin_reduction(learning_rate_start,learning_rate_stop,epo,epomin)
    lr_exp = lr_exp_reduction(learning_rate_start,epomin,epostep,factor_lr)
    lr_step = lr_step_reduction(learning_rate_step,epoch_step_reduction)
    
    #cbks
    step_cbk = tf.keras.callbacks.LearningRateScheduler(lr_step)
    lr_cbk = tf.keras.callbacks.LearningRateScheduler(lr_sched)
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
                           minEpoch=epomin) 
    cbks = []
    if(use_early_callback == True):
        cbks.append(es_cbk)
    if(use_linear_callback == True):
        cbks.append(lr_cbk)
    if(use_exp_callback == True):
        cbks.append(exp_cbk)
    if(use_step_callback == True):
        cbks.append(step_cbk)
    
    #Index selection
    lval = int(len(x)*val_split)
    allind = np.arange(0,len(x))
    i_train,i_val = split_validation_training_index(allind,lval,val_disjoint,i)
    print("Train-Test split at Train:",len(i_train),"Test",len(i_val),"Total",len(x))
    
    xtrain = [np_x[i_train],np_grad[i_train]]
    ytrain = [y1[i_train],y2[i_train]]
    xval = [np_x[i_val],np_grad[i_val]]
    yval = [y1[i_val],y2[i_val]]
    
    #Do actual fitting
    temp_model = create_model_energy_gradient_precomputed(hypermodel)
    if(reinit_weights==False):
        try:
            temp_model.load_weights(os.path.join(outdir,"weights"+'_v%i'%i+'.h5'))
            print("Info: Load old weights at:",os.path.join(outdir,"weights"+'_v%i'%i+'.h5'))
        except:
            print("Error: Can't load old weights...")
    else:
        print("Info: Reinitializing weights.")
            
    temp_model.summary()
    
    hist = temp_model.fit(x=xtrain, y=ytrain, epochs=epo,batch_size=batch_size,callbacks=cbks,validation_freq=epostep,validation_data=(xval,yval),verbose=2)
    
    try:
        #Save weights
        temp_model.save_weights(os.path.join(outdir,"weights"+'_v%i'%i+'.h5'))
        outname = os.path.join(dir_save,"history_"+".json")           
        outhist = {a: np.array(b,dtype=np.float64).tolist() for a,b in hist.history.items()}
        with open(outname, 'w') as f:
            json.dump(outhist, f)
    except:
          print("Warning: Cant save weights or history")
    
    try:
        #Plot and Save
        yval_plot = [y[0][i_val] * y_energy_unit_conv , y[1][i_val] * y_gradient_unit_conv]
        ytrain_plot = [y[0][i_train] * y_energy_unit_conv , y[1][i_train] * y_gradient_unit_conv]
        # Convert back scaling, not units!!
        pval = temp_model.predict(xval)
        ptrain = temp_model.predict(xtrain)
        pval = [pval[0] * y_energy_std + y_energy_mean * y_energy_unit_conv,
                pval[1] * y_energy_std ]
        ptrain = [ptrain[0] * y_energy_std + y_energy_mean * y_energy_unit_conv,
                  ptrain[1] * y_energy_std]
    
        print("")
        print("Predicted Energy shape:",ptrain[0].shape)
        print("Predicted Gradient shape:",ptrain[1].shape)
        print("")
        print("Plot fit stats...")        
        
        #Plot
        plot_energy_gradient_fit_result(i,xval,xtrain,
                yval_plot,ytrain_plot,
                pval,ptrain,
                hist,
                epostep = epostep,
                dir_save= dir_save,
                unit_energy=unit_label_energy,
                unit_force=unit_label_grad)     
    except:
        print("Error: Could not plot fitting stats")
        
    #Val error
    error_val = None    
    try:
        testlossall_energy = hist.history['val_energy_mean_absolute_error']
        testlossall_force = hist.history['val_force_mean_absolute_error']
        trainlossall_energy = hist.history['energy_mean_absolute_error']
        trainlossall_force = hist.history['force_mean_absolute_error']     
        error_val = np.array([testlossall_energy[-1],testlossall_force[-1]])      
        error_train = np.array([trainlossall_energy[-1],trainlossall_force[-1]])
        #Havt to correct rescaling here 
        error_val[0] = error_val[0]* y_energy_std / y_energy_unit_conv
        error_val[1] = error_val[1]* y_energy_std / y_gradient_unit_conv
        error_train[0] = error_train[0]* y_energy_std / y_energy_unit_conv
        error_train[1] = error_train[1] *y_energy_std / y_gradient_unit_conv
    except:
        print("Error: Can not evaluate fiterror from history")
    
    try:
        #Safe fitting Error MAE
        # pval = temp_model.predict(x=xval)
        # ptrain = temp_model.predict(x=xtrain)
        # pval = [pval[0] / y_energy_unit_conv * y_energy_std + y_energy_mean,
        #         pval[1] / y_gradient_unit_conv * y_energy_std]
        # ptrain = [ptrain[0] / y_energy_unit_conv * y_energy_std + y_energy_mean,
        #           ptrain[1]/ y_gradient_unit_conv * y_energy_std]
        # error_val = [np.mean(np.abs(pval[0]-y[0][i_val])),np.mean(np.abs(pval[1]-y[1][i_val])) ]
        # error_train = [np.mean(np.abs(ptrain[0]-y[0][i_train])),np.mean(np.abs(ptrain[1]-y[1][i_train])) ]
        np.save(os.path.join(outdir,"fiterr_valid" +'_v%i'%i+ ".npy"),error_val)
        np.save(os.path.join(outdir,"fiterr_train" +'_v%i'%i+".npy"),error_train)
    except:
        print("Error: Can not save fiterror")
    
    return error_val



if __name__ == "__main__":
    print("Training Model: ", args['filepath'])
    print("Network instance: ", args['index'])
    out = train_model_energy_gradient(args['index'],args['filepath'],args['mode'])
    
fstdout.close()