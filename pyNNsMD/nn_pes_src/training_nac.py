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
from pyNNsMD.nn_pes_src.models_nac import create_model_nac_precomputed,NACModel
#from pyNNsMD.nn_pes_src.legacy import compute_feature_derivative
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
    #plots
    unit_label_nac = hyperall['plots']['unit_nac']
    #Fit
    hyper = hyperall[mode]
    phase_less_loss= ['phase_less_loss']
    epo = hyper['epo']
    batch_size = hyper['batch_size']
    epostep = hyper['epostep']
    pre_epo = hyper['pre_epo']
    val_disjoint = hyper['val_disjoint']
    val_split = hyper['val_split'] 
    reinit_weights = hyper['reinit_weights']
    learning_rate = hyper['learning_rate']
    #step
    use_step_callback = hyper['step_callback']['use']
    epoch_step_reduction = hyper['step_callback']['epoch_step_reduction']
    learning_rate_step = hyper['step_callback']['learning_rate_step']
    #lin
    use_linear_callback = hyper['linear_callback']['use']
    learning_rate_start = hyper['linear_callback']['learning_rate_start']
    learning_rate_stop = hyper['linear_callback']['learning_rate_stop']
    epomin_lin = hyper['linear_callback']['epomin']
    #exp
    use_exp_callback = hyper['exp_callback']['use']    
    epomin_exp = hyper['exp_callback']['epomin']
    factor_lr_exp = hyper['exp_callback']['factor_lr']
    #early
    use_early_callback = hyper['early_callback']['use']
    epomin_early = hyper['early_callback']['epomin']
    factor_lr_early = hyper['early_callback']['factor_lr']
    patience =  hyper['early_callback']['patience']
    max_time = hyper['early_callback']['max_time']
    delta_loss = hyper['early_callback']['delta_loss']
    loss_monitor = hyper['early_callback']['loss_monitor']
    learning_rate_start_early = hyper['linear_callback']['learning_rate_start']
    learning_rate_stop_early = hyper['linear_callback']['learning_rate_stop']

    
    #Data Check here:
    if(len(x.shape) != 3):
        raise ValueError("Input x-shape must be (batch,atoms,3)")
    else:
        print("Found x-shape of",x.shape)
    if(len(y_in.shape) != 4):
        raise ValueError("Input nac-shape must be (batch,states,atoms,3)")
    else:
        print("Found nac-shape of",y_in.shape)
    
    #Set stat dir    
    dir_save = os.path.join(outdir,"fit_stats")
    os.makedirs(dir_save,exist_ok=True)

    #Learning rate schedule        
    lr_sched = lr_lin_reduction(learning_rate_start,learning_rate_stop,epo,epomin_lin)
    lr_exp = lr_exp_reduction(learning_rate_start,epomin_exp,epostep,factor_lr_exp)
    lr_step = lr_step_reduction(learning_rate_step,epoch_step_reduction)
    
    #cbks
    step_cbk = tf.keras.callbacks.LearningRateScheduler(lr_step)
    lr_cbk = tf.keras.callbacks.LearningRateScheduler(lr_sched)
    exp_cbk = tf.keras.callbacks.LearningRateScheduler(lr_exp)
    es_cbk = EarlyStopping(patience = patience,
                           minutes=max_time,
                           epochs=epo,
                           learning_rate=learning_rate_start_early,
                           min_delta=delta_loss,
                           epostep=epostep,
                           min_lr=learning_rate_stop_early,
                           monitor=loss_monitor,
                           factor=factor_lr_early,
                           minEpoch=epomin_early) 
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
    print("Info: Train-Test split at Train:",len(i_train),"Test",len(i_val),"Total",len(x))
    
    #Make all Models
    out_model = NACModel(hypermodel)
    temp_model_feat = create_feature_models(hypermodel)
    temp_model = create_model_nac_precomputed(hypermodel,learning_rate,phase_less_loss)
    
    npeps = np.finfo(float).eps
    if(reinit_weights==False):
        try:
            out_model.load_weights(os.path.join(outdir,"weights"+'_v%i'%i+'.h5'))
            print("Info: Load old weights at:",os.path.join(outdir,"weights"+'_v%i'%i+'.h5'))
            #Transfer weights
            print("Info: Transferring weights...")
            temp_model.get_layer('mlp').set_weights(out_model.get_layer('mlp').get_weights())
            temp_model.get_layer('virt').set_weights(out_model.get_layer('virt').get_weights())
            print("Info: Reading standardization...")
            feat_x_mean,feat_x_std = out_model.get_layer('feat_std').get_weights()
            x_mean,x_std = out_model.get_layer('scale_coord').get_weights()
            y_nac_mean, y_nac_std = out_model.get_layer('rev_std_nac').get_weights()
        except:
            print("Error: Can't load old weights...")
    else:
        print("Info: Keeping newly initialized weights.")
        print("Info: Calculating std-values.")
        yit = y_in[i_train]
        y_nac_std = np.std(yit,axis=(0,3),keepdims=True)
        y_nac_mean = np.zeros_like(y_nac_std)
        feat_x_mean = None
        feat_x_std = None
        x_mean,x_std = np.array(0.0),np.array(1.0)
        
    #print(hyper)    

    y = (y_in - y_nac_mean) / (y_nac_std + npeps)
    
    #Calculate features
    feat_x, feat_grad = temp_model_feat.predict_in_chunks(x,batch_size=batch_size)
    
    #Finding std.
    if(feat_x_mean is None):
        feat_x_mean = np.mean(feat_x[i_train],axis=0,keepdims=True)
    if(feat_x_std is None):
        feat_x_std = np.std(feat_x[i_train],axis=0,keepdims=True)
        
    xtrain = [feat_x[i_train],feat_grad[i_train]]
    ytrain = y[i_train]
    xval = [feat_x[i_val],feat_grad[i_val]]
    yval = y[i_val]
    
    #Actutal Fitting
    temp_model.get_layer('feat_std').set_weights([feat_x_mean,feat_x_std])   
    temp_model.metrics_y_nac_std = tf.constant(y_nac_std,dtype=tf.float32) # For metrics
    temp_model.summary()
    
    print("Info: All-data NAC std",np.std(y_in,axis=(0,3),keepdims=True)[0,:,:,0])
    print("Info: Using nac-std:", y_nac_std.shape, y_nac_std[0,:,:,0])
    print("Info: Using x-scale:" , x_std)
    print("Info: Using x-offset:" , x_mean)
    print("Info: Using feature-scale:" , feat_x_std)
    print("Info: Using feature-offset:" , feat_x_mean)
    
    print("")
    print("Start fit.")   
    
    #Prefit if necessary
    if(pre_epo>0):
        temp_model_prefit = create_model_nac_precomputed(hypermodel,learning_rate,False)
        temp_model_prefit.set_weights(temp_model.get_weights())
        temp_model_prefit.fit(x=xtrain, y=ytrain, epochs=pre_epo,batch_size=batch_size,verbose=2)
        temp_model.set_weights(temp_model_prefit.get_weights())
    
    hist = temp_model.fit(x=xtrain, y=ytrain, epochs=epo,batch_size=batch_size,callbacks=cbks,validation_freq=epostep,validation_data=(xval,yval),verbose=2)
    
    try:
        outname = os.path.join(dir_save,"history_"+".json")
        outhist = {a: np.array(b,dtype=np.float64).tolist() for a,b in hist.history.items()}
        with open(outname, 'w') as f:
            json.dump(outhist, f)
    except:
          print("Warning: Cant save history")
        
    try:
        #Save Weights
        out_model.get_layer('mlp').set_weights(temp_model.get_layer('mlp').get_weights())
        out_model.get_layer('virt').set_weights(temp_model.get_layer('virt').get_weights())
        out_model.get_layer('feat_std').set_weights([feat_x_mean,feat_x_std])
        out_model.get_layer('scale_coord').set_weights([x_mean,x_std])
        out_model.get_layer('rev_std_nac').set_weights([y_nac_mean, y_nac_std])
        out_model.save_weights(os.path.join(outdir,"weights"+'_v%i'%i+'.h5'))
    except:
        print("Warning: Cant save weights")
        
   
    try:
        #Plot stats
        yval_plot = y_in[i_val] 
        ytrain_plot  = y_in[i_train] 
        #Revert standard but keep unit conversion
        pval = temp_model.predict(xval)
        ptrain = temp_model.predict(xtrain)
        pval = pval * y_nac_std + y_nac_mean 
        ptrain = ptrain * y_nac_std + y_nac_mean 
        
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
        #Safe fitting Error MAE
        pval = temp_model.predict(xval)
        ptrain = temp_model.predict(xtrain)
        pval = pval * y_nac_std + y_nac_mean
        ptrain = ptrain  * y_nac_std + y_nac_mean
        ptrain2 = out_model.predict(x[i_train])
        print("MAE between precomputed and full keras model:")      
        print("NAC", np.mean(np.abs(ptrain-ptrain2))) 
        error_val = np.mean(np.abs(pval-y_in[i_val]))
        error_train = np.mean(np.abs(ptrain-y_in[i_train]))
        print("error_val:" ,error_val)
        print("error_train:",error_train )
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