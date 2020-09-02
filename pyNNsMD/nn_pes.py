"""
Main class for neural network (NN) container to provide multiple NN instances to
enable uncertainty estimate as well as training and prediction of tf.keras models
for energy plus gradient and non-adiabatic couplings (NAC). The python class is supposed to
allow parallel training and further operations like resampling and hyper optimization. 
"""

import logging
import os
import pickle
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import json


from pyNNsMD.nn_pes_src.models_feat import create_feature_models
from pyNNsMD.nn_pes_src.models_nac import create_model_nac_precomputed,NACModel
from pyNNsMD.nn_pes_src.models_eg import create_model_energy_gradient_precomputed,EnergyModel
from pyNNsMD.nn_pes_src.fit import fit_model_energy_gradient,fit_model_nac,fit_model_energy_gradient_async,fit_model_nac_async
from pyNNsMD.nn_pes_src.predict import call_model_energy_gradient_precomputed,call_model_nac_precomputed
from pyNNsMD.nn_pes_src.predict import predict_model_energy_gradient_precomputed,predict_model_nac_precomputed
from pyNNsMD.nn_pes_src.predict import predict_model_energy_gradient,predict_model_nac
from pyNNsMD.nn_pes_src.hyper import DEFAULT_HYPER_PARAM_ENERGY_GRADS,DEFAULT_HYPER_PARAM_NAC
from pyNNsMD.nn_pes_src.hyper import _save_hyp,_load_hyp
from pyNNsMD.nn_pes_src.data import model_save_data_to_folder,datalist_make_random_shuffle,merge_data_in_chunks,index_data_in_y_dict,index_make_random_shuffle
from pyNNsMD.nn_pes_src.oracle import find_samples_with_max_error
from pyNNsMD.nn_pes_src.plot import plot_resampling_gradient,plot_resampling_nac


class NeuralNetPes:
    """ 
    Main class NeuralNetPes(directory) that keeps multiple keras models and
    manages training and prediction. The individual model types are further 
    stored to file in the directory specified in __init__(directory). 
    Each model you create within a NeuralNetPes is referenced by a dictionary.
    The information like predictions and hyperparameters are also passed in form of python dictionaries. 
    See the default parameters in nn_pes_src.hyper for the scope of all parameters and their explanation. 
    
    Example
    -------
    nn = NeuralNetPes("mol2")
    hyper_eg = {'general' : {'model_type':'energy_gradient'} , 'model' : {'atoms':12}} 
    hyper_nac = {'general': {'model_type':'nac'} , 'model': {'atoms':12}}
    nn.create({'mol2_energy': hyper_eg , 'mol2_nac' : hyper_nac})
    
    """
    def __init__(self,directory: str,mult_nn = 2):
        """
        Initilialize empty NeuralNetPes instance.

        Parameters
        ----------
        directory : str
            Directory where models, hyperparameter, logs and fitresults are stored.
        mult_nn : int, optional
            Number of NN instances to create for error estimate. The default is 2.

        Returns
        -------
        NueralNetPes instance.

        """
        self._models_implemented = ['energy_gradient', 'nac']

        self.logger = logging.getLogger(type(self).__name__)
        self.logger.info("Operating System: ",sys.platform)
        self.logger.info("Tested for tf-gpu= 2.3 This tf version: ",tf.__version__)        
        self.logger.debug("Models implemented:" , self._models_implemented)
        
        # Private memebers
        self._models = {}
        self._models_features = {} # will be removed
        self._models_precomputed = {} # Will be removed
        self._models_hyper = {}
        
        self._directory = directory
        self._addNN = mult_nn
        
        self._last_shuffle = None
        
    
    def _merge_hyper(self,dictold, dictnew,exclude_category = []):
        temp = {}
        temp.update(dictold)
        for hkey in dictnew.keys():
            if(hkey not in temp):
                print("Warning: Unknown category:", hkey)
                temp[hkey] = {}
            if(hkey in exclude_category):
                print("Error: Can not update specific category",hkey)
            else:
                for hhkey in dictnew[hkey].keys():
                    if(hhkey not in temp[hkey]):
                        print("Warning: Unknown key:", hhkey , "in", hkey)
                    temp[hkey][hhkey] = dictnew[hkey][hhkey]
        return temp

    
    def _create_models(self,key,h_dict):
        #Check if hpyer is a list of dict or a single dict
        if(isinstance(h_dict, dict)):
            model_type = h_dict['general']['model_type']
            #Make a list with the same hyp
            hyp = [h_dict for x in range(self._addNN)]
        elif(isinstance(h_dict, list)):
            if(len(h_dict) != self._addNN):
                self.logger.error(f"Error in hyp for number NNs for {key}")
            model_type = h_dict[0]['general']['model_type']
            for x in h_dict:
                if(x['general']['model_type'] != model_type):
                    self.logger.error(f"Inconsistent Input for {key}")
            #Accept list of hyperdicts
            hyp = h_dict
        else:
            self.logger.error(f"Unknwon Input tpye of hyper for {key}")
            raise TypeError(f"Unknwon Input tpye of hyper for {key}")
        
        # Create the correct model with hyper
        models = {}
        models[key] = []
        models_prec = {}
        models_prec[key] = []
        features = {}
        features[key] = []
        if(model_type == 'energy_gradient'):
            for i in range(self._addNN):
                #Fill missing hyper
                temp = self._merge_hyper(DEFAULT_HYPER_PARAM_ENERGY_GRADS,hyp[i])              
                hyp[i] = temp
                models[key].append(EnergyModel(hyper=temp['model']))  
                models_prec[key].append(create_model_energy_gradient_precomputed(hyper=temp['model']))  
                features[key].append(create_feature_models(hyper=temp['model']))  
        elif(model_type == 'nac'):
            for i in range(self._addNN):
                #Fill missing hyper
                temp = self._merge_hyper(DEFAULT_HYPER_PARAM_NAC,hyp[i])
                hyp[i] = temp
                models[key].append(NACModel(hyper=temp['model'])) 
                models_prec[key].append(create_model_nac_precomputed(hyper=temp['model'])) 
                features[key].append(create_feature_models(hyper=temp['model']))  
        else:
            self.logger.error(f"Unknwon Model type in hyper dict for {model_type}")
            
        return models,models_prec,hyp,features
    
    
    def create(self,hyp_dict):
        """
        Initialize and build a model. Missing hyperparameter are filled from default.

        Parameters
        ----------
        hyp_dict : dict
            Dictionary with hyper-parameter. {'model' : hyper_dict, 'model2' : hyper_dict2 ....}
            Missing hyperparameter in hyper_dict are filled up from default, see nn_utils for complete set.

        Returns
        -------
        list
            created models.

        """
        for key, value in hyp_dict.items():
            mod,modprec,hyp,feat = self._create_models(key,value)
            self._models.update(mod)
            self._models_hyper[key] = hyp
            self._models_features.update(feat)
            self._models_precomputed.update(modprec)
        
        return self._models
   
    
    def update(self,hyp_dict):
        """
        Update hyper parameters if possible.

        Parameters
        ----------
        hyp_dict : dict
            Dictionary with hyper-parameter. {'model' : hyper_dict, 'model2' : hyper_dict2 , ...} to update.
            Note: model parameters will not be updated.

        Returns
        -------
        None.

        """
        for key, value in hyp_dict.items():
            if(key in self._models_hyper.keys()):
                if(isinstance(value, dict)):
                    value = [value for i in range(self._addNN)]
                elif(len(value) != self._addNN):
                    self.logger.error(f"Error in hyp for number NNs for {key}")
                else:
                    self.logger.error(f"Unknwon Input tpye of hyper for {key}")
                    raise TypeError(f"Unknwon Input tpye of hyper for {key}")
                for i in range(self._addNN):
                    self._models_hyper[key][i] = self._merge_hyper(self._models_hyper[key][i],
                                                                   value[i],
                                                                   exclude_category=['model'])
    
    
    def _save(self,directory,name):
        # Check if model name can be saved
        if(name not in self._models_precomputed):
            raise TypeError("Cannot save model before init.")
            
        # Folder to store model in
        filename = os.path.abspath(os.path.join(directory,name))
        os.makedirs(filename,exist_ok=True)
        
        #Store weights and hyper
        for i,x in enumerate(self._models[name]):
            x.save_weights(os.path.join(filename,'weights'+'_v%i'%i+'.h5'))
        for i,x in enumerate(self._models_hyper[name]):
            _save_hyp(x,os.path.join(filename,'hyper'+'_v%i'%i+".json"))
        
        return filename
        
    
    def save(self,model_name=None):
        """
        Save a model weights and hyperparameter into class folder with a certain name.
        The model itself is not saved, use export to store the model itself.
        Thes same holds for load. Here the model is recreated from hyperparameters and weights.

        Parameters
        ----------
        model_name : str, optional
            Name of the Model to save. The default is None, which means save all

        Returns
        -------
        out_dirs : list
            Saved directories.

        """
        dirname = self._directory
        directory = os.path.abspath(dirname)
        os.makedirs(directory, exist_ok=True)
        
        # Safe model_name 
        out_dirs = []
        if(isinstance(model_name, str)):
            out_dirs.append(self._save(directory,model_name))
            
        elif(isinstance(model_name, list)):
            for name in model_name:
                out_dirs.append(self._save(directory,name))
        
        #Default just save all
        else:    
            for name in self._models.keys():
                out_dirs.append(self._save(directory,name))
        
        return out_dirs
   
    
    def _load(self, folder, model_name):
        fname = os.path.join(folder,model_name)
        # Check if folder exists
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Cannot find model directory {fname}")
        
        #Load  Hyperparameters
        hyp = []
        for i in range(self._addNN):
            hyp.append(_load_hyp(os.path.join(fname,'hyper'+'_v%i'%i+".json")))
        
        #Create Model
        self.create({model_name : hyp})
        
        #Load weights
        for i in range(self._addNN):
            self._models_precomputed[model_name][i].load_weights(os.path.join(fname,'weights'+'_v%i'%i+'.h5'))
            self._models[model_name][i].load_weights(os.path.join(fname,'weights'+'_v%i'%i+'.h5'))
            self.logger.info("Imported weights for: %s"%(model_name+'_v%i'%i))
                     
    
    def load(self, model_name=None):
        """
        Load a model from weights and hyperparamter that are stored in class folder.
        The tensorflow.keras.model is not loaded itself but created new from hyperparameters.

        Parameters
        ----------
        model_name : str,list, optional
            Model names on file. The default is None.

        Raises
        ------
        FileNotFoundError
            If Directory not found.

        Returns
        -------
        list
            Loaded models.

        """
        if not os.path.exists(self._directory):
            raise FileNotFoundError(f"Cannot find class directory")
        directory = os.path.abspath(self._directory)
        
        # Load model_name 
        if(isinstance(model_name, str)):
            self._load(directory,model_name)
            
        elif(isinstance(model_name, list)):
            for name in model_name:
                self._load(directory,name)
        
        #Default just save all
        else:
            savemod_list = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(self._directory, f))]    
            for name in savemod_list:
                self._load(directory,name)
        
        self.logger.debug(f"loaded all models.")
        return self._models
    
      
    def _fit_models(self, target_model,x, y,gpu,proc_async,fitmode,random_shuffle=False):
        # modelfolder
        mod_dir = os.path.join(os.path.abspath(self._directory),target_model)
        #Save data
        model_save_data_to_folder(x, y,target_model,mod_dir,random_shuffle)
        # Pick modeltype from fist hyper
        model_type = self._models_hyper[target_model][0]['general']['model_type']
        #Start proc per NN
        proclist = []
        for i in range(self._addNN):
            if(model_type == 'energy_gradient'):
                if(proc_async==True):
                    proclist.append(fit_model_energy_gradient_async(i,mod_dir,gpu[i],fitmode))
                else:
                    fit_model_energy_gradient(i,mod_dir,gpu[i],fitmode)
            if(model_type == 'nac'):
                if(proc_async==True):
                    proclist.append(fit_model_nac_async(i,mod_dir,gpu[i],fitmode))
                else:
                    fit_model_nac(i,mod_dir,gpu[i],fitmode)
     
        self.logger.debug(f"successfully started training for models {target_model}")
        return proclist
                 
 
    def _read_fit_error(self,target_model):
        # modelfolder
        outdir = os.path.join(os.path.abspath(self._directory),target_model)
        fit_error = []
        try:
            for i in range(self._addNN):
                error_val = np.load(os.path.join(outdir,"fiterr_valid" +'_v%i'%i+ ".npy"))
                fit_error.append(error_val)
        except:
            print(f"Error: Can not find fit error output {target_model}. Fit may not have run correctly!")
        
        return fit_error
    
    
    def fit(self, x, y , gpu_dist = {}, proc_async = True, fitmode= "training", random_shuffle = False):
        """
        Fit NN to data. Model weights and hyper parameters are always saved to file before fit.
        The fit routine calls training scripts on the datafolder with parallel runtime.
        The type of execution is found in nn_pes_src.fit with the training nn_pes_src.training_ scripts.
        
        Parameters
        ----------
        x : np.array
            Coordinates in Angstroem of shape (batch,Atoms,3)
        y : dict
            dictionary of y values for each model. 
            Energy in Bohr, Gradients in Hatree/Bohr, NAC in 1/Hatre by default.
            Units are cast for fitting into eV/Angstroem and can be accessed in hyperparameters.
        gpu_dist : dict
            Dictionary with same modelname and list of GPUs for each NN. Default is {}
            Example {'nac' : [0,0] } both NNs for NAC on GPU:0
        proc_async : True
            Try to run parallel. Default is true.
        fitmode : str
            Whether to do 'training' or 'retraining' the existing model in hyperparameter category. Default is 'training'.  
            In principle every reasonable category can be created in hyperparameters.
        random_shuffle : bool
            Whether to shuffle data before fitting. Default is False.  
            
        Returns
        -------
        ferr : dict
            Fitting Error.

        """
        #List of models
        models_available = sorted(list(self._models.keys()))
        models_to_train = sorted(list(y.keys()))
        
        #Check if model can be identified
        for modtofit in models_to_train:
            if(modtofit not in models_available):
                raise TypeError(f"Cannot train on data: {models_to_train} does not match models {models_available}!")
        
        #Check GPU Assignment and default to -1
        gpu_dict_clean = {}
        for modtofit in models_to_train:
            if(modtofit in gpu_dist):
                gpu_dict_clean[modtofit] = gpu_dist[modtofit]
            else:
                gpu_dict_clean[modtofit] = [-1 for i in range(self._addNN)] 
        
        #Check Fitmode
        if(fitmode!= 'training' and fitmode!= 'retraining' and fitmode!= "resample"):
            print("Warning: Unkown fitmode not completed by default hyperparameter.")
        
        #Fitting
        proclist = []
        for target_model, ydata in y.items():
            #Save model here with hyper !!!!
            self.save(target_model)
            self.logger.debug(f"starting training model {target_model}")
            proclist += self._fit_models(target_model,x, ydata,gpu_dict_clean[target_model],proc_async,fitmode,random_shuffle)
        
        #Wait for fits
        if(proc_async == True):
            print("Fits submitted, waiting...")
            #Wait for models to finish
            for proc in proclist:
                proc.wait()    
        
        #Look for fiterror in folder
        print("Seraching Folder for fitresult...")
        self.load(models_to_train)
        fit_error = {}
        for target_model in y.keys():
            fit_error[target_model] = self._read_fit_error(target_model)
            

        return fit_error

    
    def _predict_models(self,name,x): 
        #Check type with first hyper
        if(self._models_hyper[name][0]['general']['model_type'] == 'energy_gradient'):
            energy = []
            gradient = []
            for i in range(self._addNN):
                temp =  predict_model_energy_gradient(self._models[name][i],x,
                                                      batch_size_predict=self._models_hyper[name][i]['predict']['batch_size_predict'])
                # temp =  predict_model_energy_gradient_precomputed(self._models_precomputed[name][i],
                #                                      self._models_features[name][i],
                #                                      x,
                #                                      hyper=self._models_hyper[name][i])
                energy.append(temp[0])
                gradient.append(temp[1])
            energy = np.array(energy)
            gradient = np.array(gradient)
            energy_mean = np.mean(energy,axis=0)
            gradient_mean = np.mean(gradient,axis=0)
            energy_std = np.std(energy,axis=0)*2
            gradient_std = np.std(gradient,axis=0)*2
            return [energy_mean,gradient_mean],[energy_std,gradient_std]
            
        if(self._models_hyper[name][0]['general']['model_type'] == 'nac'):
            nac = []
            for i in range(self._addNN):
                temp = predict_model_nac(self._models[name][i],x,
                                         batch_size_predict=self._models_hyper[name][i]['predict']['batch_size_predict'])
                # temp = predict_model_nac_precomputed(self._models_precomputed[name][i],
                #                          self._models_features[name][i],
                #                          x,
                #                          hyper=self._models_hyper[name][i])
                nac.append(temp)
            nac = np.array(nac)
            nac_mean = np.mean(nac,axis=0)
            nac_std = np.std(nac,axis=0)*2 
            return nac_mean,nac_std
        
    
    def predict(self, x):
        """
        Prediction for all models available. Prediction is slower but works on large data.

        Parameters
        ----------
        x : np.array
            Coordinates in Angstroem of shape (batch,Atoms,3)

        Returns
        -------
        result : dict
            All model predictions: {'energy_gradient' : [np.aray,np.array] , 'nac' : np.array , ..}.
        error : dict
            Error estimate for each value: {'energy_gradient' : [np.aray,np.array] , 'nac' : np.array , ..}.

        """
        result = {}
        error = {}
        for name in self._models.keys():
            temp = self._predict_models(name,x)
            result[name] = temp[0]
            error[name] = temp[1]
        
        return result,error
    

    def _call_models(self,name,x): 
        #Check type with first hyper
        if(self._models_hyper[name][0]['general']['model_type'] == 'energy_gradient'):
            energy = []
            gradient = []
            for i in range(self._addNN):
                # temp =  call_model_energy_gradient_precomputed(self._models_precomputed[name][i],
                #                                       self._models_features[name][i],
                #                                       x,
                #                                       hyper=self._models_hyper[name][i])
                temp = self._models[name][i](x)
                energy.append(tf.expand_dims(temp[0],axis=0))
                gradient.append(tf.expand_dims(temp[1],axis=0))
            energy = tf.concat(energy,axis=0)
            gradient = tf.concat(gradient,axis=0)
            energy_mean = tf.math.reduce_mean(energy,axis=0)
            gradient_mean = tf.math.reduce_mean(gradient,axis=0)
            energy_std = tf.math.reduce_std(energy,axis=0)*2
            gradient_std = tf.math.reduce_std(gradient,axis=0)*2
            return [energy_mean.numpy(),gradient_mean.numpy()],[energy_std.numpy(),gradient_std.numpy()]
            
        if(self._models_hyper[name][0]['general']['model_type'] == 'nac'):
            nac = []
            for i in range(self._addNN):
                # temp = call_model_nac_precomputed(self._models_precomputed[name][i],
                #                           self._models_features[name][i],
                #                           x,
                #                           hyper=self._models_hyper[name][i])
                temp = self._models[name][i](x)
                nac.append(tf.expand_dims(temp,axis=0))
            nac = tf.concat(nac,axis=0)
            nac_mean = tf.math.reduce_mean(nac,axis=0)
            nac_std = tf.math.reduce_std(nac,axis=0)*2 
            return nac_mean.numpy(),nac_std.numpy()
    
    
    def call(self,x):
        """
        Faster prediction without looping batches. Requires single small batch (batch, Atoms,3) that fit into memory.

        Parameters
        ----------
        x : np.array
            Coordinates in Angstroem of shape (batch,Atoms,3)

        Returns
        -------
        result : dict
            All model predictions: {'energy_gradient' : [np.aray,np.array] , 'nac' : np.array , ..}.
        error : dict
            Error estimate for each value: {'energy_gradient' : [np.aray,np.array] , 'nac' : np.array , ..}.

        """
        result = {}
        error = {}
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        for name in self._models.keys():
            temp = self._call_models(name,x)
            result[name] = temp[0]
            error[name] = temp[1]
        
        
        return result,error

    
    def shuffle(self,datalist):
        """
        Shuffle datalist consistently, i.e. each data [x,y,y2,...] in the same way.

        Parameters
        ----------
        datalist : list
            List of numpy arrays that have the same datalength (axis=0).

        Returns
        -------
        shuffle_ind : np.array
            Index assignment for the shuffle for x,y etc.
        outlist : list
            Shuffled list of the same data.

        """
        if(isinstance(datalist,list) == False):
            print("Warning: Expected datalist")
            datalist = [datalist]
        shuffle_ind, outlist = datalist_make_random_shuffle(datalist)
        self._last_shuffle = shuffle_ind
        return shuffle_ind, outlist
    


    def merge(self,datalist,datalist2,val_split=0.1):
        """
        Merge two datasets so that the correct segments of validation split are kept for merged split.

        Parameters
        ----------
        datalist : list
            List of numpy arrays.
        datalist2 : list
            List of numpy arrays. They will be merged with the respective np.array from datalist.
        val_split : float, optional
            Size of the validation split. The default is 0.1.

        Returns
        -------
        outlist : list
            Single list of merged datasets.

        """
        if(isinstance(datalist,list) == False):
            print("Warning: Expected datalist")
            datalist = [datalist]
        if(isinstance(datalist2,list) == False):
            print("Warning: Expected datalist")
            datalist2 = [datalist2]
        
        if(len(datalist) != len(datalist2)):
            print("Error: Datalists do not match in length",len(datalist),"and",len(datalist2))
        
        outlist = []
        for i in range(len(datalist)):
            outlist.append(merge_data_in_chunks(datalist[i],datalist2[i],val_split))
        
        return outlist
    
    
    
    def _resample_update_active(self,x,y,indall,ind_act,chunks):
        #Select indall/indact = ind_unkwon
        ind_unknown = indall[np.isin(indall,ind_act,invert=True)]
        x_unknown = x[ind_unknown]
        y_unknown = index_data_in_y_dict(y,ind_unknown)
        
        #Predict unkown
        y_pred = self.predict(x_unknown)[0]
        
        #Get most dataindex of largest error
        maxerrind,errors = find_samples_with_max_error(y_unknown ,y_pred)
        #Select a chunk of the largest error index
        ind_add = ind_unknown[maxerrind[:chunks]]
        #Add new 
        ind_new = np.concatenate([ind_act,ind_add],axis=0)
        
        return ind_new,errors
    
    
        
    def _resample_plot_stats(self,name,out_index,out_error,out_fiterr,out_testerr):
        # Take type and scaling into from first NN for each model
        if(self._models_hyper[name][0]['general']['model_type'] == 'energy_gradient'):
            plot_resampling_gradient(os.path.join(self._directory,name,"fit_stats"),
                                     out_index,
                                     np.array(out_error),
                                     np.array(out_fiterr) ,
                                     np.array(out_testerr),
                                     unit_energy_conv = self._models_hyper[name][0]['model']['y_energy_unit_conv'],
                                     unit_force_conv = self._models_hyper[name][0]['model']['y_gradient_unit_conv'],
                                     unit_energy=self._models_hyper[name][0]['plots']['unit_energy'],
                                     unit_force=self._models_hyper[name][0]['plots']['unit_gradient']
                                     )
        if(self._models_hyper[name][0]['general']['model_type'] == 'nac'):
            plot_resampling_nac(os.path.join(self._directory,name,"fit_stats"),
                                     out_index,
                                     np.array(out_error),
                                     np.array(out_fiterr),
                                     np.array(out_testerr),
                                     unit_nac_conv = self._models_hyper[name][0]['model']['y_nac_unit_conv'],
                                     unit_nac = self._models_hyper[name][0]['plots']['unit_nac']
                                     )

    def resample(self,x,y,gpu_dist,proc_async=True,random_shuffle=False,stepsize = 0.05,test_size = 0.05):
        """
        Use uncertainty sampling as active learning to effectively reduce dataset size.

        Parameters
        ----------
        x : np.array
            Coordinates in Angstroem of shape (batch,Atoms,3)
        y : dict
            dictionary of y values for each model. 
            Energy in Bohr, Gradients in Hatree/Bohr. NAC in 1/Hatree.
            Units are cast for fitting into eV/Angstroem.
        gpu_dist : dict
            Dictionary with same modelname and list of GPUs for each NN. Default is {}
            Example {'nac' : [0,0] } both NNs for NAC on GPU:0
        proc_async : bool
            Try to run parallel. Default is True.    
        random_shuffle : bool
            Whether to shuffle data before fitting. Default is False. 
        stepsize : float
            Fraction of the original dataset size to add during each iteration.
        test_size : float
            Fraction of test set which is kept out of sampling. Default is 0.05
            
        Returns
        -------
        out_index : list
            List of np.array of used indices from original data for each iteration.
        out_error : dict
            Error of the unseen data.
        out_fiterr : dict
            Validation error of fit.
        out_testerr : dict 
            Error on test set.    
        
        """
        #Output stats and info
        out_index = []  # the used data-indices for each iteration
        out_error = []  # Error of total datast
        out_fiterr = [] # Error of validation set
        out_testerr = [] # Error of test set
                
        #Temporary set number of NN to 1
        numNN = self._addNN
        self._addNN = 1
        
        # Length of sets
        total_len = len(x)
        chunks = int(total_len*stepsize)
        testchunk = int(total_len*test_size)
        # Index list is used for active learning
        index_all = np.arange(0,total_len)
        
        if(random_shuffle==True):
            index_all = index_make_random_shuffle(index_all)
          
        #select active and test indices
        ind_test = index_all[:testchunk] 
        index_all = index_all[testchunk:] #Remove testset from all index
        ind_activ = index_all[:chunks]
        
        #Fix test data
        x_test = x[ind_test]
        y_test = index_data_in_y_dict(y,ind_test)
        
        #Start selection
        for i_runs in range(int(1/stepsize)):
            out_index.append(ind_activ)
            #Select active data
            x_active = x[ind_activ]
            y_active = index_data_in_y_dict(y,ind_activ)
            #Make fit
            fiterrfun = self.fit(x_active,y_active,gpu_dist,proc_async,fitmode='resample',random_shuffle=False)
            out_fiterr.append(fiterrfun)
            #Get test error
            _, ertestd = find_samples_with_max_error(y_test,self.predict(x_test)[0])
            out_testerr.append(ertestd)
            #Resample
            ind_new,errs = self._resample_update_active(x,y,index_all,ind_activ,chunks)
            out_error.append(errs)
            
            #Set new acitve
            ind_activ = ind_new
            
            if(len(ind_activ)>= len(index_all)):
                break
            else:
                print("Next fit with length: ",len(ind_activ))
        
        self._addNN = numNN
        
        # Possible plot of results here.
        for name in y.keys():
            self._resample_plot_stats(name,
                                      out_index,
                                      [x[name] for x in out_error],
                                      [x[name] for x in out_fiterr],
                                      [x[name] for x in out_testerr])
                
        
        return out_index,out_error,out_fiterr,out_testerr
