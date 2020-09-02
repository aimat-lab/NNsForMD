"""
Default hyperparameters
"""

import json


DEFAULT_HYPER_PARAM_ENERGY_GRADS = {
                        
                        'general':
                        {
                            'model_type' : 'energy_gradient',   #which model type to use
                            'main_dir' : '',     # not used atm
                            'model_dir' : '',    # not used atm
                            'info' : '',     # not used atm
                            'pyr_version' : None    # not used atm
                        },
                        'model': #Model Parameters   # fixes model, cannot be changed after init
                        {
                            'atoms' : 2,     #number of atoms
                            'states' : 1,    # for 1: (batch,) and (batch,atoms,3) or states>1 : (batch,states) and (batch,states,atoms,3)
                            'use_dropout' : False,   #Whether to use dropout
                            'dropout' : 0.005,   #dropout values
                            'use_reg_activ' : None,      #activity regularization as string, set to: 'l2','l1','l1_l2'
                            'use_reg_weight' : None,    #weight regularization as string, set to: 'l2','l1','l1_l2'
                            'use_reg_bias' : None,  #bias regularization as string, set to: 'l2','l1','l1_l2'
                            'reg_l2' : 1e-5,     #regularization parameter l2
                            'reg_l1' : 1e-5,     #regularization parameter l1
                            'Depth' : 3,     # number of layers
                            'activ' : 'leaky_softplus',  # activation function
                            'activ_alpha' : 0.03,       # alpha parameter in activation function
                            'loss_weights' : [1,10],     # weights between energy and gradients
                            'nn_size' : 1000,     # size of each layer
                            'y_energy_mean' : 0,     # offset for energy only
                            'y_energy_std' : 1,      # scaling for both energy and gradients
                            'y_energy_unit_conv' :27.21138624598853 ,     # conversion Hatree to eV after scaling
                            'y_gradient_unit_conv': 27.21138624598853/0.52917721090380,  # conversion from H/bohr to eV/A after scaling
                            #Features
                            'use_invdist' : True,  # use invers distances as features (default) 
                            'invd_index' : [],  # not used yet
                            'invd_mean' :0,  #offset for inverse distance
                            'invd_std' : 1,   #scaling for inverse distance
                            'use_bond_angles' : False,
                            'angle_index'  : [] ,  # list-only of shape (N,3) angle: 0-1-2  or alpha(1->0,1->2)
                            'angle_mean' : 0,   # remove offset from angle
                            'angle_std' : 1,  # scale angles
                            'use_dihyd_angles' : False,  # use dihydral angles between planes
                            'dihyd_index'  : [] ,  # list of dihydral angles with index ijkl angle is between ijk and jkl
                            'dihyd_mean' : 0,  # remove offset from dihyd
                            'dihyd_std' : 1,  # scale dihyd
                            'learning_rate_compile' : 1e-3
                        },
                        'training':
                        {
                            #can be changed after model created
                            'reinit_weights' : True,  #Whether to reset the weights before fit.
                            'val_disjoint' : True,
                            'val_split' : 0.1, 
                            'epo': 10000,  # total epochs
                            'epomin' : 5000,# minimum number of epochs before doing something
                            'patience' : 600, # patience before reducing learning rate
                            'max_time' : 600,  # maximum time to wait for fit, used only in early callback
                            'batch_size' : 64,  # batch size
                            'delta_loss' : 1e-5,  # minimal change for expected loss, used only in early callback
                            'loss_monitor': 'val_loss',  #monitor
                            'factor_lr' : 0.1, #factor to reduce learning rate, also for exp callback
                            'epostep' : 10,  # steps of epochs for validation, also steps for exp callback
                            'learning_rate_step' : [1e-3,1e-4,1e-5,1e-6], #for step callback
                            'epoch_step_reduction' : [4000,3000,2000,1000], #for step callback
                            'learning_rate_start' : 1e-3,
                            'learning_rate_stop' : 1e-6,
                            'use_linear_callback' : False,
                            'use_early_callback' : False,
                            'use_exp_callback' : False,
                            'use_step_callback' : True
                        },
                        'retraining': #used for transfer learning
                        {
                            #can be changed after model created
                            'reinit_weights' : False,
                            'val_disjoint' : True,
                            'val_split' : 0.1, 
                            'epo': 2000,  # total epochs
                            'epomin' : 1000,# minimum number of epochs before doing something
                            'patience' : 600, # patience before reducing learning rate
                            'max_time' : 600,  # maximum time to wait for fit, used only in early callback
                            'batch_size' : 64,  # batch size
                            'delta_loss' : 1e-5,  # minimal change for expected loss, used only in early callback
                            'loss_monitor': 'val_loss',  #monitor
                            'factor_lr' : 0.1, #factor to reduce learning rate, also for exp callback
                            'epostep' : 10,  # steps of epochs for validation, also steps for exp callback
                            'learning_rate_step' : [1e-4,1e-5,1e-6], #for step callback
                            'epoch_step_reduction' : [1000,500,250], #for step callback
                            'learning_rate_start' : 1e-3,
                            'learning_rate_stop' : 1e-6,
                            'use_linear_callback' : False,
                            'use_early_callback' : False,
                            'use_exp_callback' : False,
                            'use_step_callback' : True,
                        },
                        'resample': #used active learning
                        {
                            #can be changed after model created
                            'reinit_weights' : True,
                            'val_disjoint' : False,
                            'val_split' : 0.05, 
                            'epo': 10000,  # total epochs
                            'epomin' : 5000,# minimum number of epochs before doing something
                            'patience' : 600, # patience before reducing learning rate
                            'max_time' : 600,  # maximum time to wait for fit, used only in early callback
                            'batch_size' : 64,  # batch size
                            'delta_loss' : 1e-5,  # minimal change for expected loss, used only in early callback
                            'loss_monitor': 'val_loss',  #monitor
                            'factor_lr' : 0.1, #factor to reduce learning rate, also for exp callback
                            'epostep' : 10,  # steps of epochs for validation, also steps for exp callback
                            'learning_rate_step' : [1e-3,1e-4,1e-5,1e-6], #for step callback
                            'epoch_step_reduction' : [4000,3000,2000,1000], #for step callback
                            'learning_rate_start' : 1e-3,
                            'learning_rate_stop' : 1e-6,
                            'use_linear_callback' : False,
                            'use_early_callback' : False,
                            'use_exp_callback' : False,
                            'use_step_callback' : True,
                        },
                        'predict':
                        {
                            'batch_size_predict' : 265,
                            'try_predict_hessian' : False, #not implemented yet
                        },
                        'plots':
                        {
                            'unit_energy' : "eV",
                            'unit_gradient' : "eV/A"
                        }
                    }



DEFAULT_HYPER_PARAM_NAC = { 
    
                        'general':
                        {
                            'model_type' : 'nac',
                            'main_dir' : '',
                            'model_dir' : '',
                            'info' : '',
                            'pyr_version' : None,
                        },
                        'model':#Model Parameters # fixed model, cannot be changed after init
                        { 
                            'atoms' : 2,
                            'states' : 1, # for 1: (batch,) and (batch,atoms,3) or states>1 : (batch,states) and (batch,states,atoms,3)
                            'dropout' : 0.005,
                            'use_dropout' : False,
                            'use_reg_activ' : None,  #activity regularization as string 'l2','l1','l1_l2'
                            'use_reg_weight' : None, #weight regularization as string 'l2','l1','l1_l2'
                            'use_reg_bias' : None,  #bias regularization as string 'l2','l1','l1_l2'
                            'reg_l2' : 1e-5,
                            'reg_l1' : 1e-5,
                            'Depth' : 3,
                            'activ' : 'leaky_softplus',
                            'activ_alpha' : 0.05,
                            'nn_size' : 1000,
                            'invd_mean' : 0, 
                            'invd_std' : 1,
                            'y_nac_unit_conv' : 1/0.52917721090380 , # conversion 1/Bohr to 1/A after scaling!!
                            'y_nac_mean' : 0,
                            'y_nac_std' : 1, 
                            'use_invdist' : True,
                            'invd_index' : [],  # not used yet
                            'invd_mean' : 0, 
                            'invd_std' : 1,
                            'use_bond_angles' : False,
                            'angle_index'  : [] ,  # list-only of shape (N,3) angle: 0-1-2  or alpha(1->0,1->2)
                            'angle_mean' : 0,   # remove offset from angle
                            'angle_std' : 1,  # scale angles
                            'use_dihyd_angles' : False,  # use dihydral angles between planes
                            'dihyd_index'  : [] ,  # list of dihydral angles (N,4) with index ijkl angle is between ijk and jkl
                            'dihyd_mean' : 0,  # remove offset from dihyd
                            'dihyd_std' : 1,  # scale dihyd
                            'learning_rate_compile' : 1e-3,
                            'phase_less_loss' : True
                        },
                        'training':{
                            #Fit information
                            'reinit_weights' : True,
                            'val_disjoint' : True,
                            'val_split' : 0.1, 
                            'epo': 10000,
                            'pre_epo' : 100, # number of epochs without phaseless loss
                            'epomin' : 5000,
                            'patience' : 600,
                            'max_time' : 600,
                            'batch_size' : 64,
                            'delta_loss' : 1e-5,
                            'loss_monitor': 'val_loss', 
                            'factor_lr' : 0.1, 
                            'epostep' : 10,
                            'learning_rate_step' : [1e-3,1e-4,1e-5,1e-6], #for step callback
                            'epoch_step_reduction' : [4000,3000,2000,1000],#for step callback
                            'learning_rate_start' : 1e-3,
                            'learning_rate_stop' : 1e-6,
                            'use_linear_callback' : False,
                            'use_early_callback' : False,
                            'use_exp_callback' : False,
                            'use_step_callback' : True, 
                        },
                        'retraining':{
                            #Fit information
                            'reinit_weights' : False,
                            'phase_less_loss' : True,
                            'val_disjoint' : True,
                            'val_split' : 0.1, 
                            'epo': 2000,
                            'pre_epo' : 100, 
                            'epomin' : 1000,
                            'patience' : 600,
                            'max_time' : 600,
                            'batch_size' : 64,
                            'delta_loss' : 1e-5,
                            'loss_monitor': 'val_loss', 
                            'factor_lr' : 0.1, 
                            'epostep' : 10,
                            'learning_rate_step' : [1e-4,1e-5,1e-6], #for step callback
                            'epoch_step_reduction' : [1000,500,250],#for step callback
                            'learning_rate_start' : 1e-3,
                            'learning_rate_stop' : 1e-6,
                            'use_linear_callback' : False,
                            'use_early_callback' : False,
                            'use_exp_callback' : False,
                            'use_step_callback' : True, 
                        },
                        'resample':{
                            #Fit information
                            'reinit_weights' : True,
                            'phase_less_loss' : True,
                            'val_disjoint' : False,
                            'val_split' : 0.05, 
                            'epo': 10000,
                            'pre_epo' : 100, 
                            'epomin' : 5000,
                            'patience' : 600,
                            'max_time' : 600,
                            'batch_size' : 64,
                            'delta_loss' : 1e-5,
                            'loss_monitor': 'val_loss', 
                            'factor_lr' : 0.1, 
                            'epostep' : 10,
                            'learning_rate_step' : [1e-3,1e-4,1e-5,1e-6], #for step callback
                            'epoch_step_reduction' : [4000,3000,2000,1000],#for step callback
                            'learning_rate_start' : 1e-3,
                            'learning_rate_stop' : 1e-6,
                            'use_linear_callback' : False,
                            'use_early_callback' : False,
                            'use_exp_callback' : False,
                            'use_step_callback' : True, 
                        },
                        "predict":
                        {
                            'batch_size_predict' : 265,
                            'try_predict_hessian' : False, #not implemented yet
                        },
                        'plots':
                        {
                            'unit_nac' : "1/A"
                        }
                    }


def _save_hyp(HYPERPARAMETER,filepath): 
    with open(filepath, 'w') as f:
        json.dump(HYPERPARAMETER, f)

def _load_hyp(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)