"""
Test/Example for Neural net PES

@author: Patrick
"""

import numpy as np
import json
import pickle

from pyNNsMD.nn_pes_src.device import set_gpu

set_gpu([-1]) #No GPU for prediciton

from pyNNsMD.nn_pes import NeuralNetPes
import tensorflow as tf

anglist = [[1, 0, 2], [1, 0, 4], [2, 0, 4], [0, 1, 3], [0, 1, 8], [3, 1, 8], [0, 4, 5], [0, 4, 6], [0, 4, 7], [6, 4, 7], [5, 4, 7], [5, 4, 6], [9, 8, 10], [1, 8, 10], [9, 8, 11], [1, 8, 9], [1, 8, 11], [10, 8, 11]]
dihydlist = [[5,1,2,9],[3,1,2,4]]

#load data
#with open('data6180_060520.json','r') as indata:
with open('data6232-32.json','r') as indata:
#with open('data2701-Initial.json','r') as indata:
    data=json.load(indata)

natom, nstate, xyz, invr, energy, grad, nac, ci,_ = data

#Target Properties y
x = np.array(xyz)
x = np.array(x[:,:,1:],dtype=np.float)  
grads = np.array(grad) * 27.21138624598853/0.52917721090380
Energy = np.array(energy) *27.21138624598853 
nacs= np.array(nac)/0.52917721090380

nn = NeuralNetPes("NN5fit4e")

hyper_energy =  {    #Model
                'general':{
                    'model_type' : 'mlp_eg'
                },
                'model':{
                    'atoms': 12,
                    'states': 2, 
                    'Depth' : 3,                    
                    'nn_size' : 1000,   # size of each layer
                    'use_reg_activ' : {'class_name': 'L1', 'config': {'l1': 1e-4}},
                    'invd_index' : True,
                    'angle_index' : [],# anglist,
                    'dihyd_index'  : [], #dihydlist ,  # list of dihydral angles with index ijkl angle is between ijk and jkl
                },
                'training':{
                    'normalization_mode' : 2,
                    'epo': 50,
                    'loss_weights' : [1,10], 
                    'val_split' : 0.1, 
                    'batch_size' : 64,
                    'step_callback' : {'use': True , 'epoch_step_reduction' : [2000,2000,500,500],'learning_rate_step' :[1e-3,1e-4,1e-5,1e-6]},
                }
                }
hyper_nac =  {    #Model
                'general':{
                    'model_type' : 'mlp_nac',
                },
                'model':{
                    'atoms' : 12,
                    'states': 1 , 
                    'Depth' : 3,
                    'nn_size' : 1000,
                    'use_reg_activ' : {'class_name': 'L1', 'config': {'l1': 1e-4}},
                    'invd_index' : True,
                    'angle_index' : [],#,anglist,
                    'dihyd_index'  : [],#dihydlist ,  # list of dihydral angles with index ijkl angle is between ijk and jkl
                },
                'training':{
                    'phase_less_loss' : False,
                    'normalization_mode' : 2, 
                    'epo': 50,
                    'val_split' : 0.1, 
                    'pre_epo': 10,
                    'batch_size' : 64,
                    'step_callback' : {'use': True , 'epoch_step_reduction' : [2000,2000,500,500],'learning_rate_step' :[1e-3,1e-4,1e-5,1e-6]},                     
                }
            } 



nn.create({ 
            'eg': hyper_energy,
            'nac': hyper_nac  
            })


y = {
      'eg': [Energy,grads],
      'nac' : nacs,
      }


fitres = nn.fit(x,
                y,
                gpu_dist= { #Set to {} or to -1 if no gpu to use
                            'eg' : [0,0],
                            'nac' : [0,0],
                            },
                proc_async=True,
                fitmode='training',
                random_shuffle=True)

# out = nn.resample(x, 
#                 y,
#                 gpu_dist= { #Set to {} or to -1 if no gpu to use
#                             'eg' : [0,0],
#                             'nac' : [0,0],
#                             },
#                 proc_async=True,
#                 random_shuffle=True)


