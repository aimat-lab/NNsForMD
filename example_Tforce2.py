"""
Test/Example for Neural net learing coupling and gradient of the Hamiltonian

@author: Mila
"""

import numpy as np
import json


from parse_data import parse_force_data, get_shuffled_indices
from parse_data import normalize_invdist, normalize_angles, normalize_dihedrals

from pyNNsMD.nn_pes_src.device import set_gpu

set_gpu([-1]) #No GPU for prediciton

from pyNNsMD.nn_pes import NeuralNetPes
import tensorflow as tf

# list of angles/dihedrals to include in the representation
anglist = [] #[[1, 0, 2], [1, 0, 4], [2, 0, 4], [0, 1, 3], [0, 1, 8], [3, 1, 8], [0, 4, 5], [0, 4, 6], [0, 4, 7], [6, 4, 7], [5, 4, 7], [5, 4, 6], [9, 8, 10], [1, 8, 10], [9, 8, 11], [1, 8, 9], [1, 8, 11], [10, 8, 11]]
dihydlist = [] #[[5,1,2,9],[3,1,2,4]]

#load data
natom = 24
data_path = "data"
data = parse_force_data(data_path, natom)

pair_xyz, couplings, site_xyz, sites = data

#decide if site or couplings
xyz = pair_xyz
y = couplings

#small set for experimentation
shuffled_idx = get_shuffled_indices(y.shape[0])
ntrain = 2500

xyz = xyz[shuffled_idx[:ntrain]]
y = y[shuffled_idx[:ntrain]]

#separate coordinates from gradients
x = xyz[:,:,1:4]
grads = xyz[:,:,4:]




print(x.dtype, y.dtype)

nn = NeuralNetPes("Tforce1")

hyper_energy =  {    #Model
                'general':{
                    'model_type' : 'energy_gradient'
                },
                'model':{
                    'atoms': 2*natom,
                    'states': 1, 
                    'Depth' : 3,                    
                    'nn_size' : 500,   # size of each layer
                    'use_reg_weight' : {'class_name': 'L1', 'config': {'l1': 1e-3}},
                    'invd_index' : True,
                    'angle_index' : [],# anglist,
                    'dihyd_index'  : [], #dihydlist ,  # list of dihydral angles with index ijkl angle is between ijk and jkl
                },
                'training':{
                    'epo': 3250,
                    'loss_weights' : [1,2], 
                    'val_split' : 0.1, 
                    'batch_size' : 64,
                    'step_callback' : {'use': True , 'epoch_step_reduction' : [50,3000,100,100],'learning_rate_step' :[1e-3,1e-4,1e-5,1e-6]},
                }
                }

nn.create({
            'energy_gradient': hyper_energy
            })

y = {
      'energy_gradient': [np.expand_dims(y,axis=-1),np.expand_dims(grads,axis=1)]
      }

# fitres = nn.fit(x,
#                 y,
#                 gpu_dist= { #Set to {} or to -1 if no gpu to use
#                             'energy_gradient' : [0,0],
#                             },
#                 proc_async=True,
#                 fitmode='training',
#                 random_shuffle=True)


# out = nn.resample(x,
#                 y,
#                 gpu_dist= { #Set to {} or to -1 if no gpu to use
#                             'energy_gradient' : [0,0],
#                             'nac' : [0,0],
#                             },
#                 proc_async=True,
#                 random_shuffle=True)

#print(fitres)

nn.load()
#nn.save()
out = nn.predict(x)
out2 = nn.call(x)
