
"""
Test/Example for Neural net PES

@author: Patrick
"""

import numpy as np
import json
import pickle

from pyNNsMD.nn_pes_src.device import set_gpu

set_gpu([0]) #No GPU for prediciton

from pyNNsMD.nn_pes import NeuralNetPes
import tensorflow as tf

anglist = [[1, 0, 2], [1, 0, 4], [2, 0, 4], [0, 1, 3], [0, 1, 8], [3, 1, 8], [0, 4, 5], [0, 4, 6], [0, 4, 7], [6, 4, 7], [5, 4, 7], [5, 4, 6], [9, 8, 10], [1, 8, 10], [9, 8, 11], [1, 8, 9], [1, 8, 11], [10, 8, 11]]
dihydlist = [[5,1,2,9],[3,1,2,4]]

#load data
#with open('data6180_060520.json','r') as indata:
with open('data4961-Initial-half.json','r') as indata:
#with open('data2701-Initial.json','r') as indata:
    data=json.load(indata)

natom, nstate, xyz, invr, energy, grad, nac, ci,_ = data

#Target Properties y
x = np.array(xyz)
x = np.array(x[:,:,1:],dtype=np.float)  
grads = np.array(grad) * 27.21138624598853/0.52917721090380
Energy = np.array(energy) *27.21138624598853 
nacs= np.array(nac)/0.52917721090380

nn = NeuralNetPes("NN5fit3")

hyper_energy =  {    #Model
                'general':{
                    'model_type' : 'energy_gradient'
                },
                'model':{
                    'atoms': 12,
                    'states': 2, 
                    'Depth' : 3,                    
                    'nn_size' : 700,   # size of each layer
                    'invd_index' : True,
                    'angle_index' : [],# anglist,
                    'dihyd_index'  : [], #dihydlist ,  # list of dihydral angles with index ijkl angle is between ijk and jkl
                },
                'training':{
                    'epo': 3000,
                    'loss_weights' : [1,10], 
                    'val_split' : 0.1, 
                    'batch_size' : 64,
                    'step_callback' : {'use': True , 'epoch_step_reduction' : [500,1500,500,500],'learning_rate_step' :[1e-3,1e-4,1e-5,1e-6]},
                }
                }
hyper_nac =  {    #Model
                'general':{
                    'model_type' : 'nac',
                },
                'model':{
                    'atoms' : 12,
                    'states':1 , 
                    'Depth' : 3,
                    'nn_size' : 700,
                    'invd_index' : True,
                    'angle_index' : [],#,anglist,
                    'dihyd_index'  : [],#dihydlist ,  # list of dihydral angles with index ijkl angle is between ijk and jkl
                },
                'training':{
                    'phase_less_loss' : True,
                    'epo': 3000,
                    'val_split' : 0.1, 
                    'pre_epo': 50,
                    'batch_size' : 64,
                    'step_callback' : {'use': True , 'epoch_step_reduction' : [500,1500,500,500],'learning_rate_step' :[1e-3,1e-4,1e-5,1e-6]},                     
                }
            } 



nn.create({ 
            'energy_gradient': hyper_energy,
            'nac': hyper_nac  
            })
nn.save()
#out2 = nn.predict2(x)

#nn.load() #see if loading works 

y = {
      'energy_gradient': [Energy,grads],
      'nac' : nacs,
      }


fitres = nn.fit(x,
                y,
                gpu_dist= { #Set to {} or to -1 if no gpu to use
                            'energy_gradient' : [0,0],
                            'nac' : [0,0],
                            },
                proc_async=True,
                fitmode='training',
                random_shuffle=True)

# out = nn.resample(x,
#                 y,
#                 gpu_dist= { #Set to {} or to -1 if no gpu to use
#                             'energy_gradient' : [0,0],
#                             'nac' : [0,0],
#                             },
#                 proc_async=True,
#                 random_shuffle=True)
testout = nn.predict(x)[0]['energy_gradient']
print(np.mean(np.abs(testout[0]-Energy)))
testout = nn.predict(x)[0]['nac']
print(np.mean(np.abs(testout-nacs)))
#print(fitres)
nn.export()
#nn.save()
# test = nn.call(x[0:100])
# test2 = nn.predict(x[0:100])

