import numpy as np

from pyNNsMD.nn_pes_src.device import set_gpu

# No GPU for prediciton or the main class
set_gpu([-1])

from pyNNsMD.nn_pes import NeuralNetPes

# list of angles
anglist = [[1, 0, 2], [1, 0, 4], [2, 0, 4], [0, 1, 3], [0, 1, 8], [3, 1, 8], [0, 4, 5], [0, 4, 6], [0, 4, 7], [6, 4, 7],
           [5, 4, 7], [5, 4, 6], [9, 8, 10], [1, 8, 10], [9, 8, 11], [1, 8, 9], [1, 8, 11], [10, 8, 11]]
dihedlist = [[5, 1, 2, 9], [3, 1, 2, 4]]

# Load data
x = np.load("butene/butene_x.npy")
eng = np.load("butene/butene_energy.npy")
grads = np.load("butene/butene_force.npy")
nac = np.load("butene/butene_nac.npy")
print(x.shape,eng.shape,grads.shape,nac.shape)

# Make class and folder
nn = NeuralNetPes("NN_1", mult_nn=2)

hyper_energy = {  # Model
    'general': {
        'model_type': 'mlp_e'
    },
    'model': {
        'atoms': 12,
        'states': 2,
        'depth': 3,
        'nn_size': 200,
        'use_reg_weight': {'class_name': 'L1', 'config': {'l1': 1e-5}},
        'invd_index': True,
        'angle_index': anglist,
        'dihed_index': dihedlist,
    },
    'training': {
        'normalization_mode': 1,
        'epo': 500,
        'val_split': 0.1,
        'batch_size': 64,
        'step_callback': {'use': True, 'epoch_step_reduction': [100, 200, 400, 100],
                          'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6]},
    }
}

hyper_grads = {  # Model
    'general': {
        'model_type': 'mlp_eg'
    },
    'model': {
        'atoms': 12,
        'states': 2,
        'depth': 3,
        'nn_size': 200,
        'use_reg_weight': {'class_name': 'L1', 'config': {'l1': 1e-5}},
        'invd_index': True,
        'angle_index': anglist,
        'dihed_index': dihedlist,
    },
    'training': {
        'normalization_mode': 1,
        'epo': 500,
        'loss_weights': [1, 10],
        'val_split': 0.1,
        'batch_size': 64,
        'step_callback': {'use': True, 'epoch_step_reduction': [100, 200, 400, 100],
                          'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6]},
    }
}

hyper_nac2 = {  # Model
    'general': {
        'model_type': 'mlp_nac2',
    },
    'model': {
        'atoms': 12,
        'states': 2,
        'depth': 3,
        'nn_size': 200,
        'use_reg_weight': {'class_name': 'L1', 'config': {'l1': 1e-4}},
        'invd_index': True,
        'angle_index': anglist,
        'dihed_index': dihedlist,
    },
    'training': {
        'phase_less_loss': True,
        'normalization_mode': 1,
        'epo': 500,
        'val_split': 0.1,
        'pre_epo': 10,
        'batch_size': 64,
        'step_callback': {'use': True, 'epoch_step_reduction': [100, 200, 400, 100],
                          'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6]},
    }
}

nn.create({
    # 'e': hyper_energy,
    # 'eg': hyper_grads,
    'nac2': hyper_nac2
})

y = {
    # 'e': eng,
    # 'eg': [eng, grads],
    'nac2' : nac,
}

fitres = nn.fit(x,
                y,
                gpu_dist={  # Set to {} or to -1 if no gpu to use
                    'e': [0, 0],
                    'eg': [0, 0],
                    'nac': [0, 0],
                    'nac2': [0, 0],
                },
                proc_async=True,
                fitmode='training',
                random_shuffle=True)

test_call = nn.call(x[0:10])
test_predict = nn.predict(x[0:10])
