
DEFAULT_HYPER_PARAM_ENERGY = {
    'model': {
        "class_name": "EnergyModel",
        "config": {
            # 'atoms': 12,  # Must be set for each molecule; number of atoms
            # 'states': 2,  # (batch,states) and (batch,states,atoms,3)
            "nn_size": 100,  # size of each layer
            'depth': 3,  # number of layers
            'activ': {'class_name': "pyNNsMD>leaky_softplus", "config": {'alpha': 0.03}},  # activation function
            # Regularozation
            'use_dropout': False,  # Whether to use dropout
            'dropout': 0.005,  # dropout values
            'use_reg_activ': None,  # {'class_name': 'L1', 'config': {'l1': 0.009999999776482582}}
            'use_reg_weight': None,  # {'class_name': 'L1', 'config': {'l1': 0.009999999776482582}}
            'use_reg_bias': None,  # {'class_name': 'L1', 'config': {'l1': 0.009999999776482582}}
            # Features
            'invd_index': True,  # not used yet
            'angle_index': [],  # list-only of shape (N,3) angle: 0-1-2  or alpha(1->0,1->2)
            'dihed_index': [],  # list of dihedral angles with index ijkl angle is between ijk and jkl
            'normalization_mode': 1,  # Normalization False/0 for no normalization/unity mulitplication
            "model_module": "mlp_e",
        }
    },
    "scaler": {
        "class_name": "EnergyStandardScaler",
        "config": {
            "scaler_module": "energy"
        }
    },
    'training': {
        'initialize_weights': True,
        'epo': 3000,  # total epochs
        'batch_size': 64,  # batch size
        'epostep': 10,  # steps of epochs for validation, also steps for changing callbacks
        'loss_weights': [1, 10],  # weights between energy and gradients
        'learning_rate': 1e-3,  # learning rate, can be modified by callbacks
        "callbacks": [],
        # {"class_name": 'StepWiseLearningScheduler', "config": {'epoch_step_reduction': [500, 1500, 500, 500], 'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6]}}
        # {"class_name": 'LinearLearningRateScheduler', "config": {'learning_rate_start': 1e-3, 'learning_rate_stop': 1e-6, 'epo_min': 100, 'epo': 1000}}
        # {"class_name": 'EarlyStopping', "config": {'use': False, 'epomin': 5000, 'patience': 600, 'max_time': 600, 'min_delta': 1e-5, 'loss_monitor': 'val_loss', 'factor_lr': 0.1, 'learning_rate_start': 1e-3, 'learning_rate_stop': 1e-6, 'epostep': 1}}
        # {"class_name": 'LinearWarmupExponentialLearningRateScheduler', "config": {'epo_warmup': 10, 'decay_gamma': 0.1, 'lr_start': 1e-3, 'lr_min': 0.0}}
        'unit_energy': "eV",  # Just for plotting
        'unit_gradient': "eV/A"  # Just for plotting
    },
    'retraining': {
        'initialize_weights': False,
        'loss_weights': [1, 10],  # weights between energy and gradients
        'learning_rate': 1e-3,  # learning rate, can be modified by callbacks
        'epo': 1000,  # total epochs
        'batch_size': 64,  # batch size
        'epostep': 10,  # steps of epochs for validation, also steps for changing callbacks
        "callbacks": [],
        'unit_energy': "eV",  # Just for plottin
        'unit_gradient': "eV/A"  # Just for plottin
    }
}