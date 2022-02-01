
DEFAULT_HYPER_PARAM_ENERGY_GRADS = {
    'model': {
        "class_name": "EnergyGradientModel",
        "config": {
            # 'atoms': 12,  # Must be set for each molecule; number of atoms
            # 'states': 2,  # (batch,states) and (batch,states,atoms,3)
            'nn_size': 100,  # size of each layer
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
            "model_module": "mlp_eg"
        }
    },
    "scaler": {
        "class_name": "EnergyGradientStandardScaler",
        "config": {
            "scaler_module": "energy"
        }
    },
    'training': {
        'initialize_weights': True,
        'energy_only': False,
        'loss_weights': [1, 10],  # weights between energy and gradients
        'learning_rate': 0.5e-3,  # learning rate, can be modified by callbacks
        'epo': 3000,  # total epochs
        'batch_size': 64,  # batch size
        'epostep': 10,  # steps of epochs for validation, also steps for changing callbacks
        "callbacks": [],
        'unit_energy': "eV",
        'unit_gradient': "eV/A"
    },
    'retraining': {
        'initialize_weights': False,
        'energy_only': False,
        'normalization_mode': 1,  # Normalization False/0 for no normalization/unity mulitplication
        'loss_weights': [1, 10],  # weights between energy and gradients
        'learning_rate': 1e-3,  # learning rate, can be modified by callbacks
        'epo': 1000,  # total epochs
        'batch_size': 64,  # batch size
        'epostep': 10,  # steps of epochs for validation, also steps for changing callbacks
        "callbacks": [],
        'unit_energy': "eV",
        'unit_gradient': "eV/A"
    }
}