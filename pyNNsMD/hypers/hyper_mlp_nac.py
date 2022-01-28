
DEFAULT_HYPER_PARAM_NAC = {
    'model': {
        "class_name": "NACModel",
        "config": {
            'atoms': 12,
            'states': 2,  # (batch,states*(states-1)/2,atoms,3)
            'depth': 3,
            'activ': {'class_name': "pyNNsMD>leaky_softplus", "config": {'alpha': 0.03}},  # activation function,
            'nn_size': 100,
            # Regularization
            'dropout': 0.005,
            'use_dropout': False,
            'use_reg_activ': None,  # {'class_name': 'L1', 'config': {'l1': 0.009999999776482582}}
            'use_reg_weight': None,
            'use_reg_bias': None,
            # features
            'invd_index': True,  # not used yet
            'angle_index': [],  # list-only of shape (N,3) angle: 0-1-2  or alpha(1->0,1->2)
            'dihed_index': [],  # list of dihedral angles (N,4) with index ijkl angle is between ijk and jkl
            'normalization_mode': 1,  # Normalization False/0 for no normalization/unity mulitplication
            "model_module": "mlp_nac"
        }
    },
    "scaler": {
        "class_name": "NACStandardScaler",
        "config": {
            "scaler_module": "nac"
        }
    },
    'training': {
        'initialize_weights': True,
        'learning_rate': 0.5e-3,
        'phase_less_loss': True,
        'epo': 3000,
        'pre_epo': 50,  # number of epochs without phaseless loss
        'epostep': 10,
        'batch_size': 64,
        "callbacks": [],
        'unit_nac': "1/A"
    },
    'retraining': {
        'initialize_weights': False,  # To take old weights
        'learning_rate': 1e-3,
        'phase_less_loss': True,
        'epo': 1000,
        'pre_epo': 50,  # number of epochs without phaseless loss
        'epostep': 10,
        'batch_size': 64,
        "callbacks": [],
        'unit_nac': "1/A"
    },
}
