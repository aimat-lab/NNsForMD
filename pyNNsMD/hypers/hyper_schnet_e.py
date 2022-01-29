DEFAULT_HYPER_PARAM_SCHNET_E = {
    'model': {
        "class_name": "SchnetEnergy",
        "config": {
            "model_module": "schnet_e",
            "schnet_kwargs": {
                'name': "Schnet",
                'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                'input_embedding': {"node": {"input_dim": 95, "output_dim": 64}},
                "make_distance": True, 'expand_distance': True,
                'interaction_args': {"units": 128, "use_bias": True,
                                     "activation": 'kgcnn>shifted_softplus', "cfconv_pool": 'sum'},
                'node_pooling_args': {"pooling_method": "sum"},
                'depth': 4,
                'gauss_args': {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
                'verbose': 10,
                'last_mlp': {"use_bias": [True, True], "units": [128, 64],
                             "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus']},
                'output_embedding': 'graph',
                "use_output_mlp": True,
                'output_mlp': {"use_bias": [True, True], "units": [64, 2],
                               "activation": ['kgcnn>shifted_softplus', "linear"]}
            }
        }
    },
    "scaler": {
        "class_name": "EnergyStandardScaler",
        "config": {
            "scaler_module": "energy",
            "use_x_mean": False,  # Not possible or necessary for Schnet input.
            "use_x_std": False
        }
    },
    "training": {
        'initialize_weights': True,
        'epo': 400,  # total epochs
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
    }
}
