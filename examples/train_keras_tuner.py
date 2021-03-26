from datetime import datetime
import numpy as np
import os
import tensorflow as tf
import kerastuner as kt

from pyNNsMD.models.mlp_eg import EnergyGradientModel
from pyNNsMD.scaler.energy import EnergyGradientStandardScaler
from pyNNsMD.utils.loss import ScaledMeanAbsoluteError


def build_model(hp):
    hp_depth = hp.Int('depth', min_value=3, max_value=5, step=1)
    hp_nnsize = hp.Int('nn_size', min_value=25, max_value=400, step=25)
    hp_eneloss = hp.Int('ene_loss', min_value=1, max_value=1000, step=10)
    hp_frcloss = hp.Int('frc_loss', min_value=1, max_value=1000, step=10)
    hp_lr_start = hp.Choice('lr_start', [1e-3, 5e-4, 1e-4])
    hp_reg = hp.Choice('reg', ['l1', 'l2'])

    model = EnergyGradientModel(atoms=12, states=2, invd_index=True,
                                nn_size=hp_nnsize, depth=hp_depth,
                                use_reg_weight=hp_reg)

    model.precomputed_features = True
    # compile model with optimizer
    # And use scaled metric to revert the standardization of the output for metric during fit updates (optional).
    optimizer = tf.keras.optimizers.Adam(lr=hp_lr_start)

    model.compile(optimizer=optimizer,
                  loss=['mean_squared_error', 'mean_squared_error'],
                  loss_weights=[hp_eneloss, hp_frcloss],
                  metrics=[['mean_absolute_error'], ['mean_absolute_error']])

    return model


# Load data
x = np.load("butene/butene_x.npy")
eng = np.load("butene/butene_energy.npy")
grads = np.load("butene/butene_force.npy")
print(x.shape, eng.shape, grads.shape)

# create temp model for feature calculation
featmodel = EnergyGradientModel(atoms=12, states=2, invd_index=True)

# Scale in- and output
# Important: x, energy and gradients can not be scaled completely independent!!
scaler = EnergyGradientStandardScaler()
x_scaled, y_scaled = scaler.fit_transform(x=x, y=[eng, grads])
scaler.print_params_info()

# Precompute features plus derivative
# Features are normalized automatically
featmodel.precomputed_features = True
feat_x, feat_grad = featmodel.precompute_feature_in_chunks(x_scaled, batch_size=32)

tuner = kt.Hyperband(build_model,
                     objective='val_loss',
                     max_epochs=10, factor=3, directory="kt_test")

tuner.search(x=[feat_x[:1000], feat_grad[:1000]],
             y=[y_scaled[0][:1000], y_scaled[1][:1000]],
             epochs=100, validation_split=0.1)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
    Search complete, best hyperparameters:
    depth = {best_hps.get('depth')}
    nn_size = {best_hps.get('nn_size')}
    ene_wt = {best_hps.get('ene_loss')}
    frc_wt = {best_hps.get('frc_loss')}
    learn rate = {best_hps.get('lr_start')}
    reg = {best_hps.get('reg')}
""")
