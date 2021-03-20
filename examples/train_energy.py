import numpy as np
import tensorflow as tf

from pyNNsMD.models.mlp_eg import EnergyGradientModel
from pyNNsMD.scaler.energy import EnergyGradientStandardScaler
from pyNNsMD.utils.loss import ScaledMeanAbsoluteError

# Load data
x = np.load("butene/butene_x.npy")
eng = np.load("butene/butene_energy.npy")
grads = np.load("butene/butene_force.npy")
print(x.shape,eng.shape,grads.shape)

# Generate model
model = EnergyGradientModel(atoms=12,states=2,invd_index=True)

# Scale in- and output
# Note x,energy and gradients can not be scaled completely independent
scaler = EnergyGradientStandardScaler()
x_scaled, y_scaled = scaler.fit_transform(x=x,y=[eng,grads])
scaler.print_params_info()

# Precompute features plus derivative
model.precomputed_features = True
feat_x, feat_grad = model.precompute_feature_in_chunks(x_scaled, batch_size=32)

# Set the constant Standardization layer in the model
# Otherwise this defaults to std=1 and mean=0
model.get_layer('feat_std').compute_const_normalization(feat_x)
print("Feature norm: ",model.get_layer('feat_std').get_weights())

# compile model with optimizer
# And use scaled metric to revert the standardization of the output during fit (optional).
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
mae_energy = ScaledMeanAbsoluteError(scaling_shape=scaler.energy_std.shape)
mae_force = ScaledMeanAbsoluteError(scaling_shape=scaler.gradient_std.shape)
mae_energy.set_scale(scaler.energy_std)
mae_force.set_scale(scaler.gradient_std)
model.compile(optimizer=optimizer,
                  loss=['mean_squared_error','mean_squared_error'], loss_weights=[1,1],
                  metrics=[[mae_energy],[mae_force]])


