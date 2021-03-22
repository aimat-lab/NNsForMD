import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pyNNsMD.models.mlp_eg import EnergyGradientModel
from pyNNsMD.plots.pred import plot_scatter_prediction
from pyNNsMD.scaler.energy import EnergyGradientStandardScaler
from pyNNsMD.utils.loss import ScaledMeanAbsoluteError

# Load data
x = np.load("butene/butene_x.npy")
eng = np.load("butene/butene_energy.npy")
grads = np.load("butene/butene_force.npy")
print(x.shape, eng.shape, grads.shape)

# Generate model
model = EnergyGradientModel(atoms=12, states=2, invd_index=True)

# Scale in- and output
# Important: x, energy and gradients can not be scaled completely independent!!
scaler = EnergyGradientStandardScaler()
x_scaled, y_scaled = scaler.fit_transform(x=x, y=[eng, grads])
scaler.print_params_info()

# Precompute features plus derivative
# Features are normalized automatically
model.precomputed_features = True
feat_x, feat_grad = model.precompute_feature_in_chunks(x_scaled, batch_size=32)
model.set_const_normalization_from_features(feat_x)
print("Feature norm: ", model.get_layer('feat_std').get_weights())

# compile model with optimizer
# And use scaled metric to revert the standardization of the output for metric during fit updates (optional).
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
mae_energy = ScaledMeanAbsoluteError(scaling_shape=scaler.energy_std.shape)
mae_force = ScaledMeanAbsoluteError(scaling_shape=scaler.gradient_std.shape)
mae_energy.set_scale(scaler.energy_std)
mae_force.set_scale(scaler.gradient_std)
model.compile(optimizer=optimizer,
              loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1, 5],
              metrics=[[mae_energy], [mae_force]])

# fit with precomputed features and normalized energies, gradients
model.fit(x=[feat_x[:2000], feat_grad[:2000]], y=[y_scaled[0][:2000], y_scaled[1][:2000]],
          batch_size=32, epochs=100, verbose=2)

# Now set the model to coordinates and predict the test data
model.precomputed_features = False
y_pred = model.predict(x_scaled[2000:])

# invert standardization
x_pred, y_pred = scaler.inverse_transform(x=x_scaled[2000:], y=y_pred)

# Plot Prediction
fig = plot_scatter_prediction(eng[2000:], y_pred[0])
plt.show()
