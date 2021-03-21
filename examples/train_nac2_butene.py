import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pyNNsMD.models.mlp_nac2 import NACModel2
from pyNNsMD.scaler.nac import NACStandardScaler
from pyNNsMD.utils.loss import ScaledMeanAbsoluteError,NACphaselessLoss
from pyNNsMD.plots.pred import plot_scatter_prediction

# Load data
x = np.load("butene/butene_x.npy")
nacs = np.load("butene/butene_nac.npy")
print(x.shape,nacs.shape)

# Generate model
model = NACModel2(atoms=12,states=2,invd_index=True)

# Scale in- and output
scaler = NACStandardScaler()
x_scaled, y_scaled = scaler.fit_transform(x=x,y=nacs)
scaler.print_params_info()

# Precompute features plus derivative
# Features are normalized automatically
model.precomputed_features = True
feat_x, feat_grad = model.precompute_feature_in_chunks(x_scaled, batch_size=32)
print("Feature norm: ",model.get_layer('feat_std').get_weights())

# compile model with optimizer
# And use scaled metric to revert the standardization of the output for metric during fit updates (optional).
# Important: Nacs can be trained with a phaseless-loss!
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
mae_scale = ScaledMeanAbsoluteError(scaling_shape=scaler.nac_std.shape)
mae_scale.set_scale(scaler.nac_std)
model.compile(optimizer=optimizer,
                  loss=NACphaselessLoss(number_state=2, shape_nac=(12, 3), name='phaseless_loss'),
                  metrics=[mae_scale])

# fit with precomputed features and normalized energies, gradients
model.fit(x=[feat_x[:2000],feat_grad[:2000]],y=y_scaled[:2000],
          batch_size=32,epochs=100,verbose =2)

# Now set the model to coordinates and predict the test data
model.precomputed_features = False
y_pred = model.predict(x_scaled[2000:])

# invert standardization
x_pred, y_pred = scaler.inverse_transform(x=x_scaled[2000:],y=y_pred)

# Plot Prediction
fig = plot_scatter_prediction(nacs[2000:],y_pred)
plt.show()
