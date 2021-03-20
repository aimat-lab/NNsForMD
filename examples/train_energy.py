import numpy as np

# Load data
x = np.load("butene/butene_x.npy")
eng = np.load("butene/butene_energy.npy")
grads = np.load("butene/butene_force.npy")
print(x.shape,eng.shape,grads.shape)




