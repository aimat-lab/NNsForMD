import numpy as np
import pprint

from pyNNsMD.src.device import set_gpu

# No GPU for prediciton or the main class
set_gpu([-1])

from pyNNsMD.NNsMD import NeuralNetEnsemble
from pyNNsMD.hypers.hyper_mlp_eg import DEFAULT_HYPER_PARAM_ENERGY_GRADS as hyper

pprint.pprint(hyper)

# list of angles
anglist = [[1, 0, 2], [1, 0, 4], [2, 0, 4], [0, 1, 3], [0, 1, 8], [3, 1, 8], [0, 4, 5], [0, 4, 6], [0, 4, 7], [6, 4, 7],
           [5, 4, 7], [5, 4, 6], [9, 8, 10], [1, 8, 10], [9, 8, 11], [1, 8, 9], [1, 8, 11], [10, 8, 11]]
dihedlist = [[5, 1, 2, 9], [3, 1, 2, 4]]

# Load data
atoms = [["C","C","C","C", "F", "F", "F", "F","F", "F", "H", "H"]]*2701
geos = np.load("butene/butene_x.npy")
energy = np.load("butene/butene_energy.npy")
grads = np.load("butene/butene_force.npy")
nac = np.load("butene/butene_nac.npy")
print(geos.shape, energy.shape, grads.shape, nac.shape)

hyper["model"]["config"].update({"atoms": 12, "states": 2})

ensemble_path = "TestEnergyGradient/"

nn = NeuralNetEnsemble(ensemble_path, 2)
nn.create(models=[hyper["model"]]*2,
          scalers=[hyper["scaler"]]*2)
nn.save()

nn.data(atoms=atoms, geometries=geos, energies=energy, forces=grads)

nn.train_test_split(dataset_size=len(energy), n_splits=5)
nn.training([hyper["training"]]*2, fit_mode="training")
fit_error = nn.fit(["training_mlp_eg"]*2, fit_mode="training", gpu_dist=[0, 0], proc_async=True)
print(fit_error)

nn.load()

test = nn.predict(geos)
print("Error prediction on all data:",
      np.mean(np.abs(test[0][0]/2 + test[1][0]/2 - energy)),
      np.mean(np.abs(test[0][1]/2 + test[1][1]/2 - grads))
      )