import numpy as np
import pprint

from pyNNsMD.src.device import set_gpu

# No GPU for prediciton or the main class
set_gpu([-1])

from pyNNsMD.NNsMD import NeuralNetEnsemble
from pyNNsMD.hypers.hyper_mlp_e import DEFAULT_HYPER_PARAM_ENERGY as hyper

pprint.pprint(hyper)

# list of angles
anglist = [[1, 0, 2], [1, 0, 4], [2, 0, 4], [0, 1, 3], [0, 1, 8], [3, 1, 8], [0, 4, 5], [0, 4, 6], [0, 4, 7], [6, 4, 7],
           [5, 4, 7], [5, 4, 6], [9, 8, 10], [1, 8, 10], [9, 8, 11], [1, 8, 9], [1, 8, 11], [10, 8, 11]]
dihedlist = [[5, 1, 2, 9], [3, 1, 2, 4]]

# Load data
atoms = [["C", "C", "H", "H", "C", "F", "F", "F", "C", "F", "H", "H"]]*2701
geos = np.load("butene/butene_x.npy")
energy = np.load("butene/butene_energy.npy")
grads = np.load("butene/butene_force.npy")
nac = np.load("butene/butene_nac.npy")
print(geos.shape, energy.shape, grads.shape, nac.shape)

hyper["model"]["config"].update({"atoms": 12, "states": 2, "nn_size": 500,
                                 'angle_index': anglist,
                                 'dihed_index': dihedlist})

ensemble_path = "TestEnergy/"

nn = NeuralNetEnsemble(ensemble_path, 2)
nn.create(models=[hyper["model"]]*2,
          scalers=[hyper["scaler"]]*2)
nn.save()

# nn.data_path("data_dir/")
nn.data(atoms=atoms, geometries=geos, energies=energy)

nn.train_test_split(dataset_size=len(energy), n_splits=5, shuffle=True)
# nn.train_test_indices(train=[np.array(), np.array()], test=[np.array(), np.array()])

nn.training([hyper["training"]]*2, fit_mode="training")

fit_error = nn.fit(["training_mlp_e"]*2, fit_mode="training", gpu_dist=[0, 0], proc_async=True)
print(fit_error)

nn.load()

test = nn.predict(geos)
test_batch = nn.call(geos[:32])
print("Error prediction on all data:", np.mean(np.abs(test[0]/2 + test[1]/2 - energy)))

# nn.clean()
