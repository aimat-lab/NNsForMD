import numpy as np
import pprint

from pyNNsMD.src.device import set_gpu

# No GPU for prediciton or the main class
set_gpu([-1])

from pyNNsMD.NNsMD import NeuralNetEnsemble
from pyNNsMD.models.schnet_kgcnn import SchnetEnergy
from pyNNsMD.utils.loss import ScaledMeanAbsoluteError
from pyNNsMD.hypers.hyper_schnet_kgcnn_e_g import DEFAULT_HYPER_PARAM_SCHNET_KGCNN_E_G as hyper
from kgcnn.mol.methods import global_proton_dict
from kgcnn.utils.adj import coordinates_to_distancematrix, define_adjacency_from_distance

pprint.pprint(hyper)

# list of angles
anglist = [[1, 0, 2], [1, 0, 4], [2, 0, 4], [0, 1, 3], [0, 1, 8], [3, 1, 8], [0, 4, 5], [0, 4, 6], [0, 4, 7], [6, 4, 7],
           [5, 4, 7], [5, 4, 6], [9, 8, 10], [1, 8, 10], [9, 8, 11], [1, 8, 9], [1, 8, 11], [10, 8, 11]]
dihedlist = [[5, 1, 2, 9], [3, 1, 2, 4]]
range_dist = hyper["model"]["config"]["schnet_kwargs"]["gauss_args"]["distance"]

# Load data
atoms = [["C", "C", "H", "H", "C", "F", "F", "F", "C", "F", "H", "H"]]*2701
geos = np.load("butene/butene_x.npy")
energy = np.load("butene/butene_energy.npy")
grads = np.load("butene/butene_force.npy")
nac = np.load("butene/butene_nac.npy")
print(geos.shape, energy.shape, grads.shape, nac.shape)


atomic_number = [np.array([global_proton_dict[atom] for atom in x]) for x in atoms]
geos = [x for x in geos]
range_indices = [
    define_adjacency_from_distance(coordinates_to_distancematrix(x), max_distance=range_dist)[1] for x in geos]

ensemble_path = "TestEnergyGradientSchnetKGCNN/"

nn = NeuralNetEnsemble(ensemble_path, 2)
nn.create(models=[hyper["model"]]*2,
          scalers=[hyper["scaler"]]*2)
nn.save()

# nn.data_path("data_dir/")
nn.data(atoms=atoms, geometries=geos, energies=energy, forces=grads)

nn.train_test_split(dataset_size=len(energy), n_splits=5, shuffle=True)
# nn.train_test_indices(train=[np.array(), np.array()], test=[np.array(), np.array()])

nn.training([hyper["training"]]*2, fit_mode="training")

fit_error = nn.fit(["training_schnet_kgcnn_e_g"]*2, fit_mode="training", gpu_dist=[0, 0], proc_async=False)
print(fit_error)

nn.load()
for i in range(len(nn)):
    nn[i].energy_only = False
    nn[i].output_as_dict = True

# test = nn.predict([atomic_number, geos, range_indices])
test_call = nn.call([atomic_number[:32], geos[:32], range_indices[:32]])
# print("Error prediction on all data:", np.mean(np.abs(test[0][0]/2 + test[1][0]/2 - energy)))
print("Error call on batch:", np.mean(np.abs(test_call[0][0][:32]/2 + test_call[1][0][:32]/2 - energy[:32])))
# nn.clean()