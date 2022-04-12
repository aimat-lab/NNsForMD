[![Documentation Status](https://readthedocs.org/projects/pynnsmd/badge/?version=latest)](https://pynnsmd.readthedocs.io/en/latest/?badge=latest)
![PyPI](https://img.shields.io/pypi/v/pyNNsMD)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyNNsMD)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/aimat-lab/NNsForMD)

# NNsForMD

Neural network class for molecular dynamics to predict potential energy, gradients and non-adiabatic couplings (NACs).

# Table of Contents
* [General](#general)
* [Installation](#installation)
* [Documentation](#documentation)
* [Usage](#usage)
* [Examples](#examples)
* [Citing](#citing)
* [References](#references)

<a name="general"></a>
# General
This repo is currently under construction. The original version used as the PyRAI2MD interface is v1.0.0.



<a name="installation"></a>
# Installation

Clone repository https://github.com/aimat-lab/NNsForMD and install for example with editable mode:

```bash
pip install -e ./pyNNsMD
```
or latest release via Python Package Index.

```bash
pip install pyNNsMD
```

<a name="documentation"></a>
# Documentation

Auto-documentation generated at https://pynnsmd.readthedocs.io/en/latest/index.html

<a name="usage"></a>
# Usage

#### Ensemble
The main class ``pyNNsMD.NNsMD.NeuralNetEnsemble`` holds a list of keras models and custom scaler classes to transform or standardize input/output.
Construction of ``NeuralNetEnsemble`` requires a filepath and the number of model instances to keep.

```python
from pyNNsMD.NNsMD import NeuralNetEnsemble
nn = NeuralNetEnsemble("TestEnergy/", 2)
```

Adding the models and scaler classes to ``NeuralNetEnsemble`` via `create`. 
Custom classes can be added to the modules in ``pyNNsMD.models`` and ``pyNNsMD.scalers``, 
but which must implement proper config and weight handling. 
Note that data format between model and scaler must be compatible.
Instead of class instances a deserialization via keras config-dictionaries is supported for `create()`.

```python
from pyNNsMD.models.mlp_e import EnergyModel
from pyNNsMD.scaler.energy import EnergyStandardScaler

nn.create(models=[EnergyModel(atoms=12, states=2), EnergyModel(atoms=12, states=2)],
          scalers=[EnergyStandardScaler(), EnergyStandardScaler()])
```

The models and scaler must be saved to disk to prepare for training, which includes config and weights.

```python
nn.save()
```

#### Data

The data is stored to the directory specified in ``NeuralNetEnsemble``.
Data format passed to ``NeuralNetEnsemble.data()`` must be nested python-only lists.
The geometries are stored as `.xyz` and everything else as `.json`. 
Note that the training scripts must be compatible with the data format.

```python
atoms = [["C", "C"]]
geos = [[[0.147, 0.024, -0.680], [-0.165, -0.037, 0.652]]]
energy = [[-20386.37, -20383.93]]

nn.data(atoms=atoms, geometries=geos, energies=energy)
# nn.data_path("data_dir/") if data can't be saved in working directory.
```
#### Training

For training the train and test indices must also be saved to file for each model directory.
This can be achieved via ``train_test_split()``, 
or by directly passing an index-list for each model with ``train_test_indices()``.
Note that the different models are sought to be trained on different splits.

```python
nn.train_test_split(dataset_size=1, n_splits=1, shuffle=True) # Usually n_splits=5 or 10
# nn.train_test_indices(train=[np.array([0]), np.array([0])], test=[np.array([0]), np.array([0])])
```

The hyperparameter for training are passed as `.json` to each model folder. 
See ``pyNNsMD.hypers`` modules for example hyperparameter.

```python
nn.training([{
    'initialize_weights': True, 'epo': 1000, 'batch_size': 64, 'epostep': 10, 
    'learning_rate': 1e-3, "callbacks": [], 'unit_energy': "eV", 'unit_gradient': "eV/A"
}]*2, fit_mode="training")
```

#### Fitting

With `fit()` a training script is run for each model from the model's directory. 
The training script should be stored in ``pyNNsMD.training``. 
Note that the training script must be compatible with model and data. 
The training script must provide command line arguments 'index', 'filepath', 'gpus' and 'mode'.
The training can be distributed on multiple or a single gpu (for small networks).

```python
fit_error = nn.fit(["training_mlp_e"]*2, fit_mode="training", 
                   gpu_dist=[0, 0], proc_async=True)
print(fit_error)
```

#### Loading

After fitting the model can be recreated from config and the weights loaded from file with ``load()``.

```python
nn.load()
```

#### Prediction

The model's prediction can be obtained from the corresponding input data via `predict()` and ``call()``.
The both input and output is rescaled by the scaler to match the model standardized input and output.
Furthermore, the subclassed model should implement ``call_to_tensor_input`` and ``call_to_numpy_output`` or optionally
`predict_to_tensor_input` and `predict_to_numpy_output`, 
if the model requires a specific tensor input as in `call()`.

```python
test = nn.predict(geos)
# test_batch = nn.call(geos[:32])  # Faster than predict.
```

<a name="examples"></a>
# Examples

A set of examples can be found in [examples](examples), that demonstrate usage and typical tasks for projects.

<a name="citing"></a>
# Citing

If you want to cite this repository or the PyRAI2MD code, please refer to our publication at:
```
@Article{JingbaiLi2021,
    author ="Li, Jingbai and Reiser, Patrick and Boswell, Benjamin R. and Eberhard, André and Burns, Noah Z. and Friederich, Pascal and Lopez, Steven A.",
    title  ="Automatic discovery of photoisomerization mechanisms with nanosecond machine learning photodynamics simulations",
    journal  ="Chem. Sci.",
    year  ="2021",
    pages  ="-",
    publisher  ="The Royal Society of Chemistry",
    doi  ="10.1039/D0SC05610C",
    url  ="http://dx.doi.org/10.1039/D0SC05610C"
}
```

<a name="references"></a>
# References

* https://onlinelibrary.wiley.com/doi/full/10.1002/qua.24890
* https://pubs.acs.org/doi/abs/10.1021/acs.chemrev.0c00749