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
The main class ``pyNNsMD.NNsMD.NeuralNetEnsemble`` holds a list of keras models and custom scaler classes to transform or standardize input/output.
Construction of ``NeuralNetEnsemble`` requires a filepath and the number of model instances to keep.

```python
from pyNNsMD.NNsMD import NeuralNetEnsemble
nn = NeuralNetEnsemble("TestEnergy/", 2)
```

Adding the models and scaler classes to ``NeuralNetEnsemble`` via `create`. 
Custom classes can be added to the modules in ``pyNNsMD.models`` and ``pyNNsMD.scalers``, 
but which must implement proper config and weight handling. 
Instead of class instances a deserialization via keras config-dictionaries is supported for `create`.

```python
from pyNNsMD.models.mlp_e import EnergyModel
from pyNNsMD.scaler.energy import EnergyStandardScaler
nn.create(models=[EnergyModel(atoms=12, states=2), EnergyModel(atoms=12, states=2)],
          scalers=[EnergyStandardScaler(), EnergyStandardScaler()])
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