"""
Model selection
"""
import os

import tensorflow as tf

from pyNNsMD.models.mlp_e import EnergyModel
from pyNNsMD.models.mlp_eg import EnergyGradientModel
from pyNNsMD.models.mlp_nac import NACModel
from pyNNsMD.models.mlp_nac2 import NACModel2
from pyNNsMD.nn_pes_src.hypers.hyper_mlp_e import DEFAULT_HYPER_PARAM_ENERGY
from pyNNsMD.nn_pes_src.hypers.hyper_mlp_eg import DEFAULT_HYPER_PARAM_ENERGY_GRADS
from pyNNsMD.nn_pes_src.hypers.hyper_mlp_nac import DEFAULT_HYPER_PARAM_NAC
from pyNNsMD.nn_pes_src.predicting.predict_mlp_eg import predict_uncertainty_mlp_eg
from pyNNsMD.nn_pes_src.predicting.predict_mlp_nac import predict_uncertainty_mlp_nac
from pyNNsMD.scaler.energy import EnergyGradientStandardScaler,EnergyStandardScaler
from pyNNsMD.scaler.nac import NACStandardScaler


def get_default_hyperparameters_by_modeltype(model_type):
    """
    Select the default parameters for each model

    Args:
        model_type (str): Model identifier.

    Returns:
        dict: Default hyper parameters for model.

    """
    model_dict = {'mlp_eg': DEFAULT_HYPER_PARAM_ENERGY_GRADS,
                  'mlp_e': DEFAULT_HYPER_PARAM_ENERGY,
                  'mlp_nac': DEFAULT_HYPER_PARAM_NAC,
                  'mlp_nac2': DEFAULT_HYPER_PARAM_NAC}
    return model_dict[model_type]


def get_path_for_fit_script(model_type):
    """
    Interface to find the path of training scripts.

    For now they are expected to be in the same folder-system as calling .py script.

    Args:
        model_type (str): Name of the model.

    Returns:
        filepath (str): Filepath pointing to training scripts.

    """
    # Ways of finding path either os.getcwd() or __file__ or just set static path with install...
    # locdiR = os.getcwd()
    filepath = os.path.abspath(os.path.dirname(__file__))
    # STATIC_PATH_FIT_SCRIPT = ""
    fit_script = {"mlp_eg": "training_mlp_eg.py",
                  "mlp_nac": "training_mlp_nac.py",
                  "mlp_nac2": "training_mlp_nac2.py",
                  "mlp_e": "training_mlp_e.py"}
    outpath = os.path.join(filepath, "training", fit_script[model_type])
    return outpath


def get_default_scaler(model_type):
    """
    Get default values for scaler in and output for each model.

    Args:
        model_type (str): Model identifier.

    Returns:
        Dict: Scaling dictionary.

    """
    if (model_type == 'mlp_e'):
        return EnergyStandardScaler()
    elif model_type == 'mlp_eg':
        return EnergyGradientStandardScaler()
    elif (model_type == 'mlp_nac' or 'mlp_nac2'):
        return NACStandardScaler()
    else:
        print("Error: Unknown model type", model_type)
        raise TypeError(f"Error: Unknown model type for default scaler {model_type}")


def get_model_by_type(model_type, hyper):
    """
    Find the implemented model by its string identifier.

    Args:
        model_type (str): Model type.
        hyper (dict): Dict with hyper parameters.

    Returns:
        tf.keras.model: Defult initialized tf.keras.model.

    """
    if (model_type == 'mlp_nac'):
        return NACModel(hyper)
    elif (model_type == 'mlp_nac2'):
        return NACModel2(hyper)
    elif (model_type == 'mlp_eg'):
        return EnergyGradientModel(**hyper)
    elif (model_type == 'mlp_e'):
        return EnergyModel(**hyper)
    else:
        print("Error: Unknown model type", model_type)
        raise TypeError(f"Error: Unknown model type forn{model_type}")



def predict_uncertainty(model_type, out):
    if (model_type == 'mlp_nac'):
        return predict_uncertainty_mlp_nac(out)
    elif (model_type == 'mlp_nac2'):
        return predict_uncertainty_mlp_nac(out)
    elif (model_type == 'mlp_eg'):
        return predict_uncertainty_mlp_eg(out)
    elif (model_type == 'mlp_e'):
        return predict_uncertainty_mlp_eg(out)
    else:
        print("Error: Unknown model type for predict", model_type)
        raise TypeError(f"Error: Unknown model type for predict {model_type}")


def call_convert_x_to_tensor(model_type, x):
    if (model_type == 'mlp_eg'):
        return tf.convert_to_tensor(x, dtype=tf.float32)
    elif (model_type == 'mlp_nac'):
        return tf.convert_to_tensor(x, dtype=tf.float32)
    elif (model_type == 'mlp_nac2'):
        return tf.convert_to_tensor(x, dtype=tf.float32)
    elif (model_type == 'mlp_e'):
        return tf.convert_to_tensor(x, dtype=tf.float32)
    else:
        print("Error: Unknown model type for predict", model_type)
        raise TypeError(f"Error: Unknown model type for predict {model_type}")


def call_convert_y_to_numpy(model_type, temp):
    if (model_type == 'mlp_nac'):
        return temp.numpy()
    if (model_type == 'mlp_nac2'):
        return temp.numpy()
    elif (model_type == 'mlp_eg'):
        return [temp[0].numpy(), temp[1].numpy()]
    elif (model_type == 'mlp_e'):
        return [temp[0].numpy(), temp[1].numpy()]
    else:
        print("Error: Unknown model type for predict", model_type)
        raise TypeError(f"Error: Unknown model type for predict {model_type}")
