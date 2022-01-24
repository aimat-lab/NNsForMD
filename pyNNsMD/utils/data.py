import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import yaml
import json
import os
from importlib.machinery import SourceFileLoader


def save_pickle_file(outlist, filepath):
    """Save to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(outlist, f)


def load_pickle_file(filepath):
    """Load pickle file."""
    with open(filepath, 'rb') as f:
        outlist = pickle.load(f)
    return outlist


def save_json_file(outlist, filepath):
    """Save to json file."""
    with open(filepath, 'w') as json_file:
        json.dump(outlist, json_file)


def load_json_file(filepath):
    """Load json file."""
    with open(filepath, 'r') as json_file:
        file_read = json.load(json_file)
    return file_read


def load_yaml_file(fname):
    """Load yaml file."""
    with open(fname, 'r') as stream:
        outlist = yaml.safe_load(stream)
    return outlist


def save_yaml_file(outlist, fname):
    """Save to yaml file."""
    with open(fname, 'w') as yaml_file:
        yaml.dump(outlist, yaml_file, default_flow_style=False)


def load_hyper_file(file_name):
    """Load hyper-parameters from file. File type can be '.yaml', '.json', '.pickle' or '.py'
    Args:
        file_name (str): Path or name of the file containing hyper-parameter.
    Returns:
        hyper (dict): Dictionary of hyper-parameters.
    """
    if "." not in file_name:
        print("Can not determine file-type.")
        return {}
    type_ending = file_name.split(".")[-1]
    if type_ending == "json":
        return load_json_file(file_name)
    elif type_ending == "yaml":
        return load_yaml_file(file_name)
    elif type_ending == "pickle":
        return load_pickle_file(file_name)
    elif type_ending == "py":
        path = os.path.realpath(file_name)
        hyper = getattr(SourceFileLoader(os.path.basename(path).replace(".py", ""), path).load_module(), "hyper")
        return hyper
    else:
        print("Unsupported file type %s" % type_ending)
    return {}