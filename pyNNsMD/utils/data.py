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


def parse_list_to_xyz_str(mol: list, comment: str = ""):
    """Convert list of atom and coordinates list into xyz-string.
    Args:
        mol (list): Tuple or list of `[['C', 'H', ...], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], ... ]]`.
        comment (str): Comment for comment line in xyz string. Default is "".
    Returns:
        str: Information in xyz-string format.
    """
    atoms = mol[0]
    coordinates = mol[1]
    if len(atoms) != len(coordinates):
        raise ValueError("Number of atoms does not match number of coordinates for xyz string.")
    xyz_str = str(int(len(atoms))) + "\n"
    if "\n" in comment:
        raise ValueError("Line break must not be in the comment line for xyz string.")
    xyz_str = xyz_str + comment + "\n"
    for a_iter, c_iter in zip(atoms, coordinates):
        _at_str = str(a_iter)
        _c_format_str = " {:.10f}" * len(c_iter) + "\n"
        xyz_str = xyz_str + _at_str + _c_format_str.format(*c_iter)
    return xyz_str


def write_list_to_xyz_file(filepath: str, mol_list: list):
    """Write a list of nested list of atom and coordinates into xyz-string. Uses :obj:`parse_list_to_xyz_str`.
    Args:
        filepath (str): Full path to file including name.
        mol_list (list): List of molecules, which is a list of pairs of atoms and coordinates of
            `[[['C', 'H', ... ], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], ... ]], ... ]`.
    """
    with open(filepath, "w+") as file:
        for x in mol_list:
            xyz_str = parse_list_to_xyz_str(x)
            file.write(xyz_str)

