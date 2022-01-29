import time
import matplotlib as mpl
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
mpl.use('Agg')

import os
import json
import pickle
import sys

import argparse

parser = argparse.ArgumentParser(description='Train a energy-gradient model from data, parameters given in a folder')

parser.add_argument("-i", "--index", required=True, help="Index of the NN to train")
parser.add_argument("-f", "--filepath", required=True, help="Filepath to weights, hyperparameter, data etc. ")
parser.add_argument("-g", "--gpus", default=-1, required=True, help="Index of gpu to use")
parser.add_argument("-m", "--mode", default="training", required=True, help="Which mode to use train or retrain")
args = vars(parser.parse_args())

file_std_out = open(os.path.join(args['filepath'], "fitlog.txt"), 'w')
sys.stderr = file_std_out
sys.stdout = file_std_out

print("Input argpars:", args)

from pyNNsMD.src.device import set_gpu

set_gpu([int(args['gpus'])])
print("Logic Devices:", tf.config.experimental.list_logical_devices('GPU'))

import pyNNsMD.utils.callbacks
import pyNNsMD.utils.activ
from pyNNsMD.models.schnet_e import SchnetEnergy
from pyNNsMD.utils.data import load_json_file, read_xyz_file, save_json_file
from pyNNsMD.scaler.energy import EnergyStandardScaler
from pyNNsMD.utils.loss import ScaledMeanAbsoluteError, get_lr_metric, r2_metric
from pyNNsMD.plots.loss import plot_loss_curves, plot_learning_curve
from pyNNsMD.plots.pred import plot_scatter_prediction
from kgcnn.utils.adj import define_adjacency_from_distance, coordinates_to_distancematrix
from kgcnn.utils.data import ragged_tensor_from_nested_numpy

global_proton_dict = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11,
                      'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                      'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
                      'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38,
                      'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47,
                      'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56,
                      'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
                      'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
                      'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83,
                      'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
                      'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101,
                      'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
                      'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117,
                      'Og': 118, 'Uue': 119}
inverse_global_proton_dict = {value: key for key, value in global_proton_dict.items()}


def train_model_energy(i=0, out_dir=None, mode='training'):
    r"""Train an energy model. Uses precomputed feature. Always require scaler.

    Args:
        i (int, optional): Model index. The default is 0.
        out_dir (str, optional): Directory for this training. The default is None.
        mode (str, optional): Fit-mode to take from hyper-parameters. The default is 'training'.

    Raises:
        ValueError: Wrong input shape.

    Returns:
        error_val (list): Validation error for (energy,gradient).
    """
    i = int(i)
    np_eps = np.finfo(float).eps

    # Load everything from folder
    training_config = load_json_file(os.path.join(out_dir, mode+"_config.json"))
    model_config = load_json_file(os.path.join(out_dir, "model_config.json"))
    i_train = np.load(os.path.join(out_dir, "train_index.npy"))
    i_val = np.load(os.path.join(out_dir, "test_index.npy"))
    scaler_config = load_json_file(os.path.join(out_dir, "scaler_config.json"))

    # training parameters
    unit_label_energy = training_config['unit_energy']
    epo = training_config['epo']
    batch_size = training_config['batch_size']
    epostep = training_config['epostep']
    initialize_weights = training_config['initialize_weights']
    learning_rate = training_config['learning_rate']
    use_callbacks = training_config['callbacks']

    # Load data.
    data_dir = os.path.dirname(out_dir)
    xyz = read_xyz_file(os.path.join(data_dir, "geometries.xyz"))
    coords = [np.array(x[1]) for x in xyz]
    atoms = [np.array([global_proton_dict[at] for at in x[0]]) for x in xyz]
    range_indices = [define_adjacency_from_distance(coordinates_to_distancematrix(x), max_distance=4)[1] for x in coords]
    y = load_json_file(os.path.join(data_dir, "energies.json"))
    y = np.array(y)

    # Fit stats dir
    dir_save = os.path.join(out_dir, "fit_stats")
    os.makedirs(dir_save, exist_ok=True)

    # cbks,Learning rate schedule
    cbks = []
    for x in use_callbacks:
        if isinstance(x, dict):
            # tf.keras.utils.get_registered_object()
            cb = tf.keras.utils.deserialize_keras_object(x)
            cbks.append(cb(**x["config"]))

    # Make Model
    # Only works for Energy model here
    assert model_config["class_name"] == "SchnetEnergy", "Training script only for SchnetEnergy"
    out_model = SchnetEnergy(**model_config["config"])

    # Look for loading weights
    if not initialize_weights:
        out_model.load_weights(os.path.join(out_dir, "model_weights.h5"))
        print("Info: Load old weights at:", os.path.join(out_dir, "model_weights.h5"))
    else:
        print("Info: Making new initialized weights.")

    # Recalculate standardization
    scaler = EnergyStandardScaler(**scaler_config["config"])
    scaler.fit(x=None, y=y[i_train])
    _, y1 = scaler.transform(x=None, y=y)

    # Train Test split
    xtrain = [
        ragged_tensor_from_nested_numpy([atoms[i] for i in i_train]),
        ragged_tensor_from_nested_numpy([coords[i] for i in i_train]),
        ragged_tensor_from_nested_numpy([range_indices[i] for i in i_train])
    ]
    xval = [
        ragged_tensor_from_nested_numpy([atoms[i] for i in i_val]),
        ragged_tensor_from_nested_numpy([coords[i] for i in i_val]),
        ragged_tensor_from_nested_numpy([range_indices[i] for i in i_val])
    ]
    ytrain = y1[i_train]
    yval = y1[i_val]

    # Compile model
    # This is only for metric to without std.
    scaled_metric = ScaledMeanAbsoluteError(scaling_shape=scaler.energy_std.shape)
    scaled_metric.set_scale(scaler.energy_std)
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    lr_metric = get_lr_metric(optimizer)
    out_model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=[scaled_metric, lr_metric, r2_metric])

    scaler.print_params_info()

    out_model.summary()
    print("")
    print("Start fit.")
    hist = out_model.fit(x=xtrain, y=ytrain, epochs=epo, batch_size=batch_size, callbacks=cbks, validation_freq=epostep,
                         validation_data=(xval, yval), verbose=2)
    print("End fit.")
    print("")

    outname = os.path.join(dir_save, "history.json")
    outhist = {a: np.array(b, dtype=np.float64).tolist() for a, b in hist.history.items()}
    with open(outname, 'w') as f:
        json.dump(outhist, f)

    print("Info: Saving auto-scaler to file...")
    scaler.save_weights(os.path.join(out_dir, "scaler_weights.npy"))

    # Plot and Save
    yval_plot = y[i_val]
    ytrain_plot = y[i_train]
    # Convert back scaler
    pval = out_model.predict(xval)
    ptrain = out_model.predict(xtrain)
    _, pval = scaler.inverse_transform(y=pval)
    _, ptrain = scaler.inverse_transform(y=ptrain)

    print("Info: Predicted Energy shape:", ptrain.shape)
    print("Info: Predicted Gradient shape:", ptrain.shape)
    print("Info: Plot fit stats...")

    # Plot
    plot_loss_curves(hist.history['mean_absolute_error'], hist.history['val_mean_absolute_error'],
                     val_step=epostep, save_plot_to_file=True, dir_save=dir_save,
                     filename='fit' + str(i), filetypeout='.png', unit_loss=unit_label_energy, loss_name="MAE",
                     plot_title="Energy")

    plot_scatter_prediction(pval, yval_plot, save_plot_to_file=True, dir_save=dir_save, filename='fit' + str(i),
                            filetypeout='.png', unit_actual=unit_label_energy, unit_predicted=unit_label_energy,
                            plot_title="Prediction")

    plot_learning_curve(hist.history['lr'], filename='fit' + str(i), dir_save=dir_save)

    # Safe fitting Error MAE
    pval = out_model.predict(xval)
    ptrain = out_model.predict(xtrain)
    _, pval = scaler.inverse_transform(y=pval)
    _, ptrain = scaler.inverse_transform(y=ptrain)

    error_val = np.mean(np.abs(pval - y[i_val]))
    error_train = np.mean(np.abs(ptrain - y[i_train]))
    print("error_val:", error_val)
    print("error_train:", error_train)
    error_dict = {"train": error_train.tolist(), "valid": error_val.tolist()}
    save_json_file(error_dict, os.path.join(out_dir, "fit_error.json"))

    print("Info: Saving model to file...")
    out_model.precomputed_features = False
    out_model.save_weights(os.path.join(out_dir, "model_weights.h5"))
    out_model.save(os.path.join(out_dir, "model_tf"))

    return error_val


if __name__ == "__main__":
    print("Training Model: ", args['filepath'])
    print("Network instance: ", args['index'])
    out = train_model_energy(args['index'], args['filepath'], args['mode'])

file_std_out.close()
