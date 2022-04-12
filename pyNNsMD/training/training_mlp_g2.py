import matplotlib as mpl
import numpy as np
import tensorflow as tf

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


fstdout = open(os.path.join(args['filepath'], "fitlog.txt"), 'w')
sys.stderr = fstdout
sys.stdout = fstdout

print("Input argpars:", args)

from pyNNsMD.src.device import set_gpu

set_gpu([int(args['gpus'])])
print("Logic Devices:", tf.config.experimental.list_logical_devices('GPU'))

import pyNNsMD.utils.callbacks
import pyNNsMD.utils.activ
from pyNNsMD.models.mlp_g2 import GradientModel2
from pyNNsMD.scaler.energy import GradientStandardScaler
from pyNNsMD.utils.data import load_json_file, read_xyz_file, save_json_file
from pyNNsMD.utils.loss import get_lr_metric, ScaledMeanAbsoluteError, r2_metric
from pyNNsMD.plots.loss import plot_loss_curves, plot_learning_curve
from pyNNsMD.plots.pred import plot_scatter_prediction
from pyNNsMD.plots.error import plot_error_vec_mean, plot_error_vec_max


def train_model_energy_gradient(i=0, out_dir=None, mode='training'):
    """Train an energy plus gradient model. Uses precomputed feature and model representation.

    Args:
        i (int, optional): Model index. The default is 0.
        out_dir (str, optional): Direcotry for fit output. The default is None.
        mode (str, optional): Fitmode to take from hyperparameters. The default is 'training'.

    Raises:
        ValueError: Wrong input shape.

    Returns:
        error_val (list): Validation error for (energy,gradient).

    """
    i = int(i)
    # Load everything from folder
    training_config = load_json_file(os.path.join(out_dir, mode+"_config.json"))
    model_config = load_json_file(os.path.join(out_dir, "model_config.json"))
    i_train = np.load(os.path.join(out_dir, "train_index.npy"))
    i_val = np.load(os.path.join(out_dir, "test_index.npy"))
    scaler_config = load_json_file(os.path.join(out_dir, "scaler_config.json"))

    # From Config.
    num_atoms = int(model_config["config"]["atoms"])
    unit_label_grad = training_config['unit_gradient']
    epo = training_config['epo']
    batch_size = training_config['batch_size']
    epostep = training_config['epostep']
    initialize_weights = training_config['initialize_weights']
    learning_rate = training_config['learning_rate']
    use_callbacks = list(training_config["callbacks"])

    # Load data.
    data_dir = os.path.dirname(out_dir)
    xyz = read_xyz_file(os.path.join(data_dir, "geometries.xyz"))
    x = np.array([x[1] for x in xyz])
    if x.shape[1] != num_atoms:
        raise ValueError(f"Mismatch Shape between {x.shape} model and data {num_atoms}")
    y = np.array(load_json_file(os.path.join(data_dir, "forces.json")))
    print("INFO: Shape of y", y.shape)

    # Fit stats dir
    dir_save = os.path.join(out_dir, "fit_stats")
    os.makedirs(dir_save, exist_ok=True)

    # cbks,Learning rate schedule
    cbks = []
    for x in use_callbacks:
        if isinstance(x, dict):
            # tf.keras.utils.get_registered_object()
            cb = tf.keras.utils.deserialize_keras_object(x)
            cbks.append(cb)

    # Make all Model
    assert model_config["class_name"] == "GradientModel2", "Training script only for GradientModel2"
    out_model = GradientModel2(**model_config["config"])
    out_model.precomputed_features = True

    # Look for loading weights
    npeps = np.finfo(float).eps
    if not initialize_weights:
        out_model.load_weights(os.path.join(out_dir, "model_weights.h5"))
        print("Info: Load old weights at:", os.path.join(out_dir, "model_weights.h5"))
        print("Info: Transferring weights...")
    else:
        print("Info: Making new initialized weights.")

    # Scale x,y
    scaler = GradientStandardScaler(**scaler_config["config"])
    scaler.fit(x[i_train], y[i_train])
    x_rescale, y_rescale = scaler.transform(x, y)
    y1 = y_rescale

    # Model + Model precompute layer +feat
    feat_x, feat_grad = out_model.precompute_feature_in_chunks(x_rescale, batch_size=batch_size)

    # Train Test split
    xtrain = [feat_x[i_train], feat_grad[i_train]]
    ytrain = y1[i_train]
    xval = [feat_x[i_val], feat_grad[i_val]]
    yval = y1[i_val]

    # Setting constant feature normalization
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    lr_metric = get_lr_metric(optimizer)
    mae_force = ScaledMeanAbsoluteError(scaling_shape=scaler.gradient_std.shape)
    mae_force.set_scale(scaler.gradient_std)
    out_model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=[mae_force, lr_metric, r2_metric])

    scaler.print_params_info()

    print("")
    print("Start fit.")
    out_model.summary()
    hist = out_model.fit(x=xtrain, y=ytrain, epochs=epo, batch_size=batch_size,
                         callbacks=cbks, validation_freq=epostep,
                         validation_data=(xval,yval), verbose=2)
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

    print("Info: Predicted Gradient shape:", ptrain.shape)
    print("Info: Plot fit stats...")

    # Plot
    plot_loss_curves(hist.history['mean_absolute_error'],
                     hist.history['val_mean_absolute_error'],
                     label_curves=["force"],
                     val_step=epostep, save_plot_to_file=True, dir_save=dir_save,
                     filename='fit' + str(i), filetypeout='.png', unit_loss=unit_label_grad, loss_name="MAE",
                     plot_title="Force")

    plot_learning_curve(hist.history['lr'], filename='fit' + str(i), dir_save=dir_save)

    plot_scatter_prediction(pval, yval_plot, save_plot_to_file=True, dir_save=dir_save,
                            filename='fit' + str(i) + "_grad",
                            filetypeout='.png', unit_actual=unit_label_grad, unit_predicted=unit_label_grad,
                            plot_title="Prediction Gradient")

    plot_error_vec_mean([pval, ptrain], [yval_plot, ytrain_plot],
                        label_curves=["Validation gradients", "Training Gradients"], unit_predicted=unit_label_grad,
                        filename='fit' + str(i) + "_grad", dir_save=dir_save, save_plot_to_file=True,
                        filetypeout='.png', x_label='Gradients xyz * #atoms * #states ',
                        plot_title="Gradient mean error")

    plot_error_vec_max([pval, ptrain], [yval_plot, ytrain_plot],
                       label_curves=["Validation", "Training"],
                       unit_predicted=unit_label_grad, filename='fit' + str(i) + "_grad",
                       dir_save=dir_save, save_plot_to_file=True, filetypeout='.png',
                       x_label='Gradients xyz * #atoms * #states ', plot_title="Gradient max error")

    error_val = None

    # Safe fitting Error MAE
    pval = out_model.predict(xval)
    ptrain = out_model.predict(xtrain)
    _, pval = scaler.inverse_transform(y=pval)
    _, ptrain = scaler.inverse_transform(y=ptrain)
    out_model.precomputed_features = False
    out_model.output_as_dict = False
    ptrain2 = out_model.predict(x_rescale[i_train])
    _, ptrain2 = scaler.inverse_transform(y=ptrain2)
    print("Info: Max error precomputed and full gradient computation:")
    print("Gradient", np.max(np.abs(ptrain - ptrain2)))
    error_val = np.mean(np.abs(pval - y[i_val]))
    error_train = np.mean(np.abs(ptrain- y[i_train]))
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
    out = train_model_energy_gradient(args['index'], args['filepath'], args['mode'])

fstdout.close()
