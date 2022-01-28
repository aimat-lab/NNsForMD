import matplotlib as mpl
import numpy as np
import tensorflow as tf
mpl.use('Agg')
import os
import json
import pickle
import sys

import argparse

parser = argparse.ArgumentParser(description='Train a nac model from data, parameters given in a folder')

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
from pyNNsMD.models.mlp_nac2 import NACModel2
from pyNNsMD.utils.data import load_json_file, read_xyz_file
from pyNNsMD.scaler.nac import NACStandardScaler
from pyNNsMD.utils.loss import ScaledMeanAbsoluteError, get_lr_metric, r2_metric, NACphaselessLoss
from pyNNsMD.plots.loss import plot_loss_curves, plot_learning_curve
from pyNNsMD.plots.pred import plot_scatter_prediction
from pyNNsMD.plots.error import plot_error_vec_mean, plot_error_vec_max


def train_model_nac(i=0, out_dir=None, mode='training'):
    """
    Train NAC model. Uses precomputed feature and model representation.

    Args:
        i (int, optional): Model index. The default is 0.
        out_dir (str, optional): Direcotry for fit output. The default is None.
        mode (str, optional): Fitmode to take from hyperparameters. The default is 'training'.

    Raises:
        ValueError: Wrong input shape.

    Returns:
        error_val (list): Validation error for NAC.
    """
    i = int(i)
    # Load everything from folder
    training_config = load_json_file(os.path.join(out_dir, mode + "_config.json"))
    model_config = load_json_file(os.path.join(out_dir, "model_config.json"))
    i_train = np.load(os.path.join(out_dir, "train_index.npy"))
    i_val = np.load(os.path.join(out_dir, "test_index.npy"))
    scaler_config = load_json_file(os.path.join(out_dir, "scaler_config.json"))

    # Model
    num_outstates = int(model_config["config"]['states'])
    num_atoms = int(model_config["config"]['atoms'])
    unit_label_nac = training_config['unit_nac']
    phase_less_loss = training_config['phase_less_loss']
    epo = training_config['epo']
    batch_size = training_config['batch_size']
    epostep = training_config['epostep']
    pre_epo = training_config['pre_epo']
    initialize_weights = training_config['initialize_weights']
    learning_rate = training_config['learning_rate']
    use_callbacks = list(training_config["callbacks"])

    # Data Check here:
    data_dir = os.path.dirname(out_dir)
    xyz = read_xyz_file(os.path.join(data_dir, "geometries.xyz"))
    x = np.array([x[1] for x in xyz])
    if x.shape[1] != num_atoms:
        raise ValueError(f"Mismatch Shape between {x.shape} model and data {num_atoms}")
    y_in = np.load(os.path.join(data_dir, "couplings.npy"))
    print("INFO: Shape of y", y_in.shape)

    # Set stat dir
    dir_save = os.path.join(out_dir, "fit_stats")
    os.makedirs(dir_save, exist_ok=True)

    # cbks,Learning rate schedule
    cbks = []
    for x in use_callbacks:
        if isinstance(x, dict):
            # tf.keras.utils.get_registered_object()
            cb = tf.keras.utils.deserialize_keras_object(x)
            cbks.append(cb(**x["config"]))

    # Make all Models
    out_model = NACModel2(**model_config["config"])
    out_model.precomputed_features = True

    npeps = np.finfo(float).eps
    if not initialize_weights:
        out_model.load_weights(os.path.join(out_dir, "model_weights.h5"))
        print("Info: Load old weights at:", os.path.join(out_dir, "model_weights.h5"))
        print("Info: Transferring weights...")
    else:
        print("Info: Making new initialized weights..")

    scaler = NACStandardScaler(**scaler_config["config"])
    scaler.fit(x, y_in)
    x_rescale, y = scaler.transform(x=x, y=y_in)

    # Calculate features
    feat_x, feat_grad = out_model.precompute_feature_in_chunks(x_rescale, batch_size=batch_size)

    xtrain = [feat_x[i_train], feat_grad[i_train]]
    ytrain = y[i_train]
    xval = [feat_x[i_val], feat_grad[i_val]]
    yval = y[i_val]

    # Set Scaling
    scaled_metric = ScaledMeanAbsoluteError(scaling_shape=scaler.nac_std.shape)
    scaled_metric.set_scale(scaler.nac_std)
    scaler.print_params_info()

    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    lr_metric = get_lr_metric(optimizer)
    out_model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=[scaled_metric, lr_metric, r2_metric])

    # Pre -fit
    print("")
    print("Start fit.")
    if pre_epo > 0:
        print("Start Pre-fit without phaseless-loss.")
        print("Used loss:", out_model.loss)
        out_model.summary()
        out_model.fit(x=xtrain, y=ytrain, epochs=pre_epo, batch_size=batch_size, validation_freq=epostep,
                      validation_data=(xval, yval), verbose=2)
        print("End fit.")
        print("")

    print("Start fit.")
    if phase_less_loss:
        print("Recompiling with phaseless loss.")
        out_model.compile(
            loss=NACphaselessLoss(number_state=num_outstates, shape_nac=(num_atoms, 3), name='phaseless_loss'),
            optimizer=optimizer,
            metrics=[scaled_metric, lr_metric, r2_metric])
        print("Used loss:", out_model.loss)

    out_model.summary()
    hist = out_model.fit(x=xtrain, y=ytrain, epochs=epo, batch_size=batch_size, callbacks=cbks, validation_freq=epostep,
                         validation_data=(xval, yval), verbose=2)
    print("End fit.")
    print("")

    print("Info: Saving history...")
    outname = os.path.join(dir_save, "history.json")
    outhist = {a: np.array(b, dtype=np.float64).tolist() for a, b in hist.history.items()}
    with open(outname, 'w') as f:
        json.dump(outhist, f)

    print("Info: Saving auto-scaler to file...")
    scaler.save(os.path.join(out_dir, "scaler_weights.json"))

    yval_plot = y_in[i_val]
    ytrain_plot = y_in[i_train]
    # Revert standard but keep unit conversion
    pval = out_model.predict(xval)
    ptrain = out_model.predict(xtrain)
    _, pval = scaler.inverse_transform(y=pval)
    _, ptrain = scaler.inverse_transform(y=ptrain)

    print("Info: Predicted NAC shape:", ptrain.shape)
    print("Info: Plot fit stats...")

    plot_loss_curves(hist.history['mean_absolute_error'],
                     hist.history['val_mean_absolute_error'],
                     label_curves="NAC",
                     val_step=epostep, save_plot_to_file=True, dir_save=dir_save,
                     filename='fit' + str(i) + "_nac", filetypeout='.png', unit_loss=unit_label_nac,
                     loss_name="MAE",
                     plot_title="NAC")

    plot_learning_curve(hist.history['lr'], filename='fit' + str(i), dir_save=dir_save)

    plot_scatter_prediction(pval, yval_plot, save_plot_to_file=True, dir_save=dir_save,
                            filename='fit' + str(i) + "_nac",
                            filetypeout='.png', unit_actual=unit_label_nac, unit_predicted=unit_label_nac,
                            plot_title="Prediction NAC")

    plot_error_vec_mean([pval, ptrain], [yval_plot, ytrain_plot],
                        label_curves=["Validation NAC", "Training NAC"], unit_predicted=unit_label_nac,
                        filename='fit' + str(i) + "_nac", dir_save=dir_save, save_plot_to_file=True,
                        filetypeout='.png', x_label='NACs xyz * #atoms * #states ',
                        plot_title="NAC mean error")

    plot_error_vec_max([pval, ptrain], [yval_plot, ytrain_plot],
                       label_curves=["Validation", "Training"],
                       unit_predicted=unit_label_nac, filename='fit' + str(i) + "_nc",
                       dir_save=dir_save, save_plot_to_file=True, filetypeout='.png',
                       x_label='NACs xyz * #atoms * #states ', plot_title="NAC max error")
    # error out
    error_val = None

    print("Info: saving fitting error...")
    # Safe fitting Error MAE
    pval = out_model.predict(xval)
    ptrain = out_model.predict(xtrain)
    _, pval = scaler.inverse_transform(y=pval)
    _, ptrain = scaler.inverse_transform(y=ptrain)
    out_model.precomputed_features = False
    ptrain2 = out_model.predict(x_rescale[i_train])
    ptrain2 = ptrain2 * scaler.nac_std + scaler.nac_mean
    print("Info: MAE between precomputed and full keras model:")
    print("NAC", np.mean(np.abs(ptrain - ptrain2)))
    error_val = np.mean(np.abs(pval - y_in[i_val]))
    error_train = np.mean(np.abs(ptrain - y_in[i_train]))
    print("error_val:", error_val)
    print("error_train:", error_train)
    np.save(os.path.join(out_dir, "fiterr_valid" + '_v%i' % i + ".npy"), error_val)
    np.save(os.path.join(out_dir, "fiterr_train" + '_v%i' % i + ".npy"), error_train)

    # Save Weights
    print("Info: Saving weights...")
    out_model.precomputed_features = False
    out_model.save_weights(os.path.join(out_dir, "model_weights.h5"))
    out_model.save(os.path.join(out_dir, "model_tf"))

    return error_val


if __name__ == "__main__":
    print("Training Model: ", args['filepath'])
    print("Network instance: ", args['index'])
    train_model_nac(args['index'], args['filepath'], args['mode'])

fstdout.close()
