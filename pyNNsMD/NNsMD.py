import os
import sys
import numpy as np
import logging
import importlib
import tensorflow as tf

from pyNNsMD.utils.data import save_json_file, load_json_file, write_list_to_xyz_file
from pyNNsMD.src.fit import fit_model_by_script
from pyNNsMD.scaler.base import SaclerBase
from sklearn.model_selection import KFold

logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


class NeuralNetEnsemble:
    """Main class NeuralNetPes that keeps multiple keras models and manages training and prediction.

    Main class for neural network (NN) container to provide multiple NN instances.

    Enables uncertainty estimate as well as training and prediction of arbitrary tf.keras models for potentials
    plus gradients and couplings.
    The python class is supposed to allow parallel training.

    The individual model types are further stored to file in the directory specified in initialization.

    The information and models are passed via file to training scripts.

    ...:

    """

    def __init__(self, directory: str, number_models: int = 2, logger=None):
        r"""Initialize empty :obj:`NeuralNetPes` instance.

        Args:
            directory (str): Directory where models, hyper-parameter, logs and fit results are stored.
            number_models (int, optional): Number of NN instances to create for error estimate. The default is 2.
            logger: Logger for this class.
        """
        self.logger = module_logger if logger is None else logger
        # self.logger = logging.getLogger(type(self).__name__)
        self.logger.info("Operating System: %s" % sys.platform)
        self.logger.info("Tested for tf-gpu= 2.3 This tf version: %s" % tf.__version__)
        self.logger.info("Models implemented:")

        # General.
        self._directory = os.path.realpath(directory)
        self._number_models = number_models

        # Private members.
        self._models = []
        self._scalers = []

    def _create_single_model(self, kw, i):
        # The module location could be inferred from keras path or module system using '>'
        # For now keep at extra argument that models must store in their config.
        if kw is None:
            # Must have model.
            raise ValueError("Expected model kwargs, got `None` instead.")

        if isinstance(kw, tf.keras.Model):
            self.logger.info("Got `keras.Model` for model index i" % i)
            return kw

        if not isinstance(kw, dict):
            raise ValueError("Please supply a model or a dictionary for `create`.")

        if "model_module" in kw["config"]:
            if "class_name" not in kw:
                raise ValueError("Requires information about the class for model %s" % i)
            if not isinstance(kw["class_name"], str):
                raise ValueError("Requires class name to be string but got %s" % kw["class_name"])

            class_name = kw["class_name"].split(">")[-1]
            try:
                make_class = getattr(importlib.import_module("pyNNsMD.models.%s" % kw["config"]["model_module"]),
                                     class_name)
            except ModuleNotFoundError:
                raise NotImplementedError(
                    "Unknown model identifier %s for a model in pyNNsMD.models" % kw["model_class"])

            return make_class(**kw["config"])

        if "class_name" in kw:
            return tf.keras.utils.deserialize_keras_object(kw["class_name"])(**kw["config"])

        raise ValueError("Could not make model from %s" % kw)

    def _create_single_scaler(self, kw, i):
        if kw is None:
            self.logger.warning("Expected scaler kwargs, got `None` instead. Not using scaler.")
            return None

        if isinstance(kw, SaclerBase):
            self.logger.info("Got scaler for model index i" % i)
            return kw

        if not isinstance(kw, dict):
            raise ValueError("Please supply a scaler or a dictionary for `create`.")

        if "scaler_module" in kw["config"]:
            if "class_name" not in kw:
                raise ValueError("Requires information about the class for scaler %s" % i)
            if not isinstance(kw["class_name"], str):
                raise ValueError("Requires class name to be string but got %s" % kw["class_name"])

            class_name = kw["class_name"].split(">")[-1]
            try:
                make_class = getattr(importlib.import_module("pyNNsMD.scaler.%s" % kw["config"]["scaler_module"]),
                                     class_name)
            except ModuleNotFoundError:
                raise NotImplementedError(
                    "Unknown scaler identifier %s for a scaler in pyNNsMD.scaler" % kw["model_class"])

            return make_class(**kw["config"])

        raise ValueError("Could not make model from %s" % kw)

    def create(self, models: list, scalers: list):
        """Initialize and build a list of keras models. Missing hyper-parameter are filled from default.

        Args:
            models (list): Dictionary with model hyper-parameter.
                In each dictionary, information of module and model class must be provided.
            scalers (list):

        Returns:
            self
        """
        if len(models) != self._number_models:
            raise ValueError("Wrong number of model kwargs in create, expected %s" % self._number_models)

        self._models = [None]*self._number_models
        for i, kw in enumerate(models):
            self._models[i] = self._create_single_model(kw, i)

        self._scalers = [None]*self._number_models
        for i, kw in enumerate(scalers):
            self._scalers[i] = self._create_single_scaler(kw, i)

        self.logger.info("Models and Scaler created. Must be save before calling fit.")
        return self

    def _save_single_model(self, model, i, model_path, save_weights, save_model):

        model_serialization = {"class_name": self._get_name_of_class(model),
                               "config": model.get_config()}
        save_json_file(model_serialization, os.path.join(model_path, "model_config.json"))

        if save_weights:
            if not hasattr(model, "save_weights") or model is None:
                raise AttributeError("Model is not a keras model with `save_weights()` defined for %s" % i)
            model.save_weights(os.path.join(model_path, "model_weights.h5"))

        if save_model:
            if not hasattr(model, "save") or model is None:
                raise AttributeError("Model is not a keras model with `save()` defined for %s" % i)
            model.save(os.path.join(model_path, "model_tf"))

    @staticmethod
    def _get_name_of_class(class_object):
        return str(type(class_object)).replace("'>", "").split(".")[-1]

    def _get_model_path(self, i):
        return os.path.join(self._directory, "model_v%s" % i)

    def _save_single_scaler(self, scaler, i, model_path, save_scaler, save_weights):

        if scaler is None:
            self.logger.warning("No scaler for model %i saved to file" % i)
            return None

        scaler_serialization = {"class_name": self._get_name_of_class(scaler),
                                "config": scaler.get_config()}
        save_json_file(scaler_serialization, os.path.join(model_path, "scaler_config.json"))

        if save_weights:
            if not hasattr(scaler, "save_weights"):
                raise AttributeError("Scaler must implement `save_weights()` which is not defined for model %s" % i)
            scaler.save_weights(os.path.join(model_path, "scaler_weights.npy"))

        if save_scaler:
            if not hasattr(scaler, "save"):
                raise AttributeError("Scaler must implement `save()` which is not defined for %s" % i)
            scaler.save(os.path.join(model_path, "scaler_class"))

    def save(self, save_model: bool = True, save_weights: bool = True, save_scaler: bool = True):
        """Save models, scaler and hyper-parameter into class folder."""

        directory = os.path.realpath(self._directory)
        os.makedirs(directory, exist_ok=True)

        for i in range(self._number_models):
            model_path = self._get_model_path(i)
            os.makedirs(model_path, exist_ok=True)

            self._save_single_model(self._models[i], i, model_path, save_weights=save_weights, save_model=save_model)
            self._save_single_scaler(self._scalers[i], i, model_path,
                                     save_weights=save_weights, save_scaler=save_scaler)

        return self

    def _load_single_model(self, model_path, i, load_model: bool = True):

        _models = None
        # Load the separate model kwargs.
        _models_hyper = load_json_file(os.path.join(model_path, "model_config.json"))
        if _models_hyper is None:
            self.logger.error("Loaded empty model config for model %s" % i)

        # Load model
        if load_model:
            _models = tf.keras.models.load_model(os.path.join(model_path, "model_tf"))

        if not load_model:
            self.logger.warning("Recreating model from config and loading weights...")
            _models = self._create_single_model(_models_hyper, i)
            _models.load_weights(os.path.join(model_path, "model_weights.h5"))

        return _models

    def _load_single_scaler(self, model_path, i, load_scaler):
        _scaler = None

        if not os.path.exists(os.path.join(model_path, "scaler_config.json")):
            self.logger.error("Scaler for model %i was not defined" % i)
            return None
        # Load the separate scaler kwargs.
        _scaler_config = load_json_file(os.path.join(model_path, "scaler_config.json"))

        if _scaler_config is None:
            self.logger.error("Can not load scaler for model %s" % i)
            return None

        if load_scaler:
            self.logger.warning("Loading scaler directly from file is not implemented yet.")

        self.logger.info("Recreating scaler with weights.")
        _scaler = self._create_single_scaler(_scaler_config, i)
        _scaler.load_weights(os.path.join(model_path, "scaler_weights.npy"))

        return _scaler

    def load(self, load_model: bool = True, load_scaler: bool = True):
        """Load model from file that are stored in class folder.
        
        The tensorflow.keras.model is not loaded itself but created new from hyperparameters.

        Raises:
            FileNotFoundError: If Directory not found.

        Returns:
            list: Loaded Models.

        """
        directory = os.path.realpath(self._directory)
        if not os.path.exists(directory):
            raise FileNotFoundError("Can not find file directory %s for this class" % directory)

        _models_hyper = [None]*self._number_models
        _scaler_hyper = [None]*self._number_models

        for i in range(self._number_models):
            model_path = self._get_model_path(i)

            self._models[i] = self._load_single_model(model_path=model_path, i=i, load_model=load_model)
            self._scalers[i] = self._load_single_scaler(model_path=model_path, i=i, load_scaler=load_scaler)

        return self

    @staticmethod
    def _make_nested_list(in_array):
        if isinstance(in_array, np.ndarray):
            return in_array.tolist()
        elif isinstance(in_array, list):
            return [x.tolist() if isinstance(x, np.ndarray) else x for x in in_array]
        return in_array

    def data(self, atoms: list = None, geometries: list = None, forces: list = None, energies: list = None,
             couplings: list = None):
        kwargs = dict(locals())
        kwargs.pop("self")
        dir_path = self._directory
        data_length = [len(values) for key, values in kwargs.items() if values is not None]
        if len(data_length) == 0:
            self.logger.warning("Received no data to safe.")
            return
        if len(set(data_length)) > 1:
            raise ValueError("Received different data length for %s" % data_length)

        if atoms is not None and geometries is not None:
            write_list_to_xyz_file(os.path.join(dir_path, "geometries.xyz"), [x for x in zip(atoms, geometries)])
        if energies is not None:
            save_json_file(self._make_nested_list(energies), os.path.join(dir_path, "energies.json"))
        if forces is not None:
            save_json_file(self._make_nested_list(forces), os.path.join(dir_path, "forces.json"))
        if couplings is not None:
            save_json_file(self._make_nested_list(couplings), os.path.join(dir_path, "couplings.json"))

    def train_test_split(self, dataset_size, n_splits: int = 5, shuffle: bool = True, random_state: int = None):
        if n_splits < self._number_models:
            raise ValueError("Number of splits must be at least number of model but got %s" % n_splits)

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        train_indices = []
        test_indices = []
        i = 0
        for train_index, test_index in kf.split(np.expand_dims(np.arange(dataset_size), axis=-1)):
            if i >= self._number_models:
                break
            np.save(os.path.join(self._get_model_path(i), "train_index.npy"), train_index)
            np.save(os.path.join(self._get_model_path(i), "test_index.npy"), test_index)
            i = i + 1
            train_indices.append(np.array(train_index))
            test_indices.append(np.array(test_index))

        return train_indices, test_indices

    def train_test_indices(self, train: list, test: list):
        if len(train) != self._number_models:
            raise ValueError("Number of indices must match models %s" % self._number_models)
        if len(test) != self._number_models:
            raise ValueError("Number of indices must match models %s" % self._number_models)

        for i, (train_index, test_index) in enumerate(zip(train, test)):
            np.save(os.path.join(self._get_model_path(i), "train_index.npy"), train_index)
            np.save(os.path.join(self._get_model_path(i), "test_index.npy"), test_index)

    def training(self, training_hyper: list, fit_mode: str = "training"):
        if len(training_hyper) != self._number_models:
            raise ValueError("Training configs must match number of models but got %s" % len(training_hyper))
        if fit_mode in ["scaler", "model"]:
            raise ValueError("Training config can not be scaler or model, please rename.")
        for i, x in enumerate(training_hyper):
            save_json_file(x, os.path.join(self._get_model_path(i), fit_mode + "_config.json"))

    def _fit_single_model(self, i, training_script, gpu, proc_async, fit_mode):

        proc = fit_model_by_script(i, training_script, gpu,
                                   os.path.join(self._directory, "model_v%s" % i),
                                   fit_mode, proc_async)
        self.logger.info(f"Submitted training for models {training_script}")
        return proc

    def fit(self, training_scripts: list, gpu_dist: list = None, proc_async=True, fit_mode="training"):
        """Fit NN to data. Model weights and hyper parameters must always saved to file before fit.

        The fit routine calls training scripts on the data_folder in parallel.
        The type of execution is found in src.fit with the training src.training_ scripts.

        Args:
            training_scripts (list): List of training scripts for each model.
            gpu_dist (list, optional): List of GPUs for each NN. Default is [].
            proc_async (bool, optional): Try to run parallel. Default is True.
            fit_mode (str, optional):  Whether to do 'training' or 'retraining' the existing model in
                hyperparameter category. Default is 'training'.
                In principle every reasonable category can be created in hyperparameters.

        Returns:
            list: Fitting Error.
        """
        if gpu_dist is None:
            gpu_dist = [-1 for _ in range(self._number_models)]
        if len(gpu_dist) != self._number_models:
            raise ValueError("GPU distribution must be the same number of models.")
        if len(training_scripts) != self._number_models:
            raise ValueError("Training scripts must be the same number of models.")

        # Fitting
        proc_list = []
        for i, fit_script in enumerate(training_scripts):
            proc_list.append(self._fit_single_model(i, fit_script, gpu_dist[i], proc_async, fit_mode))

        # Wait for fits
        if proc_async:
            self.logger.info("Fits submitted, waiting...")
            # Wait for models to finish
            for proc in proc_list:
                if proc is None:
                    self.logger.warning("No valid process to wait for.")
                else:
                    proc.wait()

        # Look for fit-error in folder
        self.logger.info("Searching Folder for fit results...")
        self.load()

        # We must check if fit was successful
        fit_error = []
        for i in range(self._number_models):
            fit_error_path = os.path.join(self._get_model_path(i), "fit_error.json")
            if not os.path.exists(fit_error_path):
                self.logger.error("Fit %s was not successful, could not find `fit_error.json`, check logfile." % i)
                fit_error.append(None)
            else:
                fit_error.append(load_json_file(fit_error_path))

        return fit_error

    def predict(self, x, **kwargs):
        y_list = []
        for i, (model, scaler) in enumerate(zip(self._models, self._scalers)):
            if scaler is not None:
                x, _ = scaler.inverse_transform(x=x, y=None)
            if hasattr(model, "to_tensor_input"):
                x = model.to_tensor_input(x)
            y = model.predict(x, **kwargs)
            if scaler is not None:
                _, y = scaler.inverse_transform(x=x, y=y)
            y_list.append(y)
        return y_list

    def __getitem__(self, item):
        return self._models[item]