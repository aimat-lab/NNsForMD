import os
import sys
import logging
import importlib
import tensorflow as tf

from pyNNsMD.utils.data import save_json_file, load_json_file
from pyNNsMD.src.fit import fit_model_by_script
from pyNNsMD.scaler.base import SaclerBase

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
        self._directory = directory
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
        return  str(type(class_object)).replace("'>", "").split(".")[-1]

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

    def save(self, save_model: bool = True, save_weights: bool = True, save_scaler: bool  = True):
        """Save models, scaler and hyper-parameter into class folder."""

        directory = os.path.realpath(self._directory)
        os.makedirs(directory, exist_ok=True)

        for i in range(self._number_models):
            model_path = os.path.join(directory, "model_v%s" % i)
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
            self.logger.warning("Recreating scaler directly from file is not implemented yet.")

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
            model_path = os.path.join(directory, "model_v%s" % i)

            self._models[i] = self._load_single_model(model_path=model_path, i=i, load_model=load_model)
            self._scalers[i] = self._load_single_scaler(model_path=model_path, i=i, load_scaler=load_scaler)

        return self

    def data(self, geometries=None, forces=None, energies=None, couplings=None):
        pass

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
            training_scripts (list): List of training scripts
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
        proclist = []
        for i, fit_script in enumerate(training_scripts):
            proclist.append(self._fit_single_model(i, fit_script, gpu_dist[i], proc_async, fit_mode))

        # Wait for fits
        if proc_async:
            self.logger.info("Fits submitted, waiting...")
            # Wait for models to finish
            for proc in proclist:
                if proc is None:
                    self.logger.warning("No valid process to wait for.")
                else:
                    proc.wait()

        # Look for fiterror in folder
        self.logger.info("Searching Folder for fit results...")
        self.load()

        return None

    # @staticmethod
    # def _predict_model_list(x_list, model_list, batch_size_list):
    #     out = [model_list[i].predict(x_list[i], batch_size=batch_size_list[i]) for i in range(len(model_list))]
    #     return out
    #
    # def _predict_models(self, name, x):
    #     # Check type with first hyper
    #     model_type = self._models_hyper[name][0]['general']['model_type']
    #     x_scaled = [self._models_scaler[name][i].transform(x=x)[0] for i in range(self._addNN)]
    #     temp = self._predict_model_list(x_scaled, self._models[name],
    #                                     [self._models_hyper[name][i]['predict']['batch_size_predict'] for i in
    #                                      range(self._addNN)])
    #     out = [self._models_scaler[name][i].inverse_transform(y=temp[i])[1] for i in range(self._addNN)]
    #     return predict_uncertainty(model_type, out, self._addNN)
    #
    # def predict(self, x):
    #     """
    #     Prediction for all models available. Prediction is slower but works on large data.
    #
    #     Args:
    #         x (np.array,list, dict):    Coordinates in Angstroem of shape (batch,Atoms,3)
    #                                     Or a suitable list of geometric input.
    #                                     If models require different x please provide dict matching model name.
    #
    #     Returns:
    #         result (dict): All model predictions: {'energy_gradient' : [np.aray,np.array] , 'nac' : np.array , ..}.
    #         error (dict): Error estimate for each value: {'energy_gradient' : [np.aray,np.array] ,
    #                       'nac' : np.array , ..}.
    #
    #     """
    #     result = {}
    #     error = {}
    #     for name in self._models.keys():
    #         if isinstance(x, dict):
    #             x_model = x[name]
    #         else:
    #             x_model = x
    #         temp = self._predict_models(name, x_model)
    #         result[name] = temp[0]
    #         error[name] = temp[1]
    #
    #     return result, error
    #
    # @tf.function
    # def _call_model_list(self, x_list, model_list):
    #     out = [model_list[i](x_list[i], training=False) for i in range(len(model_list))]
    #     return out
    #
    # def _call_models(self, name, x):
    #     # Check type with first hyper
    #     model_type = self._models_hyper[name][0]['general']['model_type']
    #     x_scaled = [self._models_scaler[name][i].transform(x=x)[0] for i in range(self._addNN)]
    #     x_res = [tf.convert_to_tensor(xs, dtype=tf.float32) for xs in x_scaled]
    #     temp = self._call_model_list(x_res, self._models[name])
    #     temp = [unpack_convert_y_to_numpy(model_type, xout) for xout in temp]
    #     out = [self._models_scaler[name][i].inverse_transform(y=temp[i])[1] for i in range(self._addNN)]
    #     return predict_uncertainty(model_type, out, self._addNN)
    #
    # def call(self, x):
    #     """
    #     Faster prediction without looping batches. Requires single small batch (batch, Atoms,3) that fit into memory.
    #
    #     Args:
    #         x (np.array):   Coordinates in Angstroem of shape (batch,Atoms,3)
    #                         Or a suitable list of geometric input.
    #                         If models require different x please provide dict matching model name.
    #
    #     Returns:
    #         result (dict): All model predictions: {'energy_gradient' : [np.array,np.array] , 'nac' : np.array , ..}.
    #         error (dict): Error estimate for each value: {'energy_gradient' : [np.array,np.array] ,
    #                       'nac' : np.array , ..}.
    #
    #     """
    #     result = {}
    #     error = {}
    #     for name in self._models.keys():
    #         if isinstance(x, dict):
    #             x_model = x[name]
    #         else:
    #             x_model = x
    #         temp = self._call_models(name, x_model)
    #         result[name] = temp[0]
    #         error[name] = temp[1]
    #
    #     return result, error
