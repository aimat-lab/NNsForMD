import json
import os
import sys
import logging
import numpy as np
import importlib
import tensorflow as tf

from pyNNsMD.utils.data import save_json_file, load_json_file
from pyNNsMD.src.fit import fit_model_by_script

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
        self._models_hyper = []
        self._scaler_hyper = []
        self._scaler = []

    def _create_single_model(self, kw, i):
        # The module location could be inferred from keras path or module system using '>'
        # For now keep at extra argument that models must store in their config.
        if kw is None:
            raise ValueError("Expected model kwargs, got None instead.")

        if "model_module" not in kw["config"]:
            raise ValueError("Requires information about the module for model %s" % i)

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

    def create(self, model_kwargs: list, scaler_kwargs: list):
        """Initialize and build a list of keras models. Missing hyper-parameter are filled from default.

        Args:
            model_kwargs (dict): Dictionary with model hyper-parameter.
                In each dictionary, information of module and model class must be provided.

        Returns:
            self
        """
        if len(model_kwargs) != self._number_models:
            raise ValueError("Wrong number of model kwargs in create, expected %s" % self._number_models)

        self._models = [None]*self._number_models
        self._models_hyper = [None]*self._number_models
        for i, kw in enumerate(model_kwargs):
            self._models[i] = self._create_single_model(kw, i)
            self._models_hyper[i] = kw

        self._scaler = [None]*self._number_models
        self._scaler_hyper = [None]*self._number_models
        for i, kw in enumerate(scaler_kwargs):
            pass

        self.logger.info("Models and Scaler created. Must be save before calling fit.")
        return self

    def save(self, save_model: bool = True, save_weights: bool = True):
        """Save models, scaler and hyper-parameter into class folder."""

        directory = os.path.realpath(self._directory)
        os.makedirs(directory, exist_ok=True)

        for i in range(self._number_models):
            model_path = os.path.join(directory, "model_v%s" % i)
            os.makedirs(model_path, exist_ok=True)

            # Save separate model config explicitly.
            save_json_file(self._models_hyper[i], os.path.join(model_path, "model_config.json"))

            # Save separate scaler config explicitly.
            save_json_file(self._scaler_hyper[i], os.path.join(model_path, "scaler_config.json"))

            if save_weights:
                if not hasattr(self._models[i], "save_weights") or self._models[i] is None:
                    raise AttributeError("Model is not a keras model with save_weights defined for %s" % i)
                self._models[i].save_weights(os.path.join(model_path, "model_weights.h5"))

            if save_model:
                if not hasattr(self._models[i], "save_weights") or self._models[i] is None:
                    raise AttributeError("Model is not a keras model with save_weights defined for %s" % i)
                self._models[i].save(os.path.join(model_path, "model_tf"))

        return self

    def load(self, load_model: bool = True):
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

        for i in range(self._number_models):
            model_path = os.path.join(directory, "model_v%s" % i)

            # Load the separate model kwargs.
            self._models_hyper[i] = load_json_file(os.path.join(model_path, "model_config.json"))
            if self._models_hyper[i] is None:
                self.logger.error("Loaded empty model config for model %s" % i)

            # Load the separate scaler kwargs.
            self._scaler_hyper[i] = load_json_file(os.path.join(model_path, "scaler_config.json"))
            if self._scaler_hyper[i] is None:
                self.logger.error("Loaded empty scaler config for model %s" % i)

            # Load model
            model_load_success = True
            if load_model:
                tf.keras.models.load_model(os.path.join(model_path, "model_tf"))
            else:
                model_load_success = False

            if not model_load_success:
                self.logger.warning("Recreating model from config and loading weights...")
                self._models[i] = self._create_single_model(self._models_hyper[i], i)
                self._models[i].load_weights(os.path.join(model_path, "model_weights.h5"))

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
            fit_mode (str, optional):  Whether to do 'training' or 'retraining' the existing model in hyperparameter category.
                            Default is 'training'.
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
