import logging
import time

import numpy as np
import tensorflow as tf


def lr_step_reduction(learning_rate_step=[1e-3, 1e-4, 1e-5], epoch_step_reduction=[500, 1000, 5000],use=None):
    """
    Make learning rate schedule function for step reduction.

    Args:
        learnrate_steps (list, optional): List of learning rates for each step. The default is [1e-3,1e-4,1e-5].
        learnrate_epochs (list, optional): The length of each step to keep learning rate.
                                            The default is [500,1000,5000].

    Returns:
        func: Function that can be used with LearningRateScheduler.
        
    Example:
        lr_schedule_steps = tf.keras.callbacks.LearningRateScheduler(lr_step_reduction)

    """
    learning_rate_abs = np.cumsum(np.array(epoch_step_reduction))

    def lr_out_step(epoch):
        # epo = int(learning_rate_abs[-1])
        learning_rate = float(learning_rate_step[-1])
        le = np.array(learning_rate_abs)
        lr = np.array(learning_rate_step)
        out = np.select(epoch <= le, lr, default=learning_rate)
        return float(out)

    return lr_out_step


@tf.keras.utils.register_keras_serializable(package='pyNNsMD', name='LinearLearningRateScheduler')
class LinearLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    """Callback for linear change of the learning rate.

    This class inherits from tf.keras.callbacks.LearningRateScheduler.
    """

    def __init__(self, learning_rate_start: float = 1e-3, learning_rate_stop: float = 1e-5, epo_min: int = 0,
                 epo: int = 500, verbose: int = 0):
        """Set the parameters for the learning rate scheduler.

        Args:
            learning_rate_start (float): Initial learning rate. Default is 1e-3.
            learning_rate_stop (float): End learning rate. Default is 1e-5.
            epo_min (int): Minimum number of epochs to keep the learning-rate constant before decrease. Default is 0.
            epo (int): Total number of epochs. Default is 500.
            verbose (int): Verbosity. Default is 0.
        """
        super(LinearLearningRateScheduler, self).__init__(schedule=self.schedule_epoch_lr, verbose=verbose)
        self.learning_rate_start = learning_rate_start
        self.learning_rate_stop = learning_rate_stop
        self.epo = epo
        self.epo_min = epo_min

    def schedule_epoch_lr(self, epoch, lr):
        if epoch < self.epo_min:
            out = float(self.learning_rate_start)
        else:
            out = float(self.learning_rate_start - (self.learning_rate_start - self.learning_rate_stop) / (
                    self.epo - self.epo_min) * (epoch - self.epo_min))
        return float(out)

    def get_config(self):
        config = super(LinearLearningRateScheduler, self).get_config()
        config.update({"learning_rate_start": self.learning_rate_start,
                       "learning_rate_stop": self.learning_rate_stop, "epo": self.epo, "epo_min": self.epo_min})
        return config


@tf.keras.utils.register_keras_serializable(package='pyNNsMD', name='LinearWarmupExponentialLearningRateScheduler')
class LinearWarmupExponentialLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    """Callback for linear change of the learning rate.

    This class inherits from tf.keras.callbacks.LearningRateScheduler.
    """

    def __init__(self, lr_start: float, decay_gamma: float, epo_warmup: int = 10, lr_min: float = 0.0,
                 verbose: int = 0):
        """Set the parameters for the learning rate scheduler.
        Args:
            lr_start (float): Learning rate at the start of the exp. decay.
            decay_gamma (float): Gamma parameter in the exponential.
            epo_warmup (int): Number of warm-up steps. Default is 10.
            lr_min (float): Minimum learning rate allowed during the decay. Default is 0.0.
            verbose (int): Verbosity. Default is 0.
        """
        self.decay_gamma = decay_gamma
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.epo_warmup = max(epo_warmup, 0)
        self.verbose = verbose
        super(LinearWarmupExponentialLearningRateScheduler, self).__init__(schedule=self.schedule_epoch_lr,
                                                                           verbose=verbose)

    def schedule_epoch_lr(self, epoch, lr):
        """Reduce the learning rate."""
        if epoch < self.epo_warmup:
            new_lr = self.lr_start * epoch / self.epo_warmup + self.lr_min
        elif epoch == self.epo_warmup:
            new_lr = max(self.lr_start, self.lr_min)
        else:
            new_lr = max(self.lr_start * np.exp(-(epoch - self.epo_warmup) / self.decay_gamma), self.lr_min)
        return float(new_lr)

    def get_config(self):
        config = super(LinearWarmupExponentialLearningRateScheduler, self).get_config()
        config.update({"lr_start": self.lr_start, "decay_gamma": self.decay_gamma, "epo_warmup": self.epo_warmup,
                       "lr_min": self.lr_min, "verbose": self.verbose})
        return config


@tf.keras.utils.register_keras_serializable(package='pyNNsMD', name='EarlyStopping')
class EarlyStopping(tf.keras.callbacks.Callback):
    """This Callback does basic monitoring of the learning process.
    
    Provides functionality such as learning rate decay and early stopping with custom logic as opposed to the
    callbacks provided by Keras by default which are generic.
    By André Eberhard https://github.com/patchmeifyoucan
    """

    def __init__(self,
                 max_time=np.Infinity,
                 epochs=np.Infinity,
                 learning_rate_start=1e-3,
                 epostep=1,
                 loss_monitor='val_loss',
                 min_delta=0.00001,
                 patience=100,
                 epomin=0,
                 factor_lr=0.5,
                 learning_rate_stop=0.000001,
                 store_weights=False,
                 restore_weights_on_lr_decay=False,
                 use=None
                 ):
        """Initialize callback instance for early stopping.
        
        Args:     
            max_time (int): Duration in minutes of training, stops training even if number of epochs is not reached yet.
            epochs (int): Number of epochs to train. stops training even if number of max_time is not reached yet.
            learning_rate_start (float): The learning rate for the optimizer.
            epostep (int): Step to check for monitor loss.
            loss_monitor (str): The loss quantity to monitor for early stopping operations.
            min_delta (float): Minimum improvement to reach after 'patience' epochs of training.
            patience (int): Number of epochs to wait before decreasing learning rate by a factor of 'factor'.
            epomin (int): Minimum Number of epochs to run before decreasing learning rate
            factor_lr (float): new_lr = old_lr * factor
            learning_rate_stop (float): Learning rate is not decreased any further after learning_rate_stop is reached.
            store_weights (bool): If True, stores parameters of best run so far when learning rate is decreased.
            restore_weights_on_lr_decay (bool): If True, restores parameters of best run so far when learning rate is
                decreased.
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)
        self.max_time = max_time
        self.epochs = epochs
        self.epostep = epostep
        self.epomin = epomin
        self.use = use

        self.start = None
        self.stopped = False
        self.batch_size = None
        self.batch_size_initial = None
        self.learning_rate_start = learning_rate_start
        self.learning_rate_stop = learning_rate_stop
        self.learning_rate = learning_rate_start  # Begin with lr start

        self.loss_monitor = loss_monitor
        self.min_delta = min_delta
        self.factor_lr = factor_lr
        self.patience = patience
        self.restore_weights_on_lr_decay = restore_weights_on_lr_decay
        self.store_weights = store_weights

        self.best_weights = None
        self.current_epoch = 0
        self.current_minutes = 0
        self.epochs_without_improvement = 0
        self.best_loss = np.Infinity

    def _reset_weights(self):
        if self.best_weights is None:
            return

        self.logger.info("resetting model weights")
        self.model.set_weights(self.best_weights)

    def _decrease_learning_rate(self):
        old_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        new_lr = old_lr * self.factor_lr
        if new_lr < self.learning_rate_stop:
            self.logger.info(
                f"Reached learning rate {old_lr:.8f} below acceptable {self.learning_rate_stop:.8f} without improvement")
            self.model.stop_training = True
            self.stopped = True
            new_lr = self.learning_rate_stop

        self.logger.info(f"setting learning rate from {old_lr:.8f} to {new_lr:.8f}")
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        self.learning_rate = new_lr

        if self.restore_weights_on_lr_decay is True:
            self._reset_weights()

    def _check_time(self):
        self.current_minutes = np.round((time.time() - self.start) / 60).astype("int")
        self.logger.info(f"network trained for {self.current_minutes}/{self.max_time} minutes.")
        if self.current_minutes < self.max_time:
            return

        self.logger.info(f"network trained for {self.current_minutes} minutes. stopping.")
        self.model.stop_training = True
        self.stopped = True

    def _check_loss(self, logs):
        current_loss = logs[self.loss_monitor]

        if current_loss < self.best_loss and self.best_loss - current_loss > self.min_delta:
            diff = self.best_loss - current_loss
            self.logger.info(
                f"{self.loss_monitor} improved by {diff:.6f} from {self.best_loss:.6f} to {current_loss:.6f}.")
            self.best_loss = current_loss
            if self.store_weights:
                self.best_weights = self.model.get_weights()
            self.epochs_without_improvement = 0
            return

        self.epochs_without_improvement += self.epostep
        if self.epochs_without_improvement < self.patience:
            self.logger.info(f"loss did not improve for {self.epochs_without_improvement} epochs.")
            return

        self.logger.info(f"loss did not improve for max epochs {self.patience}.")
        self.epochs_without_improvement = 0
        self._decrease_learning_rate()

    def on_train_begin(self, logs=None):

        if self.start is None:
            self.start = time.time()
            self.model.summary()
            return

        self._check_time()

    def on_train_end(self, logs=None):
        if self.stopped:
            self._reset_weights()

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1
        if self.current_epoch % self.epostep == 0:
            self._check_time()
            if self.current_epoch > self.epomin:
                self._check_loss(logs)
            loss_diff = logs['val_loss'] - logs['loss']
            self.logger.info(f'current loss_diff: {loss_diff:.6f}')

        if self.current_epoch > self.epochs:
            self.model.stop_training = True
            self.stopped = True

    def get_config(self):
        config = super(EarlyStopping, self).get_config()
        config.update({
            "max_time": self.max_time,
            "epochs": self.epochs,
            "learning_rate_start": self.learning_rate_start,
            "epostep": self.epostep,
            "loss_monitor": self.loss_monitor,
            "min_delta": self.min_delta,
            "patience": self.patience,
            "epomin": self.epomin,
            "factor_lr": self.factor_lr,
            "learning_rate_stop": self.learning_rate_stop,
            "store_weights": self.store_weights,
            "restore_weights_on_lr_decay": self.restore_weights_on_lr_decay,
            "use": self.use
        })
        return config