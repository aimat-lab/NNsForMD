import numpy as np

from pyNNsMD.scaler.base import SaclerBase


class EnergyStandardScaler(SaclerBase):

    def __init__(self,
                 scaler_module="energy",
                 use_energy_mean=True,
                 use_energy_std=True,
                 use_x_std=False,
                 use_x_mean=False,
                 ):
        super(EnergyStandardScaler, self).__init__()
        self.use_energy_std = use_energy_std
        self.use_energy_mean = use_energy_mean
        self.use_x_std = use_x_std
        self.use_x_mean = use_x_mean

        self.x_mean = np.zeros((1, 1, 1))
        self.x_std = np.ones((1, 1, 1))
        self.energy_mean = np.zeros((1, 1))
        self.energy_std = np.ones((1, 1))

        self._encountered_y_shape = None
        self._encountered_y_std = None
        self.scaler_module = scaler_module

    def transform(self, x=None, y=None):
        x_res = x
        y_res = y
        if x is not None:
            x_res = (x - self.x_mean) / self.x_std
        if y is not None:
            y_res = (y - self.energy_mean) / self.energy_std
        return x_res, y_res

    def inverse_transform(self, x=None, y=None):
        y_res = y
        x_res = x
        if y is not None:
            y_res = y * self.energy_std + self.energy_mean
        if x is not None:
            x_res = x * self.x_std + self.x_mean
        return x_res, y_res

    def fit(self, x=None, y=None):
        npeps = np.finfo(float).eps
        if self.use_x_mean:
            self.x_mean = np.mean(x)
        if self.use_x_std:
            self.x_std = np.std(x) + npeps
        if self.use_energy_mean:
            self.energy_mean = np.mean(y, axis=0, keepdims=True)
        if self.use_energy_std:
            self.energy_std = np.std(y, axis=0, keepdims=True) + npeps

        self._encountered_y_shape = np.array(y.shape)
        self._encountered_y_std = np.std(y, axis=0)

    def fit_transform(self, x=None, y=None):
        self.fit(x=x, y=y)
        return self.transform(x=x, y=y)

    def get_config(self):
        outdict = {
            "scaler_module": self.scaler_module,
            "use_energy_mean": self.use_energy_mean,
            "use_energy_std": self.use_energy_std,
            "use_x_std": self.use_x_std,
            "use_x_mean": self.use_x_mean,
        }
        return outdict

    def print_params_info(self):
        print("Info: Total-Data energy std", self._encountered_y_shape, ":", self._encountered_y_std)
        print("Info: Using energy-std", self.energy_std.shape, ":", self.energy_std)
        print("Info: Using energy-mean", self.energy_mean.shape, ":", self.energy_mean)
        print("Info: Using x-scale", self.x_std.shape, ":", self.x_std)
        print("Info: Using x-offset", self.x_mean.shape, ":", self.x_mean)

    def load_weights(self, file_path):
        weights = np.load(file_path, allow_pickle=True).item()
        self.x_mean = np.array(weights['x_mean'])
        self.x_std = np.array(weights['x_std'])
        self.energy_mean = np.array(weights['energy_mean'])
        self.energy_std = np.array(weights['energy_std'])

    def save_weights(self, file_path):
        outdict = {'x_mean': np.array(self.x_mean),
                   'x_std': np.array(self.x_std),
                   'energy_mean': np.array(self.energy_mean),
                   'energy_std': np.array(self.energy_std)}
        np.save(file_path, outdict)


class EnergyGradientStandardScaler(SaclerBase):
    def __init__(self,
                 scaler_module="energy",
                 use_energy_mean=True,
                 use_energy_std=True,
                 use_x_std=False,
                 use_x_mean=False,
                 ):
        super(EnergyGradientStandardScaler, self).__init__()
        self.use_energy_std = use_energy_std
        self.use_energy_mean = use_energy_mean
        self.use_x_std = use_x_std
        self.use_x_mean = use_x_mean

        self.x_mean = np.zeros((1, 1, 1))
        self.x_std = np.ones((1, 1, 1))
        self.energy_mean = np.zeros((1, 1))
        self.energy_std = np.ones((1, 1))
        self.gradient_mean = np.zeros((1, 1, 1, 1))
        self.gradient_std = np.ones((1, 1, 1, 1))

        self._encountered_y_shape = [None, None]
        self._encountered_y_std = [None, None]
        self.scaler_module = scaler_module

    def transform(self, x=None, y=None):
        x_res = x
        y_res = y
        if x is not None:
            x_res = (x - self.x_mean) / self.x_std
        if y is not None:
            if isinstance(y, list):
                energy = y[0]
                gradient = y[1]
            elif isinstance(y, dict):
                energy = y["energy"]
                gradient = y["force"]
            else:
                raise ValueError("Transform for expected [energy, force] but got %s" % y)
            out_e = (energy - self.energy_mean) / self.energy_std
            out_g = gradient / self.gradient_std
            y_res = [out_e, out_g]
        return x_res, y_res

    def inverse_transform(self, x=None, y=None):
        x_res = x
        y_res = y
        if x is not None:
            x_res = x * self.x_std + self.x_mean
        if y is not None:
            if isinstance(y, list):
                energy = y[0]
                gradient = y[1]
            elif isinstance(y, dict):
                energy = y["energy"]
                gradient = y["force"]
            else:
                raise ValueError("Transform for expected [energy, force] but got %s" % y)
            out_e = energy * self.energy_std + self.energy_mean
            out_g = gradient * self.gradient_std
            y_res = [out_e, out_g]
        return x_res, y_res

    def fit(self, x=None, y=None):
        npeps = np.finfo(float).eps
        if isinstance(y, list):
            y0 = y[0]
            y1 = y[1]
        elif isinstance(y, dict):
            y0 = y["energy"]
            y1 = y["force"]
        else:
            raise ValueError("Transform for expected [energy, force] but got %s" % y)
        if self.use_x_mean:
            self.x_mean = np.mean(x)
        if self.use_x_std:
            self.x_std = np.std(x) + npeps
        if self.use_energy_mean:
            self.energy_mean = np.mean(y0, axis=0, keepdims=True)
        if self.use_energy_std:
            self.energy_std = np.std(y0, axis=0, keepdims=True) + npeps

        self.gradient_std = np.expand_dims(np.expand_dims(self.energy_std, axis=-1), axis=-1) / self.x_std + npeps
        self.gradient_mean = np.zeros_like(self.gradient_std, dtype=np.float32)  # no mean shift expected

        self._encountered_y_shape = [np.array(y0.shape), np.array(y1.shape)]
        self._encountered_y_std = [np.std(y0, axis=0), np.std(y1, axis=(0, 2, 3))]

    def fit_transform(self, x=None, y=None):
        self.fit(x=x, y=y)
        return self.transform(x=x, y=y)

    def save_weights(self, file_path):
        out_dict = {
            'x_mean': self.x_mean,
            'x_std': self.x_std,
            'energy_mean': self.energy_mean,
            'energy_std': self.energy_std,
            'gradient_mean': self.gradient_mean,
            'gradient_std': self.gradient_std
        }
        np.save(file_path, out_dict)

    def load_weights(self, file_path):
        indict = np.load(file_path, allow_pickle=True).item()
        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.energy_mean = np.array(indict['energy_mean'])
        self.energy_std = np.array(indict['energy_std'])
        self.gradient_mean = np.array(indict['gradient_mean'])
        self.gradient_std = np.array(indict['gradient_std'])

    def print_params_info(self):
        print("Info: Total-Data gradient std", self._encountered_y_shape[1], ":", self._encountered_y_std[1])
        print("Info: Total-Data energy std", self._encountered_y_shape[0], ":", self._encountered_y_std[0])
        print("Info: Using energy-std", self.energy_std.shape, ":", self.energy_std[0])
        print("Info: Using energy-mean", self.energy_mean.shape, ":", self.energy_mean[0])
        print("Info: Using gradient-std", self.gradient_std.shape, ":", self.gradient_std[0, :, 0, 0])
        print("Info: Using gradient-mean", self.gradient_mean.shape, ":", self.gradient_mean[0, :, 0, 0])
        print("Info: Using x-scale", self.x_std.shape, ":", self.x_std)
        print("Info: Using x-offset", self.x_mean.shape, ":", self.x_mean)

    def get_config(self):
        outdict = {
            "scaler_module": self.scaler_module,
            "use_energy_mean": self.use_energy_mean,
            "use_energy_std": self.use_energy_std,
            "use_x_std": self.use_x_std,
            "use_x_mean": self.use_x_mean,
        }
        return outdict


class GradientStandardScaler(SaclerBase):

    def __init__(self,
                 scaler_module="energy",
                 use_x_std=False,
                 use_x_mean=False,
                 use_gradient_std=True,
                 use_gradient_mean=True
                 ):
        super(GradientStandardScaler, self).__init__()
        self.use_x_std = use_x_std
        self.use_x_mean = use_x_mean
        self.scaler_module = scaler_module
        self.use_gradient_mean = use_gradient_mean
        self.use_gradient_std = use_gradient_std

        self.x_mean = np.zeros((1, 1, 1))
        self.x_std = np.ones((1, 1, 1))
        self.gradient_mean = np.zeros((1, 1, 1, 1))
        self.gradient_std = np.ones((1, 1, 1, 1))

        self._encountered_y_shape = None
        self._encountered_y_std = None

    def transform(self, x=None, y=None):
        x_res = x
        y_res = y
        if x is not None:
            x_res = (x - self.x_mean) / self.x_std
        if y is not None:
            y_res = (y - self.gradient_mean) / self.gradient_std
        return x_res, y_res

    def inverse_transform(self, x=None, y=None):
        x_res = x
        out_gradient = y
        if x is not None:
            x_res = x * self.x_std + self.x_mean
        if y is not None:
            out_gradient = y * self.gradient_std + self.gradient_mean
        return x_res, out_gradient

    def fit(self, x, y):
        npeps = np.finfo(float).eps
        if self.use_x_mean:
            self.x_mean = np.mean(x)
        if self.use_x_std:
            self.x_std = np.std(x) + npeps
        if self.use_gradient_std:
            self.gradient_std = np.std(y, axis=(0, 3), keepdims=True) + npeps
            self.gradient_mean = np.zeros_like(self.gradient_std)

        self._encountered_y_std = np.std(y, axis=(0, 3), keepdims=True)
        self._encountered_y_shape = np.array(y.shape)

    def fit_transform(self, x=None, y=None):
        self.fit(x=x, y=y)
        return self.transform(x=x, y=y)

    def save_weights(self, file_path):
        out_dict = {
            'x_mean': self.x_mean,
            'x_std': self.x_std,
            'gradient_mean': self.gradient_mean,
            'gradient_std': self.gradient_std
        }
        np.save(file_path, out_dict)

    def load_weights(self, file_path):
        indict = np.load(file_path, allow_pickle=True).item()
        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.gradient_mean = np.array(indict['gradient_mean'])
        self.gradient_std = np.array(indict['gradient_std'])

    def get_config(self):
        conf = {
            'scaler_module': self.scaler_module,
            'use_x_std': self.use_x_std,
            'use_x_mean': self.use_x_mean,
            'use_gradient_mean': self.use_gradient_mean,
            'use_gradient_std': self.use_gradient_std,
        }
        return conf

    def print_params_info(self):
        print("Info: All-data gradient std", self._encountered_y_shape, ":", self._encountered_y_std[0, :, :, 0])
        print("Info: Using gradient-std", self.gradient_std.shape, ":", self.gradient_std[0, :, :, 0])
        print("Info: Using gradient-mean", self.gradient_mean.shape, ":", self.gradient_mean[0, :, :, 0])
        print("Info: Using x-scale", self.x_std.shape, ":", self.x_std)
        print("Info: Using x-offset", self.x_mean.shape, ":", self.x_mean)
