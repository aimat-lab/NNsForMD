import json

import numpy as np
from pyNNsMD.scaler.base import ScalerBase


class NACStandardScaler(ScalerBase):
    def __init__(self,
                 use_x_std=False,
                 use_x_mean=False,
                 use_nac_std=True,
                 use_nac_mean=True,
                 scaler_module="nac"
                 ):
        self.use_x_std = use_x_std
        self.use_x_mean = use_x_mean
        self.use_nac_std = use_nac_std
        self.use_nac_mean = use_nac_mean
        self.scaler_module = scaler_module

        # Weights
        self.x_mean = np.zeros((1, 1, 1))
        self.x_std = np.ones((1, 1, 1))
        self.nac_mean = np.zeros((1, 1, 1, 1))
        self.nac_std = np.ones((1, 1, 1, 1))

        self._encountered_y_shape = None
        self._encountered_y_std = None

    def transform(self, x=None, y=None):
        x_res = x
        y_res = y
        if x is not None:
            x_res = (x - self.x_mean) / self.x_std
        if y is not None:
            y_res = (y - self.nac_mean) / self.nac_std
        return x_res, y_res

    def inverse_transform(self, x=None, y=None):
        x_res = x
        out_nac = y
        if x is not None:
            x_res = x * self.x_std + self.x_mean
        if y is not None:
            out_nac = y * self.nac_std + self.nac_mean
        return x_res, out_nac

    def fit(self, x, y):
        npeps = np.finfo(float).eps
        if self.use_x_mean:
            self.x_mean = np.mean(x)
        if self.use_x_std:
            self.x_std = np.std(x) + npeps
        if self.use_nac_std:
            self.nac_std = np.std(y, axis=(0, 3), keepdims=True) + npeps
            self.nac_mean = np.zeros_like(self.nac_std)

        self._encountered_y_std = np.std(y, axis=(0, 3), keepdims=True)
        self._encountered_y_shape = np.array(y.shape)

    def fit_transform(self, x=None, y=None):
        self.fit(x=x, y=y)
        return self.transform(x=x,y=y)

    def save_weights(self, file_path):
        out_dict = {
            'x_mean': self.x_mean,
            'x_std': self.x_std,
            'nac_mean': self.nac_mean,
            'nac_std': self.nac_std
        }
        np.save(file_path, out_dict)

    def load_weights(self, file_path):
        indict = np.load(file_path, allow_pickle=True).item()
        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.nac_mean = np.array(indict['nac_mean'])
        self.nac_std = np.array(indict['nac_std'])

    def get_config(self):
        conf = {
            'scaler_module': self.scaler_module,
            'use_x_std': self.use_x_std,
            'use_x_mean': self.use_x_mean,
            'use_nac_mean': self.use_nac_mean,
            'use_nac_std': self.use_nac_std,
        }
        return conf

    def print_params_info(self):
        print("Info: All-data NAC std", self._encountered_y_shape, ":", self._encountered_y_std[0, :, :, 0])
        print("Info: Using nac-std", self.nac_std.shape, ":", self.nac_std[0, :, :, 0])
        print("Info: Using nac-mean", self.nac_mean.shape, ":", self.nac_mean[0, :, :, 0])
        print("Info: Using x-scale", self.x_std.shape, ":", self.x_std)
        print("Info: Using x-offset", self.x_mean.shape, ":", self.x_mean)
