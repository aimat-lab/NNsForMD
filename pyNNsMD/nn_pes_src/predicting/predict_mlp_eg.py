# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:52:27 2020

@author: Patrick
"""

import numpy as np

def _predict_uncertainty_mlp_eg(out):
    out_mean = []
    out_std = []
    for i in range(2):
        out_mean.append(np.mean(np.array([x[i] for x in out]),axis=0))
        out_std.append(np.std(np.array([x[i] for x in out]),axis=0,ddof=1))
    return out_mean,out_std