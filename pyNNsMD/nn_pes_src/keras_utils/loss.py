"""
Functions for loss.

Also includes Metrics and tools around loss.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
    

def get_lr_metric(optimizer):
    """
    Obtian learning rate from optimizer.

    Args:
        optimizer (tf.kears.optimizer): Optimizer used for training.

    Returns:
        float: learning rate.

    """

    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


def r2_metric(y_true, y_pred):
    """
    Compute r2 metric.

    Args:
        y_true (tf.tensor): True y-values.
        y_pred (tf.tensor): Predicted y-values.

    Returns:
        tf.tensor: r2 metric.

    """
    SS_res =  ks.backend.sum(ks.backend.square(y_true - y_pred)) 
    SS_tot = ks.backend.sum(ks.backend.square(y_true-ks.backend.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + ks.backend.epsilon()) )


def nac_loss(y_true, y_pred):
    """
    Phaseless loss for the NAC prediction. Needs to be adapted for multiple states.
    @TODO: define for more states with specific ordering

    Args:
        y_true (tf.tensor): True y-values.
        y_pred (tf.tensor): Predicted y-values.

    Returns:
        tf.tensor: Phaseindependent MSE.

    """
    out1 = ks.backend.mean(ks.backend.square(y_true - y_pred))
    out2 = ks.backend.mean(ks.backend.square(y_true + y_pred))
    return ks.backend.minimum(out1,out2)


def merge_hist(hist1,hist2):
    """
    Merge two hist-dicts.

    Args:
        hist1 (dict): Hist dict from fit.
        hist2 (dict): Hist dict from fit.

    Returns:
        outhist (dict): hist1 + hist2.

    """
    outhist = {}
    for x,y in hist1.items():
        outhist.update({x : hist1[x] + hist2[x]})
    return outhist
        