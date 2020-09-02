"""
Define forms of regularization
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks


def identify_regularizer(reg_str,l1_str,l2_str):
    """
    Identify regularization. This is redundant and will be replaced by tf.keras.regularizatio.get().
    """
    out_reg = None
    if(reg_str=='l1'):
        out_reg = ks.regularizers.l1(l1_str)
    if(reg_str=='l2'):
        out_reg = ks.regularizers.l2(l2_str)
    if(reg_str=='l1_l2'):
        out_reg = ks.regularizers.l1_l2(l1_str,l2_str)
    return out_reg