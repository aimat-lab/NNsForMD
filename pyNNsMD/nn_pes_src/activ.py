"""
Smooth activation functions for tensorflow.keras.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks


def leaky_softplus(alpha = 0.3):
    """
    Leaky softplus activation function similar to leakyRELU but smooth.

    Parameters
    ----------
    alpha : float, optional
        Leaking slope. The default is 0.3.

    Returns
    -------
    func
        lambda function of x.

    """
    return lambda x : ks.activations.softplus(x)*(1-alpha)+alpha*x

def shifted_sofplus(x):
    """
    Softplus function from tf.keras shifted downwards.

    Parameters
    ----------
    x : tf.tensor
        Activation input.

    Returns
    -------
    tf.tensor
        Activation.

    """
    return ks.activations.softplus(x) - ks.backend.log(2.0)


def identify_keras_activation(instr,alpha=None,beta=None):
    """
    Identify ativation function by string.

    Parameters
    ----------
    instr : str
        Name of function.
    alpha : float, optional
        Alpha Paramter. The default is None.
    beta : float, optional
        Beta Parameter. The default is None.

    Returns
    -------
    activ : fun
        Activation function.

    """
    if(instr == 'shifted_softplus'):
        activ = shifted_sofplus
    elif(instr == 'leaky_softplus'):
        activ = leaky_softplus(alpha)
    else:
        activ = instr
    return activ