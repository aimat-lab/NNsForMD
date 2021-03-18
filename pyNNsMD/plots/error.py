import numpy as np


def find_max_relative_error(preds,yval):
    """
    Find maximum error and its relative value if possible.

    Args:
        preds (np.array): Prediction array.
        yval (np.array): Validation array.

    Returns:
        pred_err (np.array): Flatten maximum error along axis=0
        prelm (np.array): Flatten Relative maximum error along axis=0

    """
    pred = np.reshape(preds,(preds.shape[0],-1))
    flat_yval = np.reshape(yval,(yval.shape[0],-1))
    maxerr_ind = np.expand_dims(np.argmax(np.abs(pred-flat_yval),axis=0),axis=0)
    pred_err = np.abs(np.take_along_axis(pred,maxerr_ind,axis=0)-
                      np.take_along_axis(flat_yval,maxerr_ind,axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        prelm = pred_err / np.abs(np.take_along_axis(flat_yval,maxerr_ind,axis=0))
    pred_err = pred_err.flatten()
    prelm = prelm.flatten()
    return pred_err,prelm
