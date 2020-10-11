"""
Created on Sat Oct 10 19:48:15 2020

@author: Patrick
"""

import numpy as np
import tensorflow as tf





def _predict_uncertainty(model_type,out):
    out_mean = []
    out_std = []
    if(model_type == 'mlp_nac'):
        out_mean = np.mean(np.array(out),axis=0)
        out_std = np.std(np.array(out),axis=0,ddof=1)
    elif(model_type == 'mlp_eg'):
        for i in range(2):
            out_mean.append(np.mean(np.array([x[i] for x in out]),axis=0))
            out_std.append(np.std(np.array([x[i] for x in out]),axis=0,ddof=1))
    else:
        print("Error: Unknown model type for predict",model_type)
        raise TypeError(f"Error: Unknown model type for predict {model_type}")
        
    return out_mean,out_std



def _call_convert_output_tonumpy(model_type,temp):
    if(model_type == 'mlp_eg'):
        return temp.numpy()
    elif(model_type == 'mlp_nac'):
        return [temp[0].numpy(),temp[1].numpy()]
    else:
        print("Error: Unknown model type for predict",model_type)
        raise TypeError(f"Error: Unknown model type for predict {model_type}")
        

def _call_convert_x_totensor(model_type,x):
    if(model_type == 'mlp_eg'):
        return tf.convert_to_tensor( x,dtype=tf.float32)
    elif(model_type == 'mlp_nac'):
        return tf.convert_to_tensor( x,dtype=tf.float32)
    else:
        print("Error: Unknown model type for predict",model_type)
        raise TypeError(f"Error: Unknown model type for predict {model_type}")
    