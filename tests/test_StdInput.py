# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:44:28 2020

@author: Patrick
"""

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K
import numpy as np


class StandardizeFeatures(ks.layers.Layer):
    def __init__(self, **kwargs):
        super(StandardizeFeatures, self).__init__(**kwargs)          
        self.axis = -1
    def build(self, input_shape):
        super(StandardizeFeatures, self).build(input_shape) 
        outshape = [1]*len(input_shape)
        outshape[self.axis] = input_shape[self.axis]
        #outshape = (input_shape[self.axis],)
        self.wmean = self.add_weight(
            'std_feat_mean',
            shape=outshape,
            initializer=tf.keras.initializers.Zeros(),
            dtype=self.dtype,
            trainable=False)         
        self.wstd = self.add_weight(
            'std_feat_std',
            shape=outshape,
            initializer= tf.keras.initializers.Ones(),
            dtype=self.dtype,
            trainable=False)
    def call(self, inputs):
        out = (inputs-self.wmean)/self.wstd
        return out 



#Model
in1 = np.array([[[1.0,1.0,0],[0,1,0],[0,0,0]],[[1.0,0,0],[0,2,0],[0,1,0]],[[0,1,0],[0,0,1.0],[2,0,0]]])

geo_input = ks.Input(shape=(3,3), dtype='float32' ,name='geo_input')
temp = StandardizeFeatures(name='std_feat')(geo_input )
#out = ks.layers.Dense(1,use_bias=True,activation='linear',kernel_initializer=ks.initializers.Constant(value=1))(temp)
out = temp
model = ks.models.Model(inputs=geo_input, outputs=out)

optimizer = tf.keras.optimizers.Adam()

model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_absolute_error'])
model.summary()

model.build((3,3))
for layer in model.layers:
    print(layer.name)
    
print(model.get_layer('std_feat').set_weights([np.array([[[0,0,0]]]),np.array([[[-1,0,1]]])]))
print(model.predict(in1))

