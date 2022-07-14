import tensorflow as tf
ks = tf.keras

# x = tf.constant([[[0.0,0.0,0.0], [1.0,1.0,1.0]], [[2.0,2.0,2.0],[3.0,3.0,3.0]]])
# edi = tf.constant([[[1,1], [0,1]],
#                    [[0,1], [0,0]]])
# mask_e = tf.constant([[[True], [True]], [[True], [False]]])


@ks.utils.register_keras_serializable(package='pyNNsMD', name='PoolingLocalEdges')
class PoolingLocalEdges(ks.layers.Layer):

    def __init__(self, pooling_method="sum", **kwargs):
        super(PoolingLocalEdges, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(PoolingLocalEdges, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        if mask is not None:
            inputs = inputs * tf.cast(mask, inputs.dtype)
        out = tf.reduce_sum(inputs, axis=2)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingLocalEdges, self).get_config()
        return config


@ks.utils.register_keras_serializable(package='pyNNsMD', name='PoolingNodes')
class PoolingNodes(ks.layers.Layer):

    def __init__(self, pooling_method="sum", **kwargs):
        super(PoolingNodes, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(PoolingNodes, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        if mask is not None:
            inputs = inputs * tf.cast(mask, inputs.dtype)
        out = tf.reduce_sum(inputs, axis=1)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingNodes, self).get_config()
        return config


@ks.utils.register_keras_serializable(package='pyNNsMD', name='DenseMasked')
class DenseMasked(ks.layers.Layer):

    def __init__(self,  units: int,
                 activation=None,
                 use_bias: bool = True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DenseMasked, self).__init__(**kwargs)
        self._layer_dense = ks.layers.Dense(units=units, activation=activation,
                                            use_bias=use_bias, kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer,
                                            activity_regularizer=activity_regularizer,
                                            kernel_constraint=kernel_constraint,
                                            bias_constraint=bias_constraint)
    def build(self, input_shape):
        """Build layer."""
        super(DenseMasked, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        out = self._layer_dense(inputs, **kwargs)
        if mask is not None:
            out = out*tf.cast(mask, out.dtype)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(DenseMasked, self).get_config()
        conf_dense = self._layer_dense.get_config()
        for x in ["units", "activation", "use_bias", "kernel_initializer", "bias_initializer",
         "kernel_regularizer", "bias_regularizer", "activity_regularizer",
         "kernel_constraint", "bias_constraint"]:
            config.update({x: conf_dense[x]})
        return config


@ks.utils.register_keras_serializable(package='pyNNsMD', name='ApplyMask')
class ApplyMask(ks.layers.Layer):

    def __init__(self, **kwargs):
        super(ApplyMask, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(ApplyMask, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        if mask is not None:
            inputs = inputs*tf.cast(mask, inputs.dtype)
        return inputs

    def get_config(self):
        """Update layer config."""
        config = super(ApplyMask, self).get_config()
        return config


@ks.utils.register_keras_serializable(package='pyNNsMD', name='GatherEmbedding')
class GatherEmbedding(ks.layers.Layer):

    def __init__(self, **kwargs):
        super(GatherEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(GatherEmbedding, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        n, idx = inputs
        if mask is not None:
            mask_n, mask_e = mask
        ed = tf.gather(n, idx, axis=1, batch_dims=1)
        if mask is not None:
            ed = ed * tf.cast(mask_e, ed.dtype)
        return ed

    def get_config(self):
        """Update layer config."""
        config = super(GatherEmbedding, self).get_config()
        return config


@ks.utils.register_keras_serializable(package='pyNNsMD', name='NodeDistance')
class NodeDistance(ks.layers.Layer):

    def __init__(self, **kwargs):
        super(NodeDistance, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(NodeDistance, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        x, idx = inputs
        if mask is not None:
            _, mask_e = mask
        pos2 = tf.gather(x, idx, axis=1, batch_dims=1)
        pos1 = tf.expand_dims(x, axis=2)
        # print(pos1.shape, pos2.shape)
        dist = pos1 - pos2
        dist = tf.reduce_sum(tf.square(dist), axis=-1, keepdims=True)
        dist = tf.sqrt(dist)
        if mask is not None:
            dist = dist *tf.cast(mask_e, dist.dtype)
        return dist

    def get_config(self):
        """Update layer config."""
        config = super(NodeDistance, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable(package='pyNNsMD', name='SchNetCFconv')
class SchNetCFconv(ks.layers.Layer):
    def __init__(self, units,
                 cfconv_pool='segment_sum',
                 use_bias=True,
                 activation='kgcnn>shifted_softplus',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize Layer."""
        super(SchNetCFconv, self).__init__(**kwargs)
        self.cfconv_pool = cfconv_pool
        self.units = units
        self.use_bias = use_bias
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        # Layer
        self.lay_dense1 = ks.layers.Dense(
            units=self.units, activation=activation, use_bias=self.use_bias, **kernel_args)
        self.lay_dense2 = ks.layers.Dense(units=self.units, activation='linear', use_bias=self.use_bias, **kernel_args)
        self.lay_sum = PoolingLocalEdges(pooling_method=cfconv_pool)
        self.gather_n = GatherEmbedding()
        self.lay_mult = ks.layers.Multiply()

    def build(self, input_shape):
        """Build layer."""
        super(SchNetCFconv, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        node, edge, indexlist = inputs
        mask_n, mask_e, mask_i = mask
        x = self.lay_dense1(edge, **kwargs)
        x = self.lay_dense2(x, **kwargs)
        node2exp = self.gather_n([node, indexlist], mask=[mask_n, mask_i], **kwargs)
        x = self.lay_mult([node2exp, x], **kwargs)
        x = self.lay_sum(x, mask=mask_e, **kwargs)
        return x

    def get_config(self):
        """Update layer config."""
        config = super(SchNetCFconv, self).get_config()
        config.update({"cfconv_pool": self.cfconv_pool, "units": self.units})
        config_dense = self.lay_dense1.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias"]:
            config.update({x: config_dense[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='pyNNsMD', name='SchNetInteraction')
class SchNetInteraction(ks.layers.Layer):
    def __init__(self,
                 units=128,
                 cfconv_pool='sum',
                 use_bias=True,
                 activation='kgcnn>shifted_softplus',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize Layer."""
        super(SchNetInteraction, self).__init__(**kwargs)
        self.cfconv_pool = cfconv_pool
        self.use_bias = use_bias
        self.units = units
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        conv_args = {"units": self.units, "use_bias": use_bias, "activation": activation, "cfconv_pool": cfconv_pool}

        # Layers
        self.lay_cfconv = SchNetCFconv(**conv_args, **kernel_args)
        self.lay_dense1 = DenseMasked(units=self.units, activation='linear', use_bias=False, **kernel_args)
        self.lay_dense2 = DenseMasked(
            units=self.units, activation=activation, use_bias=self.use_bias, **kernel_args)
        self.lay_dense3 = DenseMasked(units=self.units, activation='linear', use_bias=self.use_bias, **kernel_args)
        self.lay_add = ks.layers.Add()

    def build(self, input_shape):
        """Build layer."""
        super(SchNetInteraction, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        node, edge, indexlist = inputs
        mask_n, mask_e, mask_i = mask
        x = self.lay_dense1(node, mask=mask_n, **kwargs)
        x = self.lay_cfconv([x, edge, indexlist], mask=mask, **kwargs)
        x = self.lay_dense2(x, **kwargs)
        x = self.lay_dense3(x, mask=mask_n,**kwargs)
        out = self.lay_add([node, x], **kwargs)
        return out

    def get_config(self):
        config = super(SchNetInteraction, self).get_config()
        config.update({"cfconv_pool": self.cfconv_pool, "units": self.units, "use_bias": self.use_bias})
        conf_dense = self.lay_dense2.get_config()
        for x in ["activation", "kernel_regularizer", "bias_regularizer", "activity_regularizer",
                  "kernel_constraint", "bias_constraint", "kernel_initializer", "bias_initializer"]:
            config.update({x: conf_dense[x]})
        return config
