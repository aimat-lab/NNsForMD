import tensorflow as tf
import tensorflow.keras as ks


class EnergyGradient(ks.layers.Layer):
    def __init__(self, mult_states=1, **kwargs):
        super(EnergyGradient, self).__init__(**kwargs)
        self.mult_states = mult_states

    def build(self, input_shape):
        super(EnergyGradient, self).build(input_shape)

    def call(self, inputs):
        energy, coords = inputs
        out = [ks.backend.expand_dims(ks.backend.gradients(energy[:, i], coords)[0], axis=1) for i in
               range(self.mult_states)]
        out = ks.backend.concatenate(out, axis=1)
        return out

    def get_config(self):
        config = super(EnergyGradient, self).get_config()
        config.update({"mult_states": self.mult_states})
        return config


class NACGradient(ks.layers.Layer):
    def __init__(self, mult_states=1, atoms=1, **kwargs):
        super(NACGradient, self).__init__(**kwargs)
        self.mult_states = mult_states
        self.atoms = atoms

    def build(self, input_shape):
        super(NACGradient, self).build(input_shape)

    def call(self, inputs):
        energy, coords = inputs
        out = ks.backend.concatenate(
            [ks.backend.expand_dims(ks.backend.gradients(energy[:, i], coords)[0], axis=1) for i in
             range(self.mult_states * self.atoms)], axis=1)
        out = ks.backend.reshape(out, (ks.backend.shape(coords)[0], self.mult_states, self.atoms, self.atoms, 3))
        out = ks.backend.concatenate([ks.backend.expand_dims(out[:, :, i, i, :], axis=2) for i in range(self.atoms)],
                                     axis=2)
        return out

    def get_config(self):
        config = super(NACGradient, self).get_config()
        config.update({"mult_states": self.mult_states, 'atoms': self.atoms})
        return config


class EmptyGradient(ks.layers.Layer):
    def __init__(self, mult_states=1, atoms=1, **kwargs):
        super(EmptyGradient, self).__init__(**kwargs)
        self.mult_states = mult_states
        self.atoms = atoms

    def build(self, input_shape):
        super(EmptyGradient, self).build(input_shape)

    def call(self, inputs):
        pot = inputs
        out = tf.zeros((ks.backend.shape(pot)[0], self.mult_states, self.atoms, 3))
        return out

    def get_config(self):
        config = super(EmptyGradient, self).get_config()
        config.update({"mult_states": self.mult_states, 'atoms': self.atoms})
        return config


class PropagateEnergyGradient(ks.layers.Layer):
    def __init__(self, mult_states=1, **kwargs):
        super(PropagateEnergyGradient, self).__init__(**kwargs)
        self.mult_states = mult_states

    def build(self, input_shape):
        super(PropagateEnergyGradient, self).build(input_shape)

    def call(self, inputs):
        grads, grads2 = inputs
        out = ks.backend.batch_dot(grads, grads2, axes=(2, 1))
        return out

    def get_config(self):
        config = super(PropagateEnergyGradient, self).get_config()
        config.update({"mult_states": self.mult_states})
        return config


class PropagateNACGradient(ks.layers.Layer):
    def __init__(self, mult_states=1, atoms=1, **kwargs):
        super(PropagateNACGradient, self).__init__(**kwargs)
        self.mult_states = mult_states
        self.atoms = atoms

    def build(self, input_shape):
        super(PropagateNACGradient, self).build(input_shape)

    def call(self, inputs):
        grads, grads2 = inputs
        out = ks.backend.batch_dot(grads, grads2, axes=(3, 1))
        out = ks.backend.concatenate([ks.backend.expand_dims(out[:, :, i, i, :], axis=2) for i in range(self.atoms)],
                                     axis=2)
        return out

    def get_config(self):
        config = super(PropagateNACGradient, self).get_config()
        config.update({"mult_states": self.mult_states, 'atoms': self.atoms})
        return config


class PropagateNACGradient2(ks.layers.Layer):
    def __init__(self, axis=(2, 1), **kwargs):
        super(PropagateNACGradient2, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        super(PropagateNACGradient2, self).build(input_shape)

    def call(self, inputs):
        grads, grads2 = inputs
        out = ks.backend.batch_dot(grads, grads2, axes=self.axis)
        return out

    def get_config(self):
        config = super(PropagateNACGradient2, self).get_config()
        config.update({"axis": self.axis})
        return config
