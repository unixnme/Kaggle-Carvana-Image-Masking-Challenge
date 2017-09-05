# custom optimizer
from keras.optimizers import Optimizer
import keras.backend as K

class SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        num_batches_per_update: int >= 1. Number of batches to wait for update
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 accum_iters=1, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.momentum = K.variable(momentum, name='momentum')
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.accum_iters = K.variable(accum_iters)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations, 'float32')))
        self.updates.append(K.update_add(self.iterations, 1))
        update_flag = K.equal(self.iterations % self.accum_iters, 0)
        #update_flag = K.equal(0,0)

        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        velocity_accum = [K.zeros(shape) for shape in shapes]
        for p, g, m, v_accum in zip(params, grads, moments, velocity_accum):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update_add(v_accum, v))
            self.updates.append(K.update(m, v))

            new_p = p + v_accum * K.cast(K.cast(update_flag, 'int32'), 'float32') / K.cast(self.accum_iters, 'float32')
            self.updates[-1] = K.update_sub(v_accum, K.cast(K.cast(update_flag, 'int32'), 'float32')*v_accum)

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'accum_iters': float(K.get_value(self.accum_iters))}
        base_config = super(SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


