# custom optimizer
from keras.optimizers import Optimizer
import keras.backend as K

class Adam(Optimizer):
    """
    Adam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., accumulator=5., **kwargs):
        super(Adam, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.beta_1 = K.variable(beta_1, name='beta_1')
        self.beta_2 = K.variable(beta_2, name='beta_2')
        self.epsilon = epsilon
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.accumulator = K.variable(accumulator, name='accumulator')

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.get_variable_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.get_variable_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.get_variable_shape(p), dtype=K.dtype(p)) for p in params]

        self.weights = [self.iterations] + ms + vs

        for p, g, m, v, ga in zip(params, grads, ms, vs, gs):


            flag = K.equal(self.iterations % self.accumulator, 0)
            flag = K.cast(flag, dtype='float32')

            ga_t = (1 - flag) * (ga + g)

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * (ga + flag * g) / self.accumulator
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square((ga + flag * g) / self.accumulator)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)


            self.updates.append(K.update(m, flag * m_t + (1 - flag) * m))
            self.updates.append(K.update(v, flag * v_t + (1 - flag) * v))
            self.updates.append(K.update(ga, ga_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'accumulator': float(K.get_value(self.accumulator)),
                  'epsilon': self.epsilon}
        base_config = super(Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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


