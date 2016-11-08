import keras.backend as K
from keras.optimizers import Optimizer


class Eve(Optimizer):
    '''Eve optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2/beta_3: floats, 0 < beta < 1. Generally close to 1.
        small_k/big_K: floats
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Improving Stochastic Gradient Descent With FeedBack](http://arxiv.org/abs/1611.01505v1.pdf)
    '''

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 beta_3=0.999, small_k=0.1, big_K=10,
                 epsilon=1e-8, decay=0., **kwargs):
        super(Eve, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)
        self.beta_3 = K.variable(beta_3)
        self.small_k = K.variable(small_k)
        self.big_K = K.variable(big_K)
        self.decay = K.variable(decay)
        self.inital_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.inital_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        f = K.variable(0)
        d = K.variable(1)
        self.weights = [self.iterations] + ms + vs + [f, d]

        cond = K.greater(t, K.variable(1))
        small_delta_t = K.switch(K.greater(loss, f), self.small_k + 1, 1. / (self.big_K + 1))
        big_delta_t = K.switch(K.greater(loss, f), self.big_K + 1, 1. / (self.small_k + 1))

        c_t = K.minimum(K.maximum(small_delta_t, loss / (f + self.epsilon)), big_delta_t)
        f_t = c_t * f
        r_t = K.abs(f_t - f) / (K.minimum(f_t, f))
        d_t = self.beta_3 * d + (1 - self.beta_3) * r_t

        f_t = K.switch(cond, f_t, loss)
        d_t = K.switch(cond, d_t, K.variable(1.))

        self.updates.append(K.update(f, f_t))
        self.updates.append(K.update(d, d_t))

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - lr_t * m_t / (d_t * K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

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
                  'beta_3': float(K.get_value(self.beta_3)),
                  'small_k': float(K.get_value(self.small_k)),
                  'big_K': float(K.get_value(self.big_K)),
                  'epsilon': self.epsilon}
        base_config = super(Eve, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
