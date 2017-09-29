import keras.backend as K
from keras.optimizers import Optimizer
from keras.preprocessing import image
from keras.legacy import interfaces
import numpy as np


def random_flip(img, mask, u=0.5):
    if np.random.random() < u:
        img = image.flip_axis(img, 1)
        mask = image.flip_axis(mask, 1)
    return img, mask

def rotate(x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def random_rotate(img, mask, rotate_limit=(-20, 20), u=0.5):
    if np.random.random() < u:
        theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
        img = rotate(img, theta)
        mask = rotate(mask, theta)
    return img, mask

def shift(x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hshift * h
    ty = wshift * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    transform_matrix = translation_matrix  # no need to do offset
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def random_shift(img, mask, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.5):
    if np.random.random() < u:
        wshift = np.random.uniform(w_limit[0], w_limit[1])
        hshift = np.random.uniform(h_limit[0], h_limit[1])
        img = shift(img, wshift, hshift)
        mask = shift(mask, wshift, hshift)
    return img, mask

def zoom(x, zx, zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(zoom_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def random_zoom(img, mask, zoom_range=(0.8, 1), u=0.5):
    if np.random.random() < u:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        img = zoom(img, zx, zy)
        mask = zoom(mask, zx, zy)
    return img, mask

def shear(x, shear, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(shear_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def random_shear(img, mask, intensity_range=(-0.5, 0.5), u=0.5):
    if np.random.random() < u:
        sh = np.random.uniform(-intensity_range[0], intensity_range[1])
        img = shear(img, sh)
        mask = shear(mask, sh)
    return img, mask

def random_channel_shift(x, limit, channel_axis=2):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_ch + np.random.uniform(-limit, limit), min_x, max_x) for x_ch in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def random_gray(img, u=0.5):
    if np.random.random() < u:
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = np.sum(img * coef, axis=2)
        img = np.dstack((gray, gray, gray))
    return img

def random_contrast(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = img * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha * img + gray
        img = np.clip(img, 0., 1.)
    return img

def random_brightness(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = alpha * img
        img = np.clip(img, 0., 1.)
    return img

def random_saturation(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = img * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        img = alpha * img + (1. - alpha) * gray
        img = np.clip(img, 0., 1.)
    return img

def random_augmentation(img, mask):
    # img = random_channel_shift(img, limit=0.05)
    # img = random_brightness(img, limit=(-0.5, 0.5), u=0.5)
    # img = random_contrast(img, limit=(-0.5, 0.5), u=0.5)
    # img = random_saturation(img, limit=(-0.5, 0.5), u=0.5)
    # img = random_gray(img, u=0.2)
    # img, mask = random_rotate(img, mask, rotate_limit=(-20, 20), u=0.5)
    # img, mask = random_shear(img, mask, intensity_range=(-0.3, 0.3), u=0.2)
    img, mask = random_flip(img, mask, u=0.5)
    img, mask = random_shift(img, mask, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.3)
    img, mask = random_zoom(img, mask, zoom_range=(0.9, 1.1), u=0.5)
    return img, mask


# Version 2.0.8
# Not work
# class SGD(Optimizer):
#     """Stochastic gradient descent optimizer.
#
#     Includes support for momentum,
#     learning rate decay, and Nesterov momentum.
#
#     # Arguments
#         lr: float >= 0. Learning rate.
#         momentum: float >= 0. Parameter updates momentum.
#         decay: float >= 0. Learning rate decay over each update.
#         nesterov: boolean. Whether to apply Nesterov momentum.
#     """
#
#     def __init__(self, lr=0.01, momentum=0., decay=0.,
#                  nesterov=False, accumulator=5., **kwargs):
#         super(SGD, self).__init__(**kwargs)
#         with K.name_scope(self.__class__.__name__):
#             self.iterations = K.variable(0, dtype='int64', name='iterations')
#             self.lr = K.variable(lr, name='lr')
#             self.momentum = K.variable(momentum, name='momentum')
#             self.decay = K.variable(decay, name='decay')
#         self.initial_decay = decay
#         self.nesterov = nesterov
#         self.accumulator = K.variable(accumulator, name='accumulator')
#
#     @interfaces.legacy_get_updates_support
#     def get_updates(self, loss, params):
#         grads = self.get_gradients(loss, params)
#         self.updates = [K.update_add(self.iterations, 1)]
#
#         lr = self.lr
#         if self.initial_decay > 0:
#             lr *= (1. / (1. + self.decay * K.cast(self.iterations,
#                                                   K.dtype(self.decay))))
#         # momentum
#         shapes = [K.int_shape(p) for p in params]
#         moments = [K.zeros(shape) for shape in shapes]
#         gradients = [K.zeros(shape) for shape in shapes]
#
#         self.weights = [self.iterations] + moments
#         for p, g, m, gg in zip(params, grads, moments, gradients):
#
#             flag = K.equal(self.iterations % self.accumulator, 0)
#             # flag = K.cast(flag, dtype='float32')
#
#             gg_t = (1 - flag) * (gg + g)
#
#             v = self.momentum * m - lr * (gg + flag * g) / self.accumulator  # velocity
#
#             self.updates.append(K.update(m, flag * v + (1 - flag) * m))
#             self.updates.append((gg, gg_t))
#
#             if self.nesterov:
#                 new_p = p + self.momentum * v - lr * (gg + flag * g) / self.accumulator
#             else:
#                 new_p = p + v
#
#             # Apply constraints.
#             if getattr(p, 'constraint', None) is not None:
#                 new_p = p.constraint(new_p)
#
#             self.updates.append(K.update(p, new_p))
#         return self.updates
#
#     def get_config(self):
#         config = {'lr': float(K.get_value(self.lr)),
#                   'momentum': float(K.get_value(self.momentum)),
#                   'decay': float(K.get_value(self.decay)),
#                   'accumulator': float(K.get_value(self.accumulator)),
#                   'nesterov': self.nesterov}
#         base_config = super(SGD, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


# Version 2.0.5
class SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, accumulator=5., **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.iterations = K.variable(0., name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.momentum = K.variable(momentum, name='momentum')
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.accumulator = K.variable(accumulator, name='accumulator')

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates .append(K.update_add(self.iterations, 1))

        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        gradients = [K.zeros(shape) for shape in shapes]

        self.weights = [self.iterations] + moments
        for p, g, m, gg in zip(params, grads, moments, gradients):

            flag = K.equal(self.iterations % self.accumulator, 0)
            flag = K.cast(flag, dtype='float32')

            gg_t = (1 - flag) * (gg + g)

            v = self.momentum * m - lr * (gg + flag * g) / self.accumulator  # velocity

            self.updates.append(K.update(m, flag * v + (1 - flag) * m))
            self.updates.append((gg, gg_t))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * (gg + flag * g) / self.accumulator
            else:
                new_p = p + v

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
                  'accumulator': float(K.get_value(self.accumulator)),
                  'nesterov': self.nesterov}
        base_config = super(SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Adam(Optimizer):
    """Adam optimizer.

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
