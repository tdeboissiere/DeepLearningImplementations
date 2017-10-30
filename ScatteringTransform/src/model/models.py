import sys
import tensorflow as tf
sys.path.append("../utils")
import layers
from filters_bank import filters_bank


class Model(object):

    def __init__(self, name):
        self.name = name

    def get_trainable_variables(self):
        t_vars = tf.trainable_variables()
        t_vars_model = {v.name: v for v in t_vars if self.name in v.name}
        return t_vars_model


class Generator(Model):
    def __init__(self, list_filters, list_kernel_size, list_strides, list_padding, output_shape,
                 name="generator", batch_size=32, filters=512, dset="celebA", data_format="NCHW"):

        super(Generator, self).__init__(name)

        self.data_format = data_format

        if self.data_format == "NCHW":
            self.output_h = output_shape[1]
            self.output_w = output_shape[2]
        else:
            self.output_h = output_shape[0]
            self.output_w = output_shape[1]

        if dset == "mnist":
            self.start_dim = int(self.output_h / 4)
            self.nb_upconv = 2
        else:
            self.start_dim = int(self.output_h / 16)
            self.nb_upconv = 4

        self.output_shape = output_shape
        self.dset = dset
        self.name = name
        self.batch_size = batch_size
        self.filters = filters
        self.list_filters = list_filters
        self.list_kernel_size = list_kernel_size
        self.list_padding = list_padding
        self.list_strides = list_strides

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            # Initial dense multiplication
            x = layers.linear(x, self.filters * self.start_dim * self.start_dim, bias=True)

            # Reshape to image format
            if self.data_format == "NCHW":
                target_shape = (self.batch_size, self.filters, self.start_dim, self.start_dim)
            else:
                target_shape = (self.batch_size, self.start_dim, self.start_dim, self.filters)

            x = layers.reshape(x, target_shape)
            x = tf.contrib.layers.batch_norm(x, fused=True)
            x = layers.lrelu(x)

            # # Upsampling2D + conv blocks
            for idx, (f, k, s, p) in enumerate(zip(self.list_filters, self.list_kernel_size, self.list_strides, self.list_padding)):
                name = "upsample2D_%s" % idx
                if idx == len(self.list_filters) - 1:
                    bn = False
                    bias = False
                    activation_fn = None
                else:
                    bias = True
                    bn = True
                    activation_fn = layers.lrelu
                x = layers.upsample2d_block(name, x, f, k, s, p, data_format=self.data_format, bias=bias, bn=bn, activation_fn=activation_fn)

            x = tf.nn.tanh(x, name="X_G")

            return x


class Discriminator(Model):
    def __init__(self, name="discriminator", data_format="NCHW"):
        # Determine data format from output shape

        super(Discriminator, self).__init__(name)

        self.data_format = data_format
        self.name = name

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            x = tf.contrib.layers.batch_norm(x, data_format=self.data_format, fused=True, scope="scat_bn")
            x = layers.conv2d_block("D_conv2D1", x, 32, 3, 2, p="SAME", data_format=self.data_format, bias=False, bn=False, activation_fn=layers.lrelu)
            x = layers.conv2d_block("D_conv2D2", x, 64, 3, 2, p="SAME", data_format=self.data_format, bias=False, bn=True, activation_fn=layers.lrelu)
            # x = layers.conv2d_block("D_conv2D3", x, 128, 3, 2, p="SAME", data_format=self.data_format, bias=True, bn=True, activation_fn=layers.lrelu)

            x_shape = x.get_shape().as_list()
            target_shape = (-1, x_shape[-1] * x_shape[-2] * x_shape[-3])
            x = layers.reshape(x, target_shape)

            x = layers.linear(x, 1, name='dense2')

            return x


class Scattering(Model):
    """Scattering module.

    Runs scattering on an input image in NCHW format

    Input args:
        M, N: input image size
        J: number of layers
    """

    def __init__(self, M, N, J, name="scattering"):
        super(Scattering, self).__init__(name)
        self.M, self.N, self.J = M, N, J

        self._prepare_padding_size([1, 1, M, N])

        # Create the filters
        filters = filters_bank(self.M_padded, self.N_padded, J)

        self.Psi = filters['psi']
        self.Phi = [filters['phi'][j] for j in range(J)]

    def _prepare_padding_size(self, s):
        M = s[-2]
        N = s[-1]

        self.M_padded = ((M + 2 ** (self.J)) // 2**self.J + 1) * 2**self.J
        self.N_padded = ((N + 2 ** (self.J)) // 2**self.J + 1) * 2**self.J

        s[-2] = self.M_padded
        s[-1] = self.N_padded
        self.padded_size_batch = [a for a in s]

    # This function copies and view the real to complex
    def _pad(self, x):
        # No pre pad option. TODO: add it ?
        paddings = [[0, 0], [0, 0], [2 ** self.J, 2 ** self.J], [2 ** self.J, 2 ** self.J]]
        out_ = tf.pad(x, paddings, mode="REFLECT")
        out_ = tf.expand_dims(out_, axis=-1)
        output = tf.concat([out_, tf.zeros_like(out_)], axis=-1)
        return output

    def _unpad(self, in_):
        return in_[..., 1:-1, 1:-1]

    def __call__(self, x, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            x_shape = x.get_shape().as_list()
            x_h, x_w = x_shape[-2:]

            if (x_w != self.N or x_h != self.M):
                raise (RuntimeError('Tensor must be of spatial size (%i, %i)!' % (self.M, self.N)))

            if (len(x_shape) != 4):
                raise (RuntimeError('Input tensor must be 4D'))

            J = self.J
            phi = self.Phi
            psi = self.Psi
            n = 0

            pad = self._pad
            unpad = self._unpad

            S = []
            U_r = pad(x)

            U_0_c = compute_fft(U_r, 'C2C')  # We trick here with U_r and U_2_c
            U_1_c = periodize(cdgmm(U_0_c, phi[0]), 2**J)
            U_J_r = compute_fft(U_1_c, 'C2R')

            S.append(unpad(U_J_r))
            n = n + 1

            for n1 in range(len(psi)):
                j1 = psi[n1]['j']
                U_1_c = cdgmm(U_0_c, psi[n1][0])
                if j1 > 0:
                    U_1_c = periodize(U_1_c, k=2 ** j1)
                U_1_c = compute_fft(U_1_c, 'C2C', inverse=True)
                U_1_c = compute_fft(modulus(U_1_c), 'C2C')

                # Second low pass filter
                U_2_c = periodize(cdgmm(U_1_c, phi[j1]), k=2**(J - j1))
                U_J_r = compute_fft(U_2_c, 'C2R')
                S.append(unpad(U_J_r))
                n = n + 1

                for n2 in range(len(psi)):
                    j2 = psi[n2]['j']
                    if j1 < j2:
                        U_2_c = periodize(cdgmm(U_1_c, psi[n2][j1]), k=2 ** (j2 - j1))
                        U_2_c = compute_fft(U_2_c, 'C2C', inverse=True)
                        U_2_c = compute_fft(modulus(U_2_c), 'C2C')

                        # Third low pass filter
                        U_2_c = periodize(cdgmm(U_2_c, phi[j2]), k=2 ** (J - j2))
                        U_J_r = compute_fft(U_2_c, 'C2R')

                        S.append(unpad(U_J_r))
                        n = n + 1

            S = tf.concat(S, axis=1)
            return S


def stack_real_imag(x):

    stack_axis = len(x.get_shape().as_list())
    return tf.stack((tf.real(x), tf.imag(x)), axis=stack_axis)


def compute_fft(x, direction="C2C", inverse=False):

    if direction == 'C2R':
        inverse = True

    x_shape = x.get_shape().as_list()
    h, w = x_shape[-2], x_shape[-3]

    x_complex = tf.complex(x[..., 0], x[..., 1])

    if direction == 'C2R':
        out = tf.real(tf.ifft2d(x_complex)) * h * w
        return out

    else:
        if inverse:
            out = stack_real_imag(tf.ifft2d(x_complex)) * h * w
        else:
            out = stack_real_imag(tf.fft2d(x_complex))
        return out


def cdgmm(A, B):

    C_r = A[..., 0] * B[..., 0] - A[..., 1] * B[..., 1]
    C_i = A[..., 0] * B[..., 1] + A[..., 1] * B[..., 0]

    return tf.stack((C_r, C_i), -1)


def periodize(x, k):

    input_shape = x.get_shape().as_list()

    output_shape = [tf.shape(x)[0], input_shape[1], input_shape[2] // k, input_shape[3] // k]

    reshape_shape = [tf.shape(x)[0], input_shape[1],
                     input_shape[2] // output_shape[2], output_shape[2],
                     input_shape[3] // output_shape[3], output_shape[3]]

    x0 = x[..., 0]
    x1 = x[..., 1]

    y0 = tf.reshape(x0, tf.stack(reshape_shape))
    y1 = tf.reshape(x1, tf.stack(reshape_shape))

    y0 = tf.expand_dims(tf.reduce_mean(tf.reduce_mean(y0, axis=4), axis=2), axis=-1)
    y1 = tf.expand_dims(tf.reduce_mean(tf.reduce_mean(y1, axis=4), axis=2), axis=-1)

    out = tf.concat([y0, y1], axis=-1)
    return out


def modulus(x):

    input_shape = x.get_shape().as_list()

    out = tf.norm(x, axis=len(input_shape) - 1)
    out = tf.expand_dims(out, axis=-1)
    out = tf.concat([out, tf.zeros_like(out)], axis=-1)

    return out
