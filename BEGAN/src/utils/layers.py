import numpy as np
import tensorflow as tf


def reshape(x, target_shape):

    return tf.reshape(x, target_shape)


def linear(x, name, n_out, bias=True, activation_fn=None):
    """Dense layer"""

    with tf.variable_scope(name):

        n_in = x.shape[-1]

        # Initialize w
        w_init_std = np.sqrt(1.0 / n_out)
        w_init = tf.truncated_normal_initializer(0.0, w_init_std)
        w = tf.get_variable('w', shape=[n_in,n_out], initializer=w_init)

        # Dense mutliplication
        x = tf.matmul(x, w)

        if bias:

            # Initialize b
            b_init = tf.constant_initializer(0.0)
            b = tf.get_variable('b', shape=(n_out,), initializer=b_init)

            # Add b
            x = x + b

        if activation_fn:
            x = activation_fn(x)

        return x


def conv2d(x, name, n_in, n_out, k, s, p, bias=True, stddev=0.02, data_format="NCHW"):
    """Conv2D layer"""
    with tf.variable_scope(name):

        if data_format == "NHWC":
            strides = [1, s, s, 1]
        elif data_format == "NCHW":
            strides = [1, 1, s, s]

        # Initialize weigth
        w_init = tf.random_normal_initializer(stddev=stddev)
        w = tf.get_variable('w', [k, k, n_in, n_out], initializer=w_init)

        # Compute conv
        conv = tf.nn.conv2d(x, w, strides=strides, padding=p, data_format=data_format)

        if bias:
            # Initialize bias
            b_init = tf.constant_initializer(0.0)
            b = tf.get_variable('b', shape=(n_out,), initializer=b_init)

            # Add bias
            conv = tf.nn.bias_add(conv, b, data_format=data_format)

        return conv


def dec_conv2d_block(x, name, f, k, bias=True, bn=False, stddev=0.02, data_format="NCHW", activation_fn=tf.nn.elu):
    """Decoding 2D conv block: chain 2 convolutions with"""
    with tf.variable_scope(name):

        if data_format == "NHWC":
            n_in = x.get_shape()[-1]
        elif data_format == "NCHW":
            n_in = x.get_shape()[1]

        n_out = f

        stride = 1
        padding = "SAME"

        # First conv 2d layer
        x = conv2d(x, "conv2d_1", n_in, n_out, k, stride, padding, stddev=stddev, data_format=data_format, bias=bias)
        if bn:
            x = tf.contrib.layers.batch_norm(x, data_format=data_format, fused=True, scope="BatchNorm1")
        if activation_fn is not None:
            x = activation_fn(x)
        # Second conv 2d layer
        x = conv2d(x, "conv2d_2", n_out, n_out, k, stride, padding, stddev=stddev, data_format=data_format, bias=bias)
        if bn:
            x = tf.contrib.layers.batch_norm(x, data_format=data_format, fused=True, scope="BatchNorm2")
        if activation_fn is not None:
            x = activation_fn(x)

        return x


def upsampleNN(x, name, s, data_format="NCHW"):
    """ Upsample image data by a factor s with NN interpolation"""
    with tf.variable_scope(name):

        # Transpose before resize
        if data_format == "NCHW":
            x = tf.transpose(x, [0, 2, 3, 1])

        # Resize
        x_shape = x.get_shape().as_list()
        new_height = s * x_shape[1]
        new_width = s * x_shape[2]
        x = tf.image.resize_nearest_neighbor(x, (new_height, new_width))

        if data_format == "NCHW":
            x = tf.transpose(x, [0, 3, 1, 2])

        return x


def enc_conv2d_block(x, name, f, k, bias=True, bn=False, stddev=0.02, data_format="NCHW", activation_fn=tf.nn.elu, downsampling=True):
    """Encoding 2D conv block: chain 2 convolutions with x2 downsampling by default"""
    with tf.variable_scope(name):

        if data_format == "NHWC":
            n_in = x.get_shape()[-1]
        elif data_format == "NCHW":
            n_in = x.get_shape()[1]

        n_out = f

        stride_1, stride_2 = 1, 1
        if downsampling:
            stride_2 = 2
        padding = "SAME"

        # First conv 2d layer
        x = conv2d(x, "conv2d_1", n_in, n_out, k, stride_1, padding, stddev=stddev, data_format=data_format, bias=bias)
        if bn:
            x = tf.contrib.layers.batch_norm(x, data_format=data_format, fused=True, scope="BatchNorm1")
        if activation_fn is not None:
            x = activation_fn(x)
        # Second conv 2d layer
        x = conv2d(x, "conv2d_2", n_out, n_out, k, stride_2, padding, stddev=stddev, data_format=data_format, bias=bias)
        if bn:
            x = tf.contrib.layers.batch_norm(x, data_format=data_format, fused=True, scope="BatchNorm2")
        if activation_fn is not None:
            x = activation_fn(x)

        return x
