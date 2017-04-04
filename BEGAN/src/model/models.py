import sys
import tensorflow as tf
sys.path.append("../utils")
import layers

FLAGS = tf.app.flags.FLAGS


class Model(object):

    def __init__(self, name):
        self.name = name

    def get_trainable_variables(self):
        t_vars = tf.trainable_variables()
        t_vars_model = {v.name: v for v in t_vars if self.name in v.name}
        return t_vars_model


class Generator(Model):
    def __init__(self, name="Generator", nb_filters=64):

        super(Generator, self).__init__(name)
        self.name = name
        self.nb_filters = nb_filters

    def __call__(self, x, reuse=False, output_name=None):
        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            # Initial dense multiplication
            x = layers.linear(x, "G_FC1", self.nb_filters * 8 * 8)

            batch_size = tf.shape(x)[0]
            if FLAGS.data_format == "NHWC":
                target_shape = (batch_size, 8, 8, self.nb_filters)
            elif FLAGS.data_format == "NCHW":
                target_shape = (batch_size, self.nb_filters, 8, 8)

            x = layers.reshape(x, target_shape)
            # x = tf.contrib.layers.batch_norm(x, fused=True, data_format=FLAGS.data_format)
            x = tf.nn.elu(x)

            x = layers.dec_conv2d_block(x, "G_conv2D1", self.nb_filters, 3, data_format=FLAGS.data_format)
            x = layers.upsampleNN(x, "G_up1", 2, data_format=FLAGS.data_format)

            x = layers.dec_conv2d_block(x, "G_conv2D2", self.nb_filters, 3, data_format=FLAGS.data_format)
            x = layers.upsampleNN(x, "G_up2", 2, data_format=FLAGS.data_format)

            x = layers.dec_conv2d_block(x, "G_conv2D3", self.nb_filters, 3, data_format=FLAGS.data_format)
            x = layers.upsampleNN(x, "G_up3", 2, data_format=FLAGS.data_format)

            x = layers.dec_conv2d_block(x, "G_conv2D4", self.nb_filters, 3, data_format=FLAGS.data_format)

            # Last conv
            x = layers.conv2d(x, "G_conv2D5", self.nb_filters, FLAGS.channels, 3, 1, "SAME", data_format=FLAGS.data_format)

            x = tf.nn.tanh(x, name=output_name)

            return x


class Discriminator(Model):
    def __init__(self, name="D", h_dim=128, nb_filters=64):
        # Determine data format from output shape

        super(Discriminator, self).__init__(name)
        self.name = name
        self.h_dim = h_dim
        self.nb_filters = 64

    def __call__(self, x, reuse=False, output_name=None):
        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            ##################
            # Encoding part
            ##################

            # First conv
            x = layers.conv2d(x, "D_conv2D1", FLAGS.channels, self.nb_filters, 3, 1, "SAME", data_format=FLAGS.data_format)
            x = tf.nn.elu(x)

            # Conv blocks
            x = layers.enc_conv2d_block(x, "D_enc_conv2D2", self.nb_filters, 3, activation_fn=tf.nn.elu, data_format=FLAGS.data_format)
            x = layers.enc_conv2d_block(x, "D_enc_conv2D3", 2 * self.nb_filters, 3, activation_fn=tf.nn.elu, data_format=FLAGS.data_format)
            x = layers.enc_conv2d_block(x, "D_enc_conv2D4", 3 * self.nb_filters, 3, activation_fn=tf.nn.elu, data_format=FLAGS.data_format)
            x = layers.enc_conv2d_block(x, "D_enc_conv2D5", 4 * self.nb_filters, 3, activation_fn=tf.nn.elu, data_format=FLAGS.data_format, downsampling=False)

            # Flatten
            batch_size = tf.shape(x)[0]
            other_dims = x.get_shape().as_list()[1:]
            prod_dim = 1
            for d in other_dims:
                prod_dim *= d
            x = layers.reshape(x, (batch_size, prod_dim))

            # Linear
            x = layers.linear(x, "D_FC1", self.h_dim, activation_fn=None)

            ##################
            # Decoding part
            ##################

            x = layers.linear(x, "D_FC2", self.nb_filters * 8 * 8)

            batch_size = tf.shape(x)[0]
            if FLAGS.data_format == "NHWC":
                target_shape = (batch_size, 8, 8, self.nb_filters)
            elif FLAGS.data_format == "NCHW":
                target_shape = (batch_size, self.nb_filters, 8, 8)

            x = layers.reshape(x, target_shape)
            # x = tf.contrib.layers.batch_norm(x, fused=True, data_format=FLAGS.data_format)
            x = tf.nn.elu(x)

            x = layers.dec_conv2d_block(x, "D_dec_conv2D1", self.nb_filters, 3, data_format=FLAGS.data_format)
            x = layers.upsampleNN(x, "D_up1", 2, data_format=FLAGS.data_format)

            x = layers.dec_conv2d_block(x, "D_dec_conv2D2", self.nb_filters, 3, data_format=FLAGS.data_format)
            x = layers.upsampleNN(x, "D_up2", 2, data_format=FLAGS.data_format)

            x = layers.dec_conv2d_block(x, "D_dec_conv2D3", self.nb_filters, 3, data_format=FLAGS.data_format)
            x = layers.upsampleNN(x, "D_up3", 2, data_format=FLAGS.data_format)

            x = layers.dec_conv2d_block(x, "D_dec_conv2D4", self.nb_filters, 3, data_format=FLAGS.data_format)

            # Last conv
            x = layers.conv2d(x, "D_dec_conv2D5", self.nb_filters, FLAGS.channels, 3, 1, "SAME", data_format=FLAGS.data_format)
            x = tf.nn.tanh(x, name=output_name)

            return x
