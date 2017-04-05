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
    def __init__(self, name="Generator"):

        super(Generator, self).__init__(name)
        self.name = name

    def __call__(self, x, reuse=False, output_name=None):
        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            # Initial dense multiplication
            x = layers.linear(x, "G_FC1", 512 * 8 * 8)

            batch_size = tf.shape(x)[0]
            if FLAGS.data_format == "NHWC":
                target_shape = (batch_size, 8, 8, 512)
            elif FLAGS.data_format == "NCHW":
                target_shape = (batch_size, 512, 8, 8)

            x = layers.reshape(x, target_shape)
            x = tf.contrib.layers.batch_norm(x, fused=True, data_format=FLAGS.data_format)
            x = layers.lrelu(x)

            x = layers.G_conv2d_block(x, "G_conv2D1", 256, 3, data_format=FLAGS.data_format, bn=True)
            x = layers.upsampleNN(x, "G_up1", 2, data_format=FLAGS.data_format)

            x = layers.G_conv2d_block(x, "G_conv2D2", 128, 3, data_format=FLAGS.data_format, bn=True)
            x = layers.upsampleNN(x, "G_up2", 2, data_format=FLAGS.data_format)

            x = layers.G_conv2d_block(x, "G_conv2D3", 64, 3, data_format=FLAGS.data_format, bn=True)
            x = layers.upsampleNN(x, "G_up3", 2, data_format=FLAGS.data_format)

            # Last conv
            x = layers.conv2d(x, "G_conv2D4", 64, FLAGS.channels, 3, 1, "SAME", data_format=FLAGS.data_format)

            x = tf.nn.tanh(x, name=output_name)

            return x


class Discriminator(Model):
    def __init__(self, name="D"):
        # Determine data format from output shape

        super(Discriminator, self).__init__(name)
        self.name = name

    def __call__(self, x, reuse=False, output_name=None):
        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            ##################
            # Encoding part
            ##################

            # First conv
            x = layers.conv2d(x, "D_conv2D1", FLAGS.channels, 32, 3, 1, "SAME", data_format=FLAGS.data_format)
            x = tf.nn.elu(x)

            # Conv blocks
            x = layers.D_conv2d_block(x, "D_enc_conv2D2", 64, 3, data_format=FLAGS.data_format)
            x = layers.D_conv2d_block(x, "D_enc_conv2D3", 128, 3, data_format=FLAGS.data_format)
            x = layers.D_conv2d_block(x, "D_enc_conv2D4", 256, 3, data_format=FLAGS.data_format)
            x = layers.D_conv2d_block(x, "D_enc_conv2D5", 256, 3, data_format=FLAGS.data_format)

            # strides = [1,1,1,1]
            # if FLAGS.data_format == "NCHW":
            #     ksize = [1,1,4,4]
            # else:
            #     ksize = [1,4,4,1]

            # x = tf.nn.avg_pool(x, ksize, strides, "VALID", data_format=FLAGS.data_format)
            # x = tf.squeeze(x, name="D_output")

            # ALTERNATIVELY:
            # Flatten
            batch_size = tf.shape(x)[0]
            other_dims = x.get_shape().as_list()[1:]
            prod_dim = 1
            for d in other_dims:
                prod_dim *= d
            x = layers.reshape(x, (batch_size, prod_dim))


            # Linear
            x = layers.linear(x, "D_FC1", 1, activation_fn=None)

            return x
