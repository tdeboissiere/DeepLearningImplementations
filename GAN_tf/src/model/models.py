import sys
import tensorflow as tf
import collections
sys.path.append("../utils")
import layers


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
                # list_v = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
                # for v in list_v:
                #     print v
                # print
                # print
                # for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
                #     print v
                # import ipdb; ipdb.set_trace()
                scope.reuse_variables()

            # Store all layers in a dict
            d = collections.OrderedDict()

            # Initial dense multiplication
            x = layers.linear(x, self.filters * self.start_dim * self.start_dim)

            # Reshape to image format
            if self.data_format == "NCHW":
                target_shape = (self.batch_size, self.filters, self.start_dim, self.start_dim)
            else:
                target_shape = (self.batch_size, self.start_dim, self.start_dim, self.filters)

            x = layers.reshape(x, target_shape)
            x = tf.contrib.layers.batch_norm(x, fused=True)
            x = tf.nn.relu(x)

            import ipdb; ipdb.set_trace()

            # # Conv2D + Phase shift blocks
            # x = layers.conv2d_block("conv2D_1_1", x, 512, 3, 1, p="SAME", stddev=0.02,
            #                         data_format=self.data_format, bias=False, bn=True, activation_fn=layers.lrelu)
            # x = layers.conv2d_block("conv2D_1_2", x, 512, 3, 1, p="SAME", stddev=0.02,
            #                         data_format=self.data_format, bias=False, bn=False, activation_fn=layers.lrelu)
            # x = layers.phase_shift(x, upsampling_factor=2, name="PS1")

            # x = layers.conv2d_block("conv2D_2_1", x, 256, 3, 1, p="SAME", stddev=0.02,
            #                         data_format=self.data_format, bias=False, bn=False, activation_fn=layers.lrelu)
            # x = layers.conv2d_block("conv2D_2_2", x, 256, 3, 1, p="SAME", stddev=0.02,
            #                         data_format=self.data_format, bias=False, bn=False, activation_fn=layers.lrelu)
            # x = layers.phase_shift(x, upsampling_factor=2, name="PS2")

            # x = layers.conv2d_block("conv2D_3", x, 1, 1, 1, p="SAME", stddev=0.02,
            #                         data_format=self.data_format, bn=False)

            # # Upsampling2D + conv blocks
            # for idx, (f, k, s, p) in enumerate(zip(self.list_filters, self.list_kernel_size, self.list_strides, self.list_padding)):
            #     name = "upsample2D_%s" % idx
            #     if idx == len(self.list_filters) - 1:
            #         bn = False
            #     else:
            #         bn = True
            #     x = layers.upsample2d_block(name, x, f, k, s, p, data_format=self.data_format, bn=bn, activation_fn=layers.lrelu)

            # Transposed conv blocks
            for idx, (f, k, s, p) in enumerate(zip(self.list_filters, self.list_kernel_size, self.list_strides, self.list_padding)):
                img_size = self.start_dim * (2 ** (idx + 1))
                if self.data_format == "NCHW":
                    output_shape = (self.batch_size, f, img_size, img_size)
                else:
                    output_shape = (self.batch_size, img_size, img_size, f)
                name = "deconv2D_%s" % idx
                if idx == len(self.list_filters) - 1:
                    bn = False
                else:
                    bn = True
                x = layers.deconv2d_block(name, x, output_shape, k, s, p, data_format=self.data_format, bn=bn)

            x = tf.nn.tanh(x, name="X_G")

            return x


class Discriminator(Model):
    def __init__(self, list_filters, list_kernel_size, list_strides, list_padding, batch_size,
                 name="discriminator", data_format="NCHW"):
        # Determine data format from output shape

        super(Discriminator, self).__init__(name)

        self.data_format = data_format
        self.name = name
        self.list_filters = list_filters
        self.list_strides = list_strides
        self.list_kernel_size = list_kernel_size
        self.batch_size = batch_size
        self.list_padding = list_padding

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            for idx, (f, k, s, p) in enumerate(zip(self.list_filters, self.list_kernel_size, self.list_strides, self.list_padding)):
                if idx == 0:
                    bn = False
                else:
                    bn = True
                name = "conv2D_%s" % idx
                x = layers.conv2d_block(name, x, f, k, s, p=p, stddev=0.02,
                                        data_format=self.data_format, bias=True, bn=bn, activation_fn=layers.lrelu)

            target_shape = (self.batch_size, -1)
            x = layers.reshape(x, target_shape)

            # # Add MBD
            # x_mbd = layers.mini_batch_disc(x, num_kernels=100, dim_per_kernel=5)
            # # Concat
            # x = tf.concat([x, x_mbd], axis=1)

            x = layers.linear(x, 1)

            return x
