from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Convolution2D, Deconvolution2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
import numpy as np


def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)

    return x


def lambda_output(input_shape):
    return input_shape[:2]


# def conv_block_unet(x, f, name, bn_mode, bn_axis, bn=True, dropout=False, subsample=(2,2)):

#     x = Convolution2D(f, 3, 3, subsample=subsample, name=name, border_mode="same")(x)
#     if bn:
#         x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
#     x = LeakyReLU(0.2)(x)
#     if dropout:
#         x = Dropout(0.5)(x)

#     return x


# def up_conv_block_unet(x1, x2, f, name, bn_mode, bn_axis, bn=True, dropout=False):

#     x1 = UpSampling2D(size=(2, 2))(x1)
#     x = merge([x1, x2], mode='concat', concat_axis=bn_axis)

#     x = Convolution2D(f, 3, 3, name=name, border_mode="same")(x)
#     if bn:
#         x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
#     x = Activation("relu")(x)
#     if dropout:
#         x = Dropout(0.5)(x)

#     return x

def conv_block_unet(x, f, name, bn_mode, bn_axis, bn=True, subsample=(2,2)):

    x = LeakyReLU(0.2)(x)
    x = Convolution2D(f, 3, 3, subsample=subsample, name=name, border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)

    return x


def up_conv_block_unet(x, x2, f, name, bn_mode, bn_axis, bn=True, dropout=False):

    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(f, 3, 3, name=name, border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = merge([x, x2], mode='concat', concat_axis=bn_axis)

    return x


def deconv_block_unet(x, x2, f, h, w, batch_size, name, bn_mode, bn_axis, bn=True, dropout=False):

    o_shape = (batch_size, h * 2, w * 2, f)
    x = Activation("relu")(x)
    x = Deconvolution2D(f, 3, 3, output_shape=o_shape, subsample=(2, 2), border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = merge([x, x2], mode='concat', concat_axis=bn_axis)

    return x


def generator_unet_upsampling(img_dim, bn_mode, model_name="generator_unet_upsampling"):

    nb_filters = 64

    if K.image_dim_ordering() == "th":
        bn_axis = 1
        nb_channels = img_dim[0]
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        nb_channels = img_dim[-1]
        min_s = min(img_dim[:-1])

    unet_input = Input(shape=img_dim, name="unet_input")

    # Prepare encoder filters
    nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    # Encoder
    list_encoder = [Convolution2D(list_nb_filters[0], 3, 3,
                                  subsample=(2, 2), name="unet_conv2D_1", border_mode="same")(unet_input)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_conv2D_%s" % (i + 2)
        conv = conv_block_unet(list_encoder[-1], f, name, bn_mode, bn_axis)
        list_encoder.append(conv)

    # Prepare decoder filters
    list_nb_filters = list_nb_filters[:-2][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)

    # Decoder
    list_decoder = [up_conv_block_unet(list_encoder[-1], list_encoder[-2],
                                       list_nb_filters[0], "unet_upconv2D_1", bn_mode, bn_axis, dropout=True)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_upconv2D_%s" % (i + 2)
        # Dropout only on first few layers
        if i < 2:
            d = True
        else:
            d = False
        conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, name, bn_mode, bn_axis, dropout=d)
        list_decoder.append(conv)

    x = Activation("relu")(list_decoder[-1])
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(nb_channels, 3, 3, name="last_conv", border_mode="same")(x)
    x = Activation("tanh")(x)

    generator_unet = Model(input=[unet_input], output=[x])

    return generator_unet


def generator_unet_deconv(img_dim, bn_mode, batch_size, model_name="generator_unet_deconv"):

    assert K.backend() == "tensorflow", "Not implemented with theano backend"

    nb_filters = 64
    bn_axis = -1
    h, w, nb_channels = img_dim
    min_s = min(img_dim[:-1])

    unet_input = Input(shape=img_dim, name="unet_input")

    # Prepare encoder filters
    nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    # Encoder
    list_encoder = [Convolution2D(list_nb_filters[0], 3, 3,
                                  subsample=(2, 2), name="unet_conv2D_1", border_mode="same")(unet_input)]
    # update current "image" h and w
    h, w = h / 2, w / 2
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_conv2D_%s" % (i + 2)
        conv = conv_block_unet(list_encoder[-1], f, name, bn_mode, bn_axis)
        list_encoder.append(conv)
        h, w = h / 2, w / 2

    # Prepare decoder filters
    list_nb_filters = list_nb_filters[:-1][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)

    # Decoder
    list_decoder = [deconv_block_unet(list_encoder[-1], list_encoder[-2],
                                      list_nb_filters[0], h, w, batch_size,
                                      "unet_upconv2D_1", bn_mode, bn_axis, dropout=True)]
    h, w = h * 2, w * 2
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_upconv2D_%s" % (i + 2)
        # Dropout only on first few layers
        if i < 2:
            d = True
        else:
            d = False
        conv = deconv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, h,
                                 w, batch_size, name, bn_mode, bn_axis, dropout=d)
        list_decoder.append(conv)
        h, w = h * 2, w * 2

    x = Activation("relu")(list_decoder[-1])
    o_shape = (batch_size,) + img_dim
    x = Deconvolution2D(nb_channels, 3, 3, output_shape=o_shape, subsample=(2, 2), border_mode="same")(x)
    x = Activation("tanh")(x)

    generator_unet = Model(input=[unet_input], output=[x])

    return generator_unet


def DCGAN_discriminator(img_dim, nb_patch, bn_mode, model_name="DCGAN_discriminator", use_mbd=True):
    """
    Discriminator model of the DCGAN

    args : img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    list_input = [Input(shape=img_dim, name="disc_input_%s" % i) for i in range(nb_patch)]

    if K.image_dim_ordering() == "th":
        bn_axis = 1
    else:
        bn_axis = -1

    nb_filters = 64
    nb_conv = int(np.floor(np.log(img_dim[1]) / np.log(2)))
    list_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    # First conv
    x_input = Input(shape=img_dim, name="discriminator_input")
    x = Convolution2D(list_filters[0], 3, 3, subsample=(2, 2), name="disc_conv2d_1", border_mode="same")(x_input)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)

    # Next convs
    for i, f in enumerate(list_filters[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Convolution2D(f, 3, 3, subsample=(2, 2), name=name, border_mode="same")(x)
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)

    x_flat = Flatten()(x)
    x = Dense(2, activation='softmax', name="disc_dense")(x_flat)

    PatchGAN = Model(input=[x_input], output=[x, x_flat], name="PatchGAN")
    print("PatchGAN summary")
    PatchGAN.summary()

    x = [PatchGAN(patch)[0] for patch in list_input]
    x_mbd = [PatchGAN(patch)[1] for patch in list_input]

    if len(x) > 1:
        x = merge(x, mode="concat", name="merge_feat")
    else:
        x = x[0]

    if use_mbd:
        if len(x_mbd) > 1:
            x_mbd = merge(x_mbd, mode="concat", name="merge_feat_mbd")
        else:
            x_mbd = x_mbd[0]

        num_kernels = 100
        dim_per_kernel = 5

        M = Dense(num_kernels * dim_per_kernel, bias=False, activation=None)
        MBD = Lambda(minb_disc, output_shape=lambda_output)

        x_mbd = M(x_mbd)
        x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
        x_mbd = MBD(x_mbd)
        x = merge([x, x_mbd], mode='concat')

    x_out = Dense(2, activation="softmax", name="disc_output")(x)

    discriminator_model = Model(input=list_input, output=[x_out], name=model_name)

    return discriminator_model


def DCGAN(generator, discriminator_model, img_dim, patch_size, image_dim_ordering):

    gen_input = Input(shape=img_dim, name="DCGAN_input")

    generated_image = generator(gen_input)

    if image_dim_ordering == "th":
        h, w = img_dim[1:]
    else:
        h, w = img_dim[:-1]
    ph, pw = patch_size

    list_row_idx = [(i * ph, (i + 1) * ph) for i in range(h / ph)]
    list_col_idx = [(i * pw, (i + 1) * pw) for i in range(w / pw)]

    list_gen_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            if image_dim_ordering == "tf":
                x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(generated_image)
            else:
                x_patch = Lambda(lambda z: z[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]])(generated_image)
            list_gen_patch.append(x_patch)

    DCGAN_output = discriminator_model(list_gen_patch)

    DCGAN = Model(input=[gen_input],
                  output=[generated_image, DCGAN_output],
                  name="DCGAN")

    return DCGAN


def load(model_name, img_dim, nb_patch, bn_mode, use_mbd, batch_size):

    if model_name == "generator_unet_upsampling":
        model = generator_unet_upsampling(img_dim, bn_mode, model_name=model_name)
        print model.summary()
        from keras.utils.visualize_util import plot
        plot(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model

    if model_name == "generator_unet_deconv":
        model = generator_unet_deconv(img_dim, bn_mode, batch_size, model_name=model_name)
        print model.summary()
        from keras.utils.visualize_util import plot
        plot(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model

    if model_name == "DCGAN_discriminator":
        model = DCGAN_discriminator(img_dim, nb_patch, bn_mode, model_name=model_name, use_mbd=use_mbd)
        model.summary()
        from keras.utils.visualize_util import plot
        plot(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model


if __name__ == '__main__':

    # load("generator_unet_deconv", (256, 256, 3), 16, 2, False, 32)
    load("generator_unet_upsampling", (256, 256, 3), 16, 2, False, 32)
