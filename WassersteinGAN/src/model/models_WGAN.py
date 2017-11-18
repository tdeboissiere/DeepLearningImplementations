import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, UpSampling2D
from keras.layers.pooling import GlobalAveragePooling2D


def wasserstein(y_true, y_pred):

    # return K.mean(y_true * y_pred) / K.mean(y_true)
    return K.mean(y_true * y_pred)


def visualize_model(model):

    model.summary()
    from keras.utils import plot_model
    plot_model(model,
               to_file='../../figures/%s.png' % model.name,
               show_shapes=True,
               show_layer_names=True)


def generator_toy(noise_dim, model_name="generator_toy"):
    """
    Simple MLP generator for the MoG unrolled GAN toy experiment
    """

    gen_input = Input(shape=noise_dim, name="generator_input")

    x = Dense(128)(gen_input)
    x = Activation("tanh")(x)
    x = Dense(128)(x)
    x = Activation("tanh")(x)
    x = Dense(2)(x)

    generator_model = Model(inputs=[gen_input], outputs=[x], name=model_name)
    visualize_model(generator_model)

    return generator_model


def discriminator_toy(model_name="discriminator_toy"):
    """
    Simple MLP discriminator for the MoG unrolled GAN toy experiment
    """

    disc_input = Input(shape=(2,), name="discriminator_input")

    x = Dense(128)(disc_input)
    x = Activation("tanh")(x)
    x = Dense(128)(x)
    x = Activation("tanh")(x)
    x = Dense(1)(x)

    discriminator_model = Model(inputs=[disc_input], outputs=[x], name=model_name)
    visualize_model(discriminator_model)

    return discriminator_model


def GAN_toy(generator, discriminator, noise_dim):
    """
    Simple GAN genrator + discriminator for the MoG unrolled GAN toy experiment
    """

    gen_input = Input(shape=noise_dim, name="noise_input")
    generated_sample = generator(gen_input)
    GAN_output = discriminator(generated_sample)

    GAN_toy = Model(inputs=[gen_input],
                    outputs=[GAN_output],
                    name="GAN_toy")
    visualize_model(GAN_toy)

    return GAN_toy


def generator_upsampling(noise_dim, img_dim, bn_mode, model_name="generator_upsampling", dset="mnist"):
    """DCGAN generator based on Upsampling and Conv2D

    Args:
        noise_dim: Dimension of the noise input
        img_dim: dimension of the image output
        bn_mode: keras batchnorm mode
        model_name: model name (default: {"generator_upsampling"})
        dset: dataset (default: {"mnist"})

    Returns:
        keras model
    """

    s = img_dim[1]
    f = 512

    if dset == "mnist":
        start_dim = int(s / 4)
        nb_upconv = 2
    else:
        start_dim = int(s / 16)
        nb_upconv = 4

    if K.image_dim_ordering() == "th":
        bn_axis = 1
        reshape_shape = (f, start_dim, start_dim)
        output_channels = img_dim[0]
    else:
        reshape_shape = (start_dim, start_dim, f)
        bn_axis = -1
        output_channels = img_dim[-1]

    gen_input = Input(shape=noise_dim, name="generator_input")

    # Noise input and reshaping
    x = Dense(f * start_dim * start_dim, input_dim=noise_dim)(gen_input)
    x = Reshape(reshape_shape)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation("relu")(x)

    # Upscaling blocks: Upsampling2D->Conv2D->ReLU->BN->Conv2D->ReLU
    for i in range(nb_upconv):
        x = UpSampling2D(size=(2, 2))(x)
        nb_filters = int(f / (2 ** (i + 1)))
        x = Conv2D(nb_filters, (3, 3), padding="same", kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(nb_filters, (3, 3), padding="same", kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = Activation("relu")(x)

    # Last Conv to get the output image
    x = Conv2D(output_channels, (3, 3), name="gen_conv2d_final",
               padding="same", activation='tanh', kernel_initializer=RandomNormal(stddev=0.02))(x)

    generator_model = Model(inputs=[gen_input], outputs=[x], name=model_name)
    visualize_model(generator_model)

    return generator_model


def generator_deconv(noise_dim, img_dim, bn_mode, batch_size, model_name="generator_deconv", dset="mnist"):
    """DCGAN generator based on Deconv2D

    Args:
        noise_dim: Dimension of the noise input
        img_dim: dimension of the image output
        bn_mode: keras batchnorm mode
        batch_size: needed to reshape after the deconv2D
        model_name: model name (default: {"generator_deconv"})
        dset: dataset (default: {"mnist"})

    Returns:
        keras model
    """

    assert K.backend() == "tensorflow", "Deconv not implemented with theano"

    s = img_dim[1]
    f = 512

    if dset == "mnist":
        start_dim = int(s / 4)
        nb_upconv = 2
    else:
        start_dim = int(s / 16)
        nb_upconv = 4

    reshape_shape = (start_dim, start_dim, f)
    bn_axis = -1
    output_channels = img_dim[-1]

    gen_input = Input(shape=noise_dim, name="generator_input")

    # Noise input and reshaping
    x = Dense(f * start_dim * start_dim, input_dim=noise_dim, use_bias=False)(gen_input)
    x = Reshape(reshape_shape)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation("relu")(x)

    # Transposed conv blocks: Deconv2D->BN->ReLU
    for i in range(nb_upconv - 1):
        nb_filters = int(f / (2 ** (i + 1)))
        s = start_dim * (2 ** (i + 1))
        o_shape = (batch_size, s, s, nb_filters)
        x = Deconv2D(nb_filters, (3, 3),
                     output_shape=o_shape, strides=(2, 2),
                     padding="same", use_bias=False,
                     kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

    # Last block
    s = start_dim * (2 ** (nb_upconv))
    o_shape = (batch_size, s, s, output_channels)
    x = Deconv2D(output_channels, (3, 3),
                 output_shape=o_shape, strides=(2, 2),
                 padding="same", use_bias=False,
                 kernel_initializer=RandomNormal(stddev=0.02))(x)
    x = Activation("tanh")(x)

    generator_model = Model(inputs=[gen_input], outputs=[x], name=model_name)
    visualize_model(generator_model)

    return generator_model


def discriminator(img_dim, bn_mode, model_name="discriminator"):
    """DCGAN discriminator

    Args:
        img_dim: dimension of the image output
        bn_mode: keras batchnorm mode
        model_name: model name (default: {"generator_deconv"})

    Returns:
        keras model
    """

    if K.image_dim_ordering() == "th":
        bn_axis = 1
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        min_s = min(img_dim[:-1])

    disc_input = Input(shape=img_dim, name="discriminator_input")

    # Get the list of number of conv filters
    # (first layer starts with 64), filters are subsequently doubled
    nb_conv = int(np.floor(np.log(min_s // 4) / np.log(2)))
    list_f = [64 * min(8, (2 ** i)) for i in range(nb_conv)]

    # First conv with 2x2 strides
    x = Conv2D(list_f[0], (3, 3), strides=(2, 2), name="disc_conv2d_1",
               padding="same", use_bias=False,
               kernel_initializer=RandomNormal(stddev=0.02))(disc_input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)

    # Conv blocks: Conv2D(2x2 strides)->BN->LReLU
    for i, f in enumerate(list_f[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Conv2D(f, (3, 3), strides=(2, 2), name=name, padding="same", use_bias=False,
                   kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalization(axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)

    # Last convolution
    x = Conv2D(1, (3, 3), name="last_conv", padding="same", use_bias=False,
               kernel_initializer=RandomNormal(stddev=0.02))(x)
    # Average pooling
    x = GlobalAveragePooling2D()(x)

    discriminator_model = Model(inputs=[disc_input], outputs=[x], name=model_name)
    visualize_model(discriminator_model)

    return discriminator_model


def DCGAN(generator, discriminator, noise_dim, img_dim):
    """DCGAN generator + discriminator model

    Args:
        generator: keras generator model
        discriminator: keras discriminator model
        noise_dim: generator input noise dimension
        img_dim: real image data dimension

    Returns:
        keras model
    """

    noise_input = Input(shape=noise_dim, name="noise_input")
    generated_image = generator(noise_input)
    DCGAN_output = discriminator(generated_image)

    DCGAN = Model(inputs=[noise_input],
                  outputs=[DCGAN_output],
                  name="DCGAN")
    visualize_model(DCGAN)

    return DCGAN
