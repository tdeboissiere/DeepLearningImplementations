from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Convolution2D, Deconvolution2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K


def generator_upsampling(cat_dim, cont_dim, noise_dim, img_dim, bn_mode, model_name="generator_upsampling", dset="mnist"):
    """
    Generator model of the DCGAN

    args : img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    s = img_dim[1]
    f = 128

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

    cat_input = Input(shape=cat_dim, name="cat_input")
    cont_input = Input(shape=cont_dim, name="cont_input")
    noise_input = Input(shape=noise_dim, name="noise_input")

    gen_input = merge([cat_input, cont_input, noise_input], mode="concat")

    x = Dense(1024)(gen_input)
    x = BatchNormalization(mode=1)(x)
    x = Activation("relu")(x)

    x = Dense(f * start_dim * start_dim)(x)
    x = BatchNormalization(mode=1)(x)
    x = Activation("relu")(x)

    x = Reshape(reshape_shape)(x)

    # Upscaling blocks
    for i in range(nb_upconv):
        x = UpSampling2D(size=(2, 2))(x)
        nb_filters = int(f / (2 ** (i + 1)))
        x = Convolution2D(nb_filters, 3, 3, border_mode="same")(x)
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
        x = Activation("relu")(x)
        # x = Convolution2D(nb_filters, 3, 3, border_mode="same")(x)
        # x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
        # x = Activation("relu")(x)

    x = Convolution2D(output_channels, 3, 3, name="gen_convolution2d_final", border_mode="same", activation='tanh')(x)

    generator_model = Model(input=[cat_input, cont_input, noise_input], output=[x], name=model_name)

    return generator_model


def generator_deconv(cat_dim, cont_dim, noise_dim, img_dim, bn_mode, batch_size, model_name="generator_deconv", dset="mnist"):
    """
    Generator model of the DCGAN

    args : nb_classes (int) number of classes
           img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    assert K.backend() == "tensorflow", "Deconv not implemented with theano"

    s = img_dim[1]
    f = 128

    if dset == "mnist":
        start_dim = int(s / 4)
        nb_upconv = 2
    else:
        start_dim = int(s / 16)
        nb_upconv = 4

    reshape_shape = (start_dim, start_dim, f)
    bn_axis = -1
    output_channels = img_dim[-1]

    cat_input = Input(shape=cat_dim, name="cat_input")
    cont_input = Input(shape=cont_dim, name="cont_input")
    noise_input = Input(shape=noise_dim, name="noise_input")

    gen_input = merge([cat_input, cont_input, noise_input], mode="concat")

    x = Dense(1024)(gen_input)
    x = BatchNormalization(mode=1)(x)
    x = Activation("relu")(x)

    x = Dense(f * start_dim * start_dim)(x)
    x = BatchNormalization(mode=1)(x)
    x = Activation("relu")(x)

    x = Reshape(reshape_shape)(x)

    # Transposed conv blocks
    for i in range(nb_upconv - 1):
        nb_filters = int(f / (2 ** (i + 1)))
        s = start_dim * (2 ** (i + 1))
        o_shape = (batch_size, s, s, nb_filters)
        x = Deconvolution2D(nb_filters, 3, 3, output_shape=o_shape, subsample=(2, 2), border_mode="same")(x)
        x = BatchNormalization(mode=2, axis=bn_axis)(x)
        x = Activation("relu")(x)

    # Last block
    s = start_dim * (2 ** (nb_upconv))
    o_shape = (batch_size, s, s, output_channels)
    x = Deconvolution2D(output_channels, 3, 3, output_shape=o_shape, subsample=(2, 2), border_mode="same")(x)
    x = Activation("tanh")(x)

    generator_model = Model(input=[cat_input, cont_input, noise_input], output=[x], name=model_name)

    return generator_model


def DCGAN_discriminator(cat_dim, cont_dim, img_dim, bn_mode, model_name="DCGAN_discriminator", dset="mnist", use_mbd=False):
    """
    Discriminator model of the DCGAN

    args : img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    if K.image_dim_ordering() == "th":
        bn_axis = 1
    else:
        bn_axis = -1

    disc_input = Input(shape=img_dim, name="discriminator_input")

    if dset == "mnist":
        list_f = [128]

    else:
        list_f = [64, 128, 256]

    # First conv
    x = Convolution2D(64, 3, 3, subsample=(2, 2), name="disc_convolution2d_1", border_mode="same")(disc_input)
    x = LeakyReLU(0.2)(x)

    # Next convs
    for i, f in enumerate(list_f):
        name = "disc_convolution2d_%s" % (i + 2)
        x = Convolution2D(f, 3, 3, subsample=(2, 2), name=name, border_mode="same")(x)
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = BatchNormalization(mode=1)(x)
    x = LeakyReLU(0.2)(x)

    def linmax(x):
        return K.maximum(x, -16)

    def linmax_shape(input_shape):
        return input_shape

    # More processing for auxiliary Q
    x_Q = Dense(128)(x)
    x_Q = BatchNormalization(mode=1)(x_Q)
    x_Q = LeakyReLU(0.2)(x_Q)
    x_Q_Y = Dense(cat_dim[0], activation='softmax', name="Q_cat_out")(x_Q)
    x_Q_C_mean = Dense(cont_dim[0], activation='linear', name="dense_Q_cont_mean")(x_Q)
    x_Q_C_logstd = Dense(cont_dim[0], name="dense_Q_cont_logstd")(x_Q)
    x_Q_C_logstd = Lambda(linmax, output_shape=linmax_shape)(x_Q_C_logstd)
    # Reshape Q to nbatch, 1, cont_dim[0]
    x_Q_C_mean = Reshape((1, cont_dim[0]))(x_Q_C_mean)
    x_Q_C_logstd = Reshape((1, cont_dim[0]))(x_Q_C_logstd)
    x_Q_C = merge([x_Q_C_mean, x_Q_C_logstd], mode="concat", name="Q_cont_out", concat_axis=1)

    def minb_disc(z):
        diffs = K.expand_dims(z, 3) - K.expand_dims(K.permute_dimensions(z, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), 2)
        z = K.sum(K.exp(-abs_diffs), 2)

        return z

    def lambda_output(input_shape):
        return input_shape[:2]

    num_kernels = 300
    dim_per_kernel = 5

    M = Dense(num_kernels * dim_per_kernel, bias=False, activation=None)
    MBD = Lambda(minb_disc, output_shape=lambda_output)

    if use_mbd:
        x_mbd = M(x)
        x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
        x_mbd = MBD(x_mbd)
        x = merge([x, x_mbd], mode='concat')

    # Create discriminator model
    x_disc = Dense(2, activation='softmax', name="disc_out")(x)
    discriminator_model = Model(input=[disc_input], output=[x_disc, x_Q_Y, x_Q_C], name=model_name)

    return discriminator_model


def DCGAN(generator, discriminator_model, cat_dim, cont_dim, noise_dim):

    cat_input = Input(shape=cat_dim, name="cat_input")
    cont_input = Input(shape=cont_dim, name="cont_input")
    noise_input = Input(shape=noise_dim, name="noise_input")

    generated_image = generator([cat_input, cont_input, noise_input])
    x_disc, x_Q_Y, x_Q_C = discriminator_model(generated_image)

    DCGAN = Model(input=[cat_input, cont_input, noise_input],
                  output=[x_disc, x_Q_Y, x_Q_C],
                  name="DCGAN")

    return DCGAN


def load(model_name, cat_dim, cont_dim, noise_dim, img_dim, bn_mode, batch_size, dset="mnist", use_mbd=False):

    if model_name == "generator_upsampling":
        model = generator_upsampling(cat_dim, cont_dim, noise_dim, img_dim, bn_mode, model_name=model_name, dset=dset)
        print model.summary()
        from keras.utils.visualize_util import plot
        plot(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model
    if model_name == "generator_deconv":
        model = generator_deconv(cat_dim, cont_dim, noise_dim, img_dim, bn_mode, batch_size, model_name=model_name, dset=dset)
        print model.summary()
        from keras.utils.visualize_util import plot
        plot(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model
    if model_name == "DCGAN_discriminator":
        model = DCGAN_discriminator(cat_dim, cont_dim, img_dim, bn_mode, model_name=model_name, dset=dset, use_mbd=use_mbd)
        model.summary()
        from keras.utils.visualize_util import plot
        plot(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model


if __name__ == '__main__':

    m = generator_deconv((10,), (2,), (64,), (28, 28, 1), 2, 1, model_name="generator_deconv", dset="mnist")
    m.summary()
