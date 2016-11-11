import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, merge
from keras.layers.core import Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, AtrousConvolution2D, UpSampling2D


def residual_block(x, nb_filter, block_idx, bn=True, weight_decay=0):

    # 1st conv
    name = "block%s_conv2D%s" % (block_idx, "a")
    W_reg = l2(weight_decay)
    r = Convolution2D(nb_filter, 3, 3, border_mode="same", W_regularizer=W_reg, name=name)(x)
    if bn:
        r = BatchNormalization(mode=2, axis=1, name="block%s_bn%s" % (block_idx, "a"))(r)
    r = Activation("relu", name="block%s_relu%s" % (block_idx, "a"))(r)

    # 2nd conv
    name = "block%s_conv2D%s" % (block_idx, "b")
    W_reg = l2(weight_decay)
    r = Convolution2D(nb_filter, 3, 3, border_mode="same", W_regularizer=W_reg, name=name)(r)
    if bn:
        r = BatchNormalization(mode=2, axis=1, name="block%s_bn%s" % (block_idx, "b"))(r)
    r = Activation("relu", name="block%s_relu%s" % (block_idx, "b"))(r)

    # Merge residual and identity
    x = merge([x, r], mode='sum', concat_axis=1, name="block%s_merge" % block_idx)

    return x


def convolutional_block(x, block_idx, nb_filter, nb_conv, subsample):

    # 1st conv
    for i in range(nb_conv):
        name = "block%s_conv2D_%s" % (block_idx, i)
        if i < nb_conv - 1:
            x = Convolution2D(nb_filter, 3, 3, name=name, border_mode="same")(x)
            x = BatchNormalization(mode=2, axis=1)(x)
            x = Activation("relu")(x)
        else:
            x = Convolution2D(nb_filter, 3, 3, name=name, subsample=subsample, border_mode="same")(x)
            x = BatchNormalization(mode=2, axis=1)(x)
            x = Activation("relu")(x)

    return x


def atrous_block(x, block_idx, nb_filter, nb_conv):

    # 1st conv
    for i in range(nb_conv):
        name = "block%s_conv2D_%s" % (block_idx, i)
        x = AtrousConvolution2D(nb_filter, 3, 3, name=name, border_mode="same")(x)
        x = BatchNormalization(mode=2, axis=1)(x)

        x = Activation("relu")(x)

    return x


def simple_colorful(nb_classes, img_dim, batch_size, model_name="colorful_simple"):

    nb_resblocks = 3
    block_idx = 0
    h, w = img_dim[1:]

    # First conv block
    x_input = Input(shape=img_dim, name="input")
    x = Convolution2D(64, 3, 3, name="block%s_conv2d_0" % block_idx, border_mode="same")(x_input)
    x = Activation("relu", name="block%s_relu" % block_idx)(x)
    block_idx += 1

    # Residual blocks
    for idx, f in enumerate([64] * nb_resblocks):
        x = residual_block(x, f, block_idx, weight_decay=0)
        block_idx += 1

    # Final conv
    x = Convolution2D(nb_classes, 1, 1, name="final_conv2D", border_mode="same")(x)

    # Reshape Softmax
    def output_shape(input_shape):
        return (batch_size, h, w, nb_classes + 1)

    def reshape_softmax(x):
        x = K.permute_dimensions(x, [0, 2, 3, 1])  # last dimension in number of filters
        x = K.reshape(x, (batch_size * h * w, nb_classes))
        x = K.softmax(x)
        # Add a zero column so that x has the same dimension as the target (313 classes + 1 weight)
        xc = K.zeros((batch_size * h * w, 1))
        x = K.concatenate([x, xc], axis=1)
        # Reshape back to (batch_size, h, w, nb_classes + 1) to satisfy keras' shape checks
        x = K.reshape(x, (batch_size, h, w, nb_classes + 1))
        return x

    ReshapeSoftmax = Lambda(lambda z: reshape_softmax(z), output_shape=output_shape, name="ReshapeSoftmax")
    x = ReshapeSoftmax(x)

    # Build model
    colorful_simple = Model(input=[x_input], output=[x], name=model_name)

    return colorful_simple


def colorful(nb_classes, img_dim, batch_size, model_name="colorful"):
    """
    """
    h, w = img_dim[1:]

    x_input = Input(shape=img_dim, name="input")

    # Keep track of image h and w
    current_h, current_w = img_dim[1:]

    # Convolutional blocks parameters
    list_filter_size = [64, 128, 256, 512, 512]
    list_block_size = [2, 2, 3, 3, 3]
    subsample = [(2,2), (2,2), (2,2), (1,1), (1,1)]

    # A trous blocks parameters
    list_filter_size_atrous = [512, 512]
    list_block_size_atrous = [3, 3]

    block_idx = 0

    # First block
    f, b, s = list_filter_size[0], list_block_size[0], subsample[0]
    x = convolutional_block(x_input, block_idx, f, b, s)
    block_idx += 1
    current_h, current_w = current_h / s[0], current_w / s[1]

    # Next blocks
    for f, b, s in zip(list_filter_size[1:-1], list_block_size[1:-1], subsample[1:-1]):
        x = convolutional_block(x, block_idx, f, b, s)
        block_idx += 1
        current_h, current_w = current_h / s[0], current_w / s[1]

    # Atrous blocks
    for idx, (f, b) in enumerate(zip(list_filter_size_atrous, list_block_size_atrous)):
        x = atrous_block(x, block_idx, f, b)
        block_idx += 1

    # Block 7
    f, b, s = list_filter_size[-1], list_block_size[-1], subsample[-1]
    x = convolutional_block(x, block_idx, f, b, s)
    block_idx += 1
    current_h, current_w = current_h / s[0], current_w / s[1]

    # Block 8
    # Not using Deconvolution at the moment
    # x = Deconvolution2D(256, 2, 2,
    #                     output_shape=(None, 256, current_h * 2, current_w * 2),
    #                     subsample=(2, 2),
    #                     border_mode="valid")(x)
    x = UpSampling2D(size=(2,2), name="upsampling2d")(x)

    x = convolutional_block(x, block_idx, 256, 2, (1, 1))
    block_idx += 1
    current_h, current_w = current_h * 2, current_w * 2

    # Final conv
    x = Convolution2D(nb_classes, 1, 1, name="conv2d_final", border_mode="same")(x)

    # Reshape Softmax
    def output_shape(input_shape):
        return (batch_size, h, w, nb_classes + 1)

    def reshape_softmax(x):
        x = K.permute_dimensions(x, [0, 2, 3, 1])  # last dimension in number of filters
        x = K.reshape(x, (batch_size * current_h * current_w, nb_classes))
        x = K.softmax(x)
        # Add a zero column so that x has the same dimension as the target (313 classes + 1 weight)
        xc = K.zeros((batch_size * current_h * current_w, 1))
        x = K.concatenate([x, xc], axis=1)
        # Reshape back to (batch_size, h, w, nb_classes + 1) to satisfy keras' shape checks
        x = K.reshape(x, (batch_size, current_h, current_w, nb_classes + 1))
        x = K.resize_images(x, h / current_h, w / current_w, "tf")
        # x = K.permute_dimensions(x, [0, 3, 1, 2])
        return x

    ReshapeSoftmax = Lambda(lambda z: reshape_softmax(z), output_shape=output_shape, name="ReshapeSoftmax")
    x = ReshapeSoftmax(x)

    # Build model
    colorful = Model(input=[x_input], output=[x], name=model_name)

    return colorful


def load(model_name, nb_classes, img_dim, batch_size):

    if model_name == "colorful":
        model = colorful(nb_classes, img_dim, batch_size, model_name=model_name)

    if model_name == "simple_colorful":
        model = simple_colorful(nb_classes, img_dim, batch_size, model_name=model_name)

    return model
