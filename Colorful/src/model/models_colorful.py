import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Add
from keras.layers.core import Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, UpSampling2D


def residual_block(x, nb_filter, block_idx, bn=True, weight_decay=0):

    # 1st conv
    name = "block%s_conv2D%s" % (block_idx, "a")
    W_reg = l2(weight_decay)
    r = Conv2D(nb_filter, (3, 3), padding="same", kernel_regularizer=W_reg, name=name)(x)
    if bn:
        r = BatchNormalization(axis=1, name="block%s_bn%s" % (block_idx, "a"))(r)
    r = Activation("relu", name="block%s_relu%s" % (block_idx, "a"))(r)

    # 2nd conv
    name = "block%s_conv2D%s" % (block_idx, "b")
    W_reg = l2(weight_decay)
    r = Conv2D(nb_filter, (3, 3), padding="same", kernel_regularizer=W_reg, name=name)(r)
    if bn:
        r = BatchNormalization(axis=1, name="block%s_bn%s" % (block_idx, "b"))(r)
    r = Activation("relu", name="block%s_relu%s" % (block_idx, "b"))(r)

    # Merge residual and identity
    x = Add(name="block%s_merge" % block_idx)([x, r])

    return x


def convolutional_block(x, block_idx, nb_filter, nb_conv, strides):

    # 1st conv
    for i in range(nb_conv):
        name = "block%s_conv2D_%s" % (block_idx, i)
        if i < nb_conv - 1:
            x = Conv2D(nb_filter, (3, 3), name=name, padding="same")(x)
            x = BatchNormalization(axis=1)(x)
            x = Activation("relu")(x)
        else:
            x = Conv2D(nb_filter, (3, 3), name=name, strides=strides, padding="same")(x)
            x = BatchNormalization(axis=1)(x)
            x = Activation("relu")(x)

    return x


def colorful(nb_classes, img_dim, batch_size, model_name="colorful"):

    nb_resblocks = 3
    block_idx = 0
    h, w = img_dim[1:]

    # First conv block
    x_input = Input(shape=img_dim, name="input")
    x = Conv2D(64, (3, 3), name="block%s_conv2d_0" % block_idx, padding="same")(x_input)
    x = Activation("relu", name="block%s_relu" % block_idx)(x)
    block_idx += 1

    # Residual blocks
    for idx, f in enumerate([64] * nb_resblocks):
        x = residual_block(x, f, block_idx, weight_decay=0)
        block_idx += 1

    # Final conv
    x = Conv2D(nb_classes, (1, 1), name="final_conv2D", padding="same")(x)

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
    colorful = Model(input=[x_input], output=[x], name=model_name)

    return colorful


def load(nb_classes, img_dim, batch_size):

    model = colorful(nb_classes, img_dim, batch_size, model_name="colorful")

    return model
