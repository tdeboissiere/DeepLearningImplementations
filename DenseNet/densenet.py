from keras.models import Model
from keras.layers.core import Flatten, Reshape, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization


def conv_factory(x, nb_filter, dropout_rate=None):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout"""

    x = BatchNormalization(mode=0, axis=1)(x)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same")(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, nb_filter, dropout_rate=None):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D"""

    x = BatchNormalization(mode=0, axis=1)(x)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filter, 1, 1, init="he_uniform", border_mode="same")(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    return x


def denseblock(x, nb_layers, nb_filter, growth_rate, dropout_rate=None):
    """Build a denseblock where the output of each conv_factory is fed to subsequent ones"""

    list_feat = [x]

    for i in range(nb_layers):
        x = conv_factory(x, growth_rate, dropout_rate)
        list_feat.append(x)
        x = merge(list_feat, mode='concat', concat_axis=1)
        nb_filter += growth_rate

    return x, nb_filter


def DenseNet(nb_classes, img_dim, depth, nb_dense_block, growth_rate, nb_filter, dropout_rate=None):
    """ Build the DenseNet model"""

    model_input = Input(shape=img_dim)

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", name="initial_conv2D")(model_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate)
        # add transition
        x = transition(x, nb_filter, dropout_rate=dropout_rate)

    # The last denseblock does not have a transition
    x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate)

    x = BatchNormalization(mode=0, axis=1)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8,8))(x)
    x = Flatten()(x)
    x = Dense(nb_classes, activation='softmax')(x)

    densenet = Model(input=[model_input], output=[x], name="DenseNet")

    return densenet
