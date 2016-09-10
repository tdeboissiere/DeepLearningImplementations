from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Deconvolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers import Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization


def conv_factory(x, num_filter, dropout_rate=None):

    x = BatchNormalization(mode=2, axis=1)(x)
    x = Activation('relu')(x)
    x = Convolution2D(num_filter, 3, 3, border_mode="same")(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, num_filter, dropout_rate=None):

    x = BatchNormalization(mode=2, axis=1)(x)
    x = Activation('relu')(x)
    x = Convolution2D(num_filter, 1, 1, border_mode="same")(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    return x


def denseblock(x, num_conv, num_filter, dropout_rate=None):

    list_feat = []

    for i in range(num_conv):
        x = conv_factory(x, num_filter, dropout_rate)
        list_feat.append(x)
        x = merge(list_feat, mode='concat', concat_axis=1)

    x = transition(x, num_filter, dropout_rate=dropout_rate)

    return x


def DenseNet(nb_classes, img_dim, num_dense_block, num_conv, num_filter, dropout_rate=None):

    model_input = Input(shape=img_dim)

    # Initial convolution
    x = Convolution2D(16, 3, 3, border_mode="same")(model_input)

    # Add dense blocks
    for i in range(num_dense_block):
        x = denseblock(x, num_conv, num_filter, dropout_rate=dropout_rate)

    x = BatchNormalization(mode=2, axis=1)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8,8))
    x = Flatten()(x)
    x = Dense(nb_classes, activation='softmax')(x)

    densenet = Model(input=[model_input], output=[x], name="DenseNet")

    return densenet
