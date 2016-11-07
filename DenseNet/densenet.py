from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K


def conv_factory(x, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout

    :param x: Input keras network
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras network with b_norm, relu and convolution2d added
    :rtype: keras network
    """

    x = BatchNormalization(mode=0,
                           axis=1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filter, 3, 3,
                      init="he_uniform",
                      border_mode="same",
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D

    :param x: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool

    """

    x = BatchNormalization(mode=0,
                           axis=1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filter, 1, 1,
                      init="he_uniform",
                      border_mode="same",
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def denseblock(x, nb_layers, nb_filter, growth_rate,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones

    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """

    list_feat = [x]

    if K.image_dim_ordering() == "th":
        concat_axis = 1
    elif K.image_dim_ordering() == "tf":
        concat_axis = -1

    for i in range(nb_layers):
        x = conv_factory(x, growth_rate, dropout_rate, weight_decay)
        list_feat.append(x)
        x = merge(list_feat, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate

    return x, nb_filter


def denseblock_altern(x, nb_layers, nb_filter, growth_rate,
                      dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each conv_factory
       is fed to subsequent ones. (Alternative of a above)

    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    * The main difference between this implementation and the implementation
    above is that the one above
    """

    if K.image_dim_ordering() == "th":
        concat_axis = 1
    elif K.image_dim_ordering() == "tf":
        concat_axis = -1

    for i in range(nb_layers):
        merge_tensor = conv_factory(x, growth_rate, dropout_rate, weight_decay)
        x = merge([merge_tensor, x], mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate

    return x, nb_filter


def DenseNet(nb_classes, img_dim, depth, nb_dense_block, growth_rate,
             nb_filter, dropout_rate=None, weight_decay=1E-4):
    """ Build the DenseNet model

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """

    model_input = Input(shape=img_dim)

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Convolution2D(nb_filter, 3, 3,
                      init="he_uniform",
                      border_mode="same",
                      name="initial_conv2D",
                      bias=False,
                      W_regularizer=l2(weight_decay))(model_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        # add transition
        x = transition(x, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)

    x = BatchNormalization(mode=0,
                           axis=1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D(dim_ordering="th")(x)
    x = Dense(nb_classes,
              activation='softmax',
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x)

    densenet = Model(input=[model_input], output=[x], name="DenseNet")

    return densenet
