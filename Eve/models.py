from keras.models import Model
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def standard_conv_block(x, nb_filter, subsample=(1,1), pooling=False, bn=False, dropout_rate=None, weight_decay=0):
    x = Convolution2D(nb_filter, 3, 3,
                      subsample=subsample,
                      border_mode="same",
                      W_regularizer=l2(weight_decay))(x)
    if bn:
        x = BatchNormalization(mode=2, axis=1)(x)
    x = Activation("relu")(x)
    if pooling:
        x = MaxPooling2D()(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def FCN(img_dim, nb_classes, model_name="FCN"):

    x_input = Input(shape=img_dim, name="input")

    x = Flatten()(x_input)

    for i in range(20):
        x = Dense(50, activation="relu")(x)

    x = Dense(nb_classes, activation="softmax")(x)

    FCN = Model(input=[x_input], output=[x])
    FCN.name = model_name

    return FCN


def CNN(img_dim, nb_classes, model_name="CNN"):

    x_input = Input(shape=img_dim, name="input")

    x = standard_conv_block(x_input, 32)
    x = standard_conv_block(x, 32, pooling=True, dropout_rate=0.25)
    x = standard_conv_block(x, 64)
    x = standard_conv_block(x, 64, pooling=True, dropout_rate=0.25)

    # FC part
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(nb_classes, activation="softmax")(x)

    CNN = Model(input=[x_input], output=[x])
    CNN.name = model_name

    return CNN


def Big_CNN(img_dim, nb_classes, model_name="Big_CNN"):

    x_input = Input(shape=img_dim, name="input")

    x = standard_conv_block(x_input, 64)
    x = standard_conv_block(x, 64)
    x = standard_conv_block(x, 64, pooling=True, dropout_rate=0.5)

    x = standard_conv_block(x, 128)
    x = standard_conv_block(x, 128)
    x = standard_conv_block(x, 128, pooling=True, dropout_rate=0.5)

    x = standard_conv_block(x, 256)
    x = standard_conv_block(x, 256)
    x = standard_conv_block(x, 256, pooling=True, dropout_rate=0.5)

    # FC part
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation="softmax")(x)

    Big_CNN = Model(input=[x_input], output=[x])
    Big_CNN.name = model_name

    return Big_CNN


def load(model_name, img_dim, nb_classes):

    if model_name == "CNN":
        model = CNN(img_dim, nb_classes, model_name=model_name)
    if model_name == "Big_CNN":
        model = Big_CNN(img_dim, nb_classes, model_name=model_name)
    elif model_name == "FCN":
        model = FCN(img_dim, nb_classes, model_name=model_name)

    return model
