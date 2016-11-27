from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import numpy as np
import KerasDeconv
import cPickle as pickle
from utils import get_deconv_images
from utils import plot_deconv
from utils import plot_max_activation
from utils import find_top9_mean_act
import glob
import cv2
import os


def VGG_16(weights_path=None):
    """
    VGG Model Keras specification

    args: weights_path (str) trained weights file path

    returns model (Keras model)
    """

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        print("Loading weights...")
        model.load_weights(weights_path)

    return model


def load_model(weights_path):
    """
    Load and compile VGG model

    args: weights_path (str) trained weights file path

    returns model (Keras model)
    """

    model = VGG_16(weights_path)
    model.compile(optimizer="sgd", loss='categorical_crossentropy')
    return model

if __name__ == "__main__":

    ######################
    # Misc
    ######################
    model = None  # Initialise VGG model to None
    Dec = None  # Initialise DeconvNet model to None
    if not os.path.exists("./Figures/"):
        os.makedirs("./Figures/")

    ############
    # Load data
    ############
    list_img = glob.glob("./Data/Img/*.jpg*")
    assert len(list_img) > 0, "Put some images in the ./Data/Img folder"
    if len(list_img) < 32:
        list_img = (int(32 / len(list_img)) + 2) * list_img
        list_img = list_img[:32]
    data = []
    for im_name in list_img:
        im = cv2.resize(cv2.imread(im_name), (224, 224)).astype(np.float32)
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        im = im.transpose((2, 0, 1))
        data.append(im)
    data = np.array(data)

    ###############################################
    # Action 1) Get max activation for a secp ~/deconv_specificlection of feat maps
    ###############################################
    get_max_act = True
    if get_max_act:
        if not model:
            model = load_model('./Data/vgg16_weights.h5')
        if not Dec:
            Dec = KerasDeconv.DeconvNet(model)
        d_act_path = './Data/dict_top9_mean_act.pickle'
        d_act = {"convolution2d_13": {},
                 "convolution2d_10": {}
                 }
        for feat_map in range(10):
            d_act["convolution2d_13"][feat_map] = find_top9_mean_act(
                data, Dec, "convolution2d_13", feat_map, batch_size=32)
            d_act["convolution2d_10"][feat_map] = find_top9_mean_act(
                data, Dec, "convolution2d_10", feat_map, batch_size=32)
            with open(d_act_path, 'w') as f:
                pickle.dump(d_act, f)

    ###############################################
    # Action 2) Get deconv images of images that maximally activate
    # the feat maps selected in the step above
    ###############################################
    deconv_img = True
    if deconv_img:
        d_act_path = './Data/dict_top9_mean_act.pickle'
        d_deconv_path = './Data/dict_top9_deconv.pickle'
        if not model:
            model = load_model('./Data/vgg16_weights.h5')
        if not Dec:
            Dec = KerasDeconv.DeconvNet(model)
        get_deconv_images(d_act_path, d_deconv_path, data, Dec)

    ###############################################
    # Action 3) Get deconv images of images that maximally activate
    # the feat maps selected in the step above
    ###############################################
    plot_deconv_img = True
    if plot_deconv_img:
        d_act_path = './Data/dict_top9_mean_act.pickle'
        d_deconv_path = './Data/dict_top9_deconv.npz'
        target_layer = "convolution2d_10"
        plot_max_activation(d_act_path, d_deconv_path,
                            data, target_layer, save=True)

    ###############################################
    # Action 4) Get deconv images of some images for some
    # feat map
    ###############################################
    deconv_specific = False
    img_choice = False  # for debugging purposes
    if deconv_specific:
        if not model:
            model = load_model('./Data/vgg16_weights.h5')
        if not Dec:
            Dec = KerasDeconv.DeconvNet(model)
        target_layer = "convolution2d_13"
        feat_map = 12
        num_img = 25
        if img_choice:
            img_index = []
            assert(len(img_index) == num_img)
        else:
            img_index = np.random.choice(data.shape[0], num_img, replace=False)
        plot_deconv(img_index, data, Dec, target_layer, feat_map)
