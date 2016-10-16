from __future__ import print_function
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
import keras.backend as K
import time
import os
import numpy as np
import densenet
import json

batch_size = 64
nb_classes = 10
nb_epoch = 300

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


img_dim = X_train.shape[1:]
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = 16
dropout_rate = 0.2
learning_rate = 0.1
weight_decay = 1E-4

model = densenet.DenseNet(nb_classes,
                          img_dim,
                          depth,
                          nb_dense_block,
                          growth_rate,
                          nb_filter,
                          dropout_rate=dropout_rate,
                          weight_decay=weight_decay)

# Model output
model.summary()

# Build optimizer
opt = SGD(lr=learning_rate, decay=0, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

# uncomment to plot model architecture
# from keras.utils.visualize_util import plot
# plot(model, to_file='./figures/densenet_archi.png', show_shapes=True)
# raw_input("OK")

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalisation
X = np.vstack((X_train, X_test))
# 2 cases depending on the image ordering
if K.image_dim_ordering() == "th":
    n_channels = img_dim[0]
    for i in range(n_channels):
        mean = np.mean(X[:, i, :, :])
        std = np.std(X[:, i, :, :])
        X_train[:, i, :, :] = (X_train[:, i, :, :] - mean) / std
        X_test[:, i, :, :] = (X_test[:, i, :, :] - mean) / std

elif K.image_dim_ordering() == "tf":
    n_channels = img_dim[-1]
    for i in range(n_channels):
        mean = np.mean(X[:, :, :, i])
        std = np.std(X[:, :, :, i])
        X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
        X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std

print("Training")

list_train_loss = []
list_test_loss = []
list_learning_rate = []

for e in range(nb_epoch):

    if e == int(0.5 * nb_epoch):
        K.set_value(model.optimizer.lr, np.float32(learning_rate / 10.))

    if e == int(0.75 * nb_epoch):
        K.set_value(model.optimizer.lr, np.float32(learning_rate / 100.))

    split_size = batch_size
    num_splits = X_train.shape[0] / split_size
    arr_splits = np.array_split(np.arange(X_train.shape[0]), num_splits)

    progbar = generic_utils.Progbar(len(arr_splits))
    l_train_loss = []
    start = time.time()

    for batch_idx in arr_splits:

        X_batch, Y_batch = X_train[batch_idx], Y_train[batch_idx]
        train_logloss, train_acc = model.train_on_batch(X_batch, Y_batch)

        l_train_loss.append([train_logloss, train_acc])

    test_logloss, test_acc = model.evaluate(X_test, Y_test, verbose=0, batch_size=64)
    list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
    list_test_loss.append([test_logloss, test_acc])
    list_learning_rate.append(float(K.get_value(model.optimizer.lr)))  # to convert numpy array to json serializable
    print("")
    print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

    d_log = {}
    d_log["batch_size"] = batch_size
    d_log["nb_epoch"] = nb_epoch
    d_log["optimizer"] = opt.get_config()
    d_log["train_loss"] = list_train_loss
    d_log["test_loss"] = list_test_loss
    d_log["learning_rate"] = list_learning_rate

    json_file = os.path.join('./log/experiment_log_cifar10.json')
    with open(json_file, 'w') as fp:
        json.dump(d_log, fp, indent=4, sort_keys=True)
