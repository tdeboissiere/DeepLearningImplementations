

from __future__ import print_function
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import  LearningRateScheduler
import numpy as np
import densenet

batch_size = 64
nb_classes = 10
nb_epoch = 300

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

img_dim = (img_channels, img_rows, img_cols)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = 16
dropout_rate = 0.2
learning_rate = 0.1

model = densenet.DenseNet(nb_classes,
                          img_dim,
                          depth,
                          nb_dense_block,
                          growth_rate,
                          nb_filter,
                          dropout_rate=dropout_rate)

# Model output
model.summary()

# Build optimizer
opt = SGD(lr=learning_rate, decay=1e-4, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

from keras.utils.visualize_util import plot
plot(model, to_file='model.png', show_shapes=True)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalisation
X = np.vstack((X_train, X_test))
for i in range(img_channels):
	mean = np.mean(X[:, i, :, :])
	std = np.std(X[:, i, :, :])
	X_train[:, i, :, :] = (X_train[:, i, :, :] - mean) / std
	X_test[:, i, :, :] = (X_test[:, i, :, :] - mean) / std

def scheduleLR(epoch_index):
	
	if epoch_index == int(nb_epoch / 2):
		return learning_rate / 10

	elif epoch_index == int( 3 * nb_epoch / 4):
		return learning_rate / 100

print("Training")
model.fit(X_train, Y_train, 
	      nb_epoch=nb_epoch, 
          batch_size=64, 
          validation_data=(X_test, Y_test), 
          verbose=1,
          callbacks=[LearningRateScheduler(scheduleLR)])
