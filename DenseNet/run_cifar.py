

from __future__ import print_function
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.utils import np_utils
import densenet

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

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
depth = 19
nb_dense_block = 3
growth_rate = 12
nb_filter = 16

model = densenet.DenseNet(nb_classes,
                          img_dim,
                          depth,
                          nb_dense_block,
                          growth_rate,
                          nb_filter,
                          dropout_rate=None)

# Model output
model.summary()

# Build optimizer
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='categorical_crossentropy', optimizer=opt)

from keras.utils.visualize_util import plot
plot(model, to_file='model.png', show_shapes=True)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.

print("Training")
model.fit(X_train, Y_train, nb_epoch=nb_epoch, validation_data=(X_test, Y_test), verbose=1)
