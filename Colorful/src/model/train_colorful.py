import os
import sys
import glob
import time
import h5py
import numpy as np
from skimage import color
import models_colorful as models
from keras.utils import np_utils
from keras.utils import generic_utils
from keras.models import Model
from keras.optimizers import RMSprop, Adam, SGD
from sklearn.metrics import log_loss
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import sklearn.neighbors as nn
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm
# Utils
sys.path.append("../utils")
import batch_utils
import general_utils


def train(**kwargs):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    prob = kwargs["prob"]
    training_data_file = kwargs["training_data_file"]
    experiment = kwargs["experiment"]
    epoch_size = n_batch_per_epoch * batch_size

    # Setup environment (logging directory etc)
    general_utils.setup_logging(experiment)

    # Create a batch generator for the color data
    DataAug = batch_utils.AugDataGenerator(training_data_file,
                                           batch_size=batch_size,
                                           prob=prob,
                                           dset="training_color")
    DataAug.add_transform("h_flip")

    # Load the array of quantized ab value
    q_ab = np.load("../../data/processed/pts_in_hull.npy")
    nb_q = q_ab.shape[0]
    nb_neighbors = 10
    # Fit a NN to q_ab
    nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

    # Load the color prior factor that encourages rare colors
    prior_factor = np.load("../../data/processed/training_64_prior_factor.npy")

    # Load and rescale data
    print("Loading data")
    with h5py.File(training_data_file, "r") as hf:
        X_train = hf["training_lab_data"][:100]
        c, h, w = X_train.shape[1:]
    print("Data loaded")

    for f in glob.glob("*.h5"):
        os.remove(f)

    for f in glob.glob("../../reports/figures/*.png"):
        os.remove(f)

    try:

        # Create optimizers
        # opt = SGD(lr=5E-4, momentum=0.9, nesterov=True)
        opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Load colorizer model
        color_model = models.load("simple_colorful", nb_q, (1, h, w), batch_size)
        color_model.compile(loss='categorical_crossentropy_color', optimizer=opt)

        color_model.summary()
        from keras.utils.visualize_util import plot
        plot(color_model, to_file='colorful.png')

        # Start training
        print("Start training")
        for e in range(nb_epoch):
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1
            start = time.time()

            for batch in DataAug.gen_batch_colorful(X_train, nn_finder, nb_q, prior_factor):

                X_batch_black, X_batch_color, Y_batch = batch
                # X = color_model.predict(X_batch_black)

                # print color_model.evaluate(X_batch_black, Y_batch)
                # X = color_model.predict(X_batch_black)
                # print X[0, 0, 0, :]

                train_loss = color_model.train_on_batch(X_batch_black / 100., Y_batch)

                batch_counter += 1
                progbar.add(batch_size, values=[("loss", train_loss)])

                if batch_counter >= n_batch_per_epoch:
                    break
            print("")
            print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

            # Format X_colorized
            X_colorized = color_model.predict(X_batch_black / 100.)[:, :, :, :-1]
            X_colorized = X_colorized.reshape((batch_size * h * w, nb_q))
            X_colorized = q_ab[np.argmax(X_colorized, 1)]
            X_a = X_colorized[:, 0].reshape((batch_size, 1, h, w))
            X_b = X_colorized[:, 1].reshape((batch_size, 1, h, w))
            X_colorized = np.concatenate((X_batch_black, X_a, X_b), axis=1).transpose(0, 2, 3, 1)
            X_colorized = [np.expand_dims(color.lab2rgb(im), 0) for im in X_colorized]
            X_colorized = np.concatenate(X_colorized, 0).transpose(0, 3, 1, 2)

            X_batch_color = [np.expand_dims(color.lab2rgb(im.transpose(1, 2, 0)), 0) for im in X_batch_color]
            X_batch_color = np.concatenate(X_batch_color, 0).transpose(0, 3, 1, 2)

            print X_batch_color.shape, X_colorized.shape, X_batch_black.shape

            for i, img in enumerate(X_colorized[:min(32, batch_size)]):
                arr = np.concatenate([X_batch_color[i], np.repeat(X_batch_black[i] / 100., 3, axis=0), img], axis=2)
                np.save("../../reports/gen_image_%s.npy" % i, arr)

            plt.figure(figsize=(20,20))
            list_img = glob.glob("../../reports/*.npy")
            list_img = [np.load(im) for im in list_img]
            list_img = [np.concatenate(list_img[4 * i: 4 * (i + 1)], axis=2) for i in range(len(list_img) / 4)]
            arr = np.concatenate(list_img, axis=1)
            plt.imshow(arr.transpose(1,2,0))
            ax = plt.gca()
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            plt.tight_layout()
            plt.savefig("../../reports/figures/fig_epoch%s.png" % e)
            plt.clf()
            plt.close()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':

    ####################################
    # Test the labeling/ lab conversions
    ####################################

    # Create a batch generator for the color data
    DataAug = batch_utils.AugDataGenerator("../../data/processed/training_64_data.h5",
                                           batch_size=32,
                                           prob=0.5,
                                           dset="training_color")
    DataAug.add_transform("h_flip")

    # Load the array of quantized ab value
    q_ab = np.load("../../data/processed/pts_in_hull.npy")
    nb_q = q_ab.shape[0]
    nb_neighbors = 1
    # Fir a NN to q_ab
    nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

    # Load and rescale data
    with h5py.File("../../data/processed/training_64_data.h5", "r") as hf:
        X_train = hf["training_lab_data"][:100]
        c, h, w = X_train.shape[1:]

    for batch in DataAug.gen_batch_colorful(X_train, nn_finder, nb_q):

        X_batch_black, X_batch_color, Y_batch = batch

        # Checking Y
        idxs = np.random.randint(0, X_batch_color.shape[0], 100000)
        list_i = np.random.randint(0, X_batch_color.shape[2], 100000)
        list_j = np.random.randint(0, X_batch_color.shape[3], 100000)
        for idx, i, j in zip(idxs, list_i, list_j):
            a = X_batch_color[idx, 1, i, j]  # a
            b = X_batch_color[idx, 2, i, j]  # b
            dist, inds = nn_finder.kneighbors(np.array([a,b]).reshape((1,2)))
            assert inds[0][0] == np.argmax(Y_batch[idx, :, i, j])

        # Plotting
        fig = plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(4,4)
        for i in range(16):
            ax = plt.subplot(gs[i])
            img = color.lab2rgb(X_batch_color[i, :, :, :].transpose(1,2,0))
            ax.imshow(img)
        gs.tight_layout(fig)
        plt.show()
        print Y_batch.shape
