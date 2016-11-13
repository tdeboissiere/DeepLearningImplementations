import os
import sys
import glob
import time
import h5py
import numpy as np
import models_colorful as models
from keras.utils import generic_utils
from keras.optimizers import Adam
import sklearn.neighbors as nn
import keras.backend as K
# Utils
sys.path.append("../utils")
import batch_utils
import general_utils


def categorical_crossentropy_color(y_true, y_pred):

    # Flatten
    n, h, w, q = y_true.shape
    y_true = K.reshape(y_true, (n * h * w, q))
    y_pred = K.reshape(y_pred, (n * h * w, q))

    weights = y_true[:, 313:]  # extract weight from y_true
    weights = K.concatenate([weights] * 313, axis=1)
    y_true = y_true[:, :-1]  # remove last column
    y_pred = y_pred[:, :-1]  # remove last column

    # multiply y_true by weights
    y_true = y_true * weights

    cross_ent = K.categorical_crossentropy(y_pred, y_true)
    cross_ent = K.mean(cross_ent, axis=-1)

    return cross_ent


def train(**kwargs):
    """
    Train model

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    data_file = kwargs["data_file"]
    nb_neighbors = kwargs["nb_neighbors"]
    model_name = kwargs["model_name"]
    training_mode = kwargs["training_mode"]
    epoch_size = n_batch_per_epoch * batch_size
    img_size = int(os.path.basename(data_file).split("_")[1])

    # Setup directories to save model, architecture etc
    general_utils.setup_logging(model_name)

    # Create a batch generator for the color data
    DataGen = batch_utils.DataGenerator(data_file,
                                        batch_size=batch_size,
                                        dset="training")
    c, h, w = DataGen.get_config()["data_shape"][1:]

    # Load the array of quantized ab value
    q_ab = np.load("../../data/processed/pts_in_hull.npy")
    nb_q = q_ab.shape[0]
    # Fit a NN to q_ab
    nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

    # Load the color prior factor that encourages rare colors
    prior_factor = np.load("../../data/processed/CelebA_%s_prior_factor.npy" % img_size)

    # Load and rescale data
    if training_mode == "in_memory":
        with h5py.File(data_file, "r") as hf:
            X_train = hf["training_lab_data"][:]

    # Remove possible previous figures to avoid confusion
    for f in glob.glob("../../figures/*.png"):
        os.remove(f)

    try:

        # Create optimizers
        opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Load colorizer model
        color_model = models.load(model_name, nb_q, (1, h, w), batch_size)
        color_model.compile(loss=categorical_crossentropy_color, optimizer=opt)

        color_model.summary()
        from keras.utils.visualize_util import plot
        plot(color_model, to_file='../../figures/colorful.png', show_shapes=True, show_layer_names=True)

        # Actual training loop
        for epoch in range(nb_epoch):

            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1
            start = time.time()

            # Choose Batch Generation mode
            if training_mode == "in_memory":
                BatchGen = DataGen.gen_batch_in_memory(X_train, nn_finder, nb_q, prior_factor)
            else:
                BatchGen = DataGen.gen_batch(nn_finder, nb_q, prior_factor)

            for batch in BatchGen:

                X_batch_black, X_batch_color, Y_batch = batch

                train_loss = color_model.train_on_batch(X_batch_black / 100., Y_batch)

                batch_counter += 1
                progbar.add(batch_size, values=[("loss", train_loss)])

                if batch_counter >= n_batch_per_epoch:
                    break

            print("")
            print('Epoch %s/%s, Time: %s' % (epoch + 1, nb_epoch, time.time() - start))

            # Plot some data with original, b and w and colorized versions side by side
            general_utils.plot_batch(color_model, q_ab, X_batch_black, X_batch_color,
                                     batch_size, h, w, nb_q, epoch)

            # Save weights every 5 epoch
            if epoch % 5 == 0:
                weights_path = os.path.join('../../models/%s/%s_weights_epoch%s.h5' %
                                            (model_name, model_name, epoch))
                color_model.save_weights(weights_path, overwrite=True)

    except KeyboardInterrupt:
        pass
