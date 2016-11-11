import os
import sys
import numpy as np
import models_colorful
import sklearn.neighbors as nn
# Utils
sys.path.append("../utils")
import batch_utils
import general_utils


def eval(**kwargs):

    data_file = kwargs["data_file"]
    model_name = kwargs["model_name"]
    epoch = kwargs["epoch"]
    T = kwargs["T"]
    batch_size = kwargs["batch_size"]
    nb_neighbors = kwargs["nb_neighbors"]

    img_size = int(os.path.basename(data_file).split("_")[1])

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

    # Load colorization model
    color_model = models_colorful.load(model_name, nb_q, (1, h, w), batch_size)
    color_model.load_weights("../../models/%s/%s_weights_epoch%s.h5" %
                             (model_name, model_name, epoch))

    for batch in DataGen.gen_batch(nn_finder, nb_q, prior_factor):

        X_batch_black, X_batch_color, Y_batch = batch

        general_utils.plot_batch_eval(color_model, q_ab, X_batch_black, X_batch_color,
                                      batch_size, h, w, nb_q, T)
