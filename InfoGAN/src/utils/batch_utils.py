import time
import numpy as np
import multiprocessing
import os
import h5py
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm


class DataGenerator(object):
    """
    Generate minibatches with real-time data parallel augmentation on CPU

    args :
        hdf5_file   (str)      path to data in HDF5 format
        batch_size  (int)      Minibatch size
        dset        (str)      train/test/valid, the name of the dset to iterate over
        maxproc     (int)      max number of processes to spawn in parallel
        num_cached  (int)      max number of batches to keep in queue

    yields :
         X, y (minibatch data and labels as np arrays)
    """

    def __init__(self,
                 hdf5_file,
                 batch_size=32,
                 nb_classes=12,
                 dset="training",
                 maxproc=8,
                 num_cached=10):

        # Check file exists
        assert os.path.isfile(hdf5_file), hdf5_file + " doesn't exist"

        # Initialize class internal variables
        self.dset = dset
        self.maxproc = maxproc
        self.hdf5_file = hdf5_file
        self.batch_size = batch_size
        self.num_cached = num_cached
        self.nb_classes = nb_classes

        # Dict that will store all transformations and their parameters
        self.d_transform = {}

        # Read the data file to get dataset shape information
        with h5py.File(self.hdf5_file, "r") as hf:
            self.X_shape = hf["data"].shape
            assert len(self.X_shape) == 4,\
                ("\n\nImg data should be formatted as: \n"
                 "(n_samples, n_channels, Height, Width)")
            self.n_samples = hf["data"].shape[0]
            # Verify n_channels is at index 1
            assert self.X_shape[-3] < min(self.X_shape[-2:]),\
                ("\n\nImg data should be formatted as: \n"
                 "(n_samples, n_channels, Height, Width)")

        # Save the class internal variables to a config dict
        self.d_config = {}
        self.d_config["hdf5_file"] = hdf5_file
        self.d_config["batch_size"] = batch_size
        self.d_config["dset"] = dset
        self.d_config["num_cached"] = num_cached
        self.d_config["maxproc"] = maxproc
        self.d_config["data_shape"] = self.X_shape

    def get_config(self):

        return self.d_config

    def gen_batch_inmemory_GAN(self, X_real, batch_size=None):
        """Generate batch, assuming X is loaded in memory in the main program"""

        while True:

            bs = self.batch_size
            if batch_size is not None:
                bs = batch_size

            # Select idx at random for the batch
            idx = np.random.choice(X_real.shape[0], bs, replace=False)
            X_batch_real = X_real[idx]

            yield X_batch_real
