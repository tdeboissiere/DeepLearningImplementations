import time
import numpy as np
import multiprocessing
import os
import h5py


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

        # Dict that will store all transformations and their parameters
        self.d_transform = {}

        # Read the data file to get dataset shape information
        with h5py.File(self.hdf5_file, "r") as hf:
            self.X_shape = hf["%s_lab_data" % self.dset].shape
            assert len(self.X_shape) == 4,\
                ("\n\nImg data should be formatted as: \n"
                 "(n_samples, n_channels, Height, Width)")
            self.n_samples = hf["%s_lab_data" % self.dset].shape[0]
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

    def get_soft_encoding(self, X, nn_finder, nb_q):

        sigma_neighbor = 5

        # Get the distance to and the idx of the nearest neighbors
        dist_neighb, idx_neigh = nn_finder.kneighbors(X)

        # Smooth the weights with a gaussian kernel
        wts = np.exp(-dist_neighb**2 / (2 * sigma_neighbor**2))
        wts = wts / np.sum(wts, axis=1)[:, np.newaxis]

        # format the target
        Y = np.zeros((X.shape[0], nb_q))
        idx_pts = np.arange(X.shape[0])[:, np.newaxis]
        Y[idx_pts, idx_neigh] = wts

        return Y

    def gen_batch(self, nn_finder, nb_q, prior_factor):
        """ Use multiprocessing to generate batches in parallel. """
        try:
            queue = multiprocessing.Queue(maxsize=self.num_cached)

            # define producer (putting items into queue)
            def producer():

                try:
                    # Load the data from HDF5 file
                    with h5py.File(self.hdf5_file, "r") as hf:
                        num_chan, height, width = self.X_shape[-3:]
                        # Select start_idx at random for the batch
                        idx_start = np.random.randint(0, self.X_shape[0] - self.batch_size)
                        idx_end = idx_start + self.batch_size
                        # Get X and y
                        X_batch_color = hf["%s_lab_data" % self.dset][idx_start: idx_end, :, :, :]

                        X_batch_black = X_batch_color[:, :1, :, :]
                        X_batch_ab = X_batch_color[:, 1:, :, :]
                        npts, c, h, w = X_batch_ab.shape
                        X_a = np.ravel(X_batch_ab[:, 0, :, :])
                        X_b = np.ravel(X_batch_ab[:, 1, :, :])
                        X_batch_ab = np.vstack((X_a, X_b)).T

                        Y_batch = self.get_soft_encoding(X_batch_ab, nn_finder, nb_q)
                        # Add prior weight to Y_batch
                        idx_max = np.argmax(Y_batch, axis=1)
                        weights = prior_factor[idx_max].reshape(Y_batch.shape[0], 1)
                        Y_batch = np.concatenate((Y_batch, weights), axis=1)
                        # # Reshape Y_batch
                        Y_batch = Y_batch.reshape((npts, h, w, nb_q + 1))

                        # Put the data in a queue
                        queue.put((X_batch_black, X_batch_color, Y_batch))
                except:
                    print("Nothing here")

            processes = []

            def start_process():
                for i in range(len(processes), self.maxproc):
                    # Reset the seed ! (else the processes share the same seed)
                    np.random.seed()
                    thread = multiprocessing.Process(target=producer)
                    time.sleep(0.01)
                    thread.start()
                    processes.append(thread)

            # run as consumer (read items from queue, in current thread)
            while True:
                processes = [p for p in processes if p.is_alive()]

                if len(processes) < self.maxproc:
                    start_process()

                yield queue.get()

        except:
            for th in processes:
                th.terminate()
            queue.close()
            raise

    def gen_batch_in_memory(self, X, nn_finder, nb_q, prior_factor):
        """Generate batch, assuming X is loaded in memory in the main program"""

        while True:
            # Select idx at random for the batch
            idx = np.random.choice(X.shape[0], self.batch_size, replace=False)

            X_batch_color = X[idx]

            X_batch_black = X_batch_color[:, :1, :, :]
            X_batch_ab = X_batch_color[:, 1:, :, :]
            npts, c, h, w = X_batch_ab.shape
            X_a = np.ravel(X_batch_ab[:, 0, :, :])
            X_b = np.ravel(X_batch_ab[:, 1, :, :])
            X_batch_ab = np.vstack((X_a, X_b)).T

            Y_batch = self.get_soft_encoding(X_batch_ab, nn_finder, nb_q)
            # Add prior weight to Y_batch
            idx_max = np.argmax(Y_batch, axis=1)
            weights = prior_factor[idx_max].reshape(Y_batch.shape[0], 1)
            Y_batch = np.concatenate((Y_batch, weights), axis=1)
            # # Reshape Y_batch
            Y_batch = Y_batch.reshape((npts, h, w, nb_q + 1))

            yield X_batch_black, X_batch_color, Y_batch
