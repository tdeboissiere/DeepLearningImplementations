import time
import numpy as np
import multiprocessing
import h5py
import os
try:
    import cv2
except:
    pass


class AugDataGenerator(object):
    """
    Generate minibatches with real-time data parallel augmentation on CPU

    args :

        hdf5_file   (str)      path to data in HDF5 format

        dset        (str)      train/test/valid, the name of the dset to iterate over

        batch_size  (int)      Minibatch size

        maxproc     (int)      max number of processes to spawn in parallel

        num_cached  (int)      max number of batches to keep in queue

        random_augm (int/None) if int, randomly select random_augm distinct functions
                               to augment the data at each batch

        - "prob" indicates the probability with which a transformation is carried out
            e.g. "prob" = 0.5 means that half the samples will be transformed

        - Fixed indicates that the transformation parameter does not change
            e.g. a rotation will always be carried out with a certain angle

        - Random indicates that the transformation parameter will be uniformly
          sampled in a fixed interval.
            e.g. a rotation angle will be sampled between - angle and + angle

        - angle is an angle in degrees, can take negative values
        - tr_x/tr_y corresponds to a unit pixel translation
        - kernel_size is the size of the filter matrix used to distort the image

    yields :
         X, y (minibatch data and labels as np arrays)

    """

    def __init__(self,
                 hdf5_file,
                 batch_size=32,
                 prob=0.5,
                 dset="train",
                 maxproc=8,
                 num_cached=10,
                 random_augm=None,
                 do3D=None,
                 hdf5_file_semi=None):

        # Check file exists
        assert os.path.isfile(hdf5_file), hdf5_file + " doesn't exist"

        # Initialize class internal variables
        self.hdf5_file = hdf5_file
        self.hdf5_file_semi = hdf5_file_semi
        self.batch_size = batch_size
        self.prob = prob
        self.dset = dset
        self.num_cached = num_cached
        self.maxproc = maxproc
        self.random_augm = random_augm
        self.do3D = do3D

        # Dict that will store all transformations and their parameters
        self.d_transform = {}

        # Read the data file to get dataset shape information
        with h5py.File(self.hdf5_file, "r") as hf:
            self.X_shape = hf["%s_data" % self.dset].shape
            assert len(self.X_shape) in [4, 5],\
                ("\n\nImg data should be formatted as: \n"
                 "(n_samples, n_channels, Height, Width) or \n"
                 "(n_samples, n_frames, n_channels, Height, Width)")
            self.n_samples = hf["%s_data" % self.dset].shape[0]
            # Verify n_channels is at index 1
            assert self.X_shape[-3] < min(self.X_shape[-2:]),\
                ("\n\nImg data should be formatted as: \n"
                 "(n_samples, n_channels, Height, Width) or \n"
                 "(n_samples, n_frames, n_channels, Height, Width)")

        # Save the class internal variables to a config dict
        self.d_config = {}
        self.d_config["hdf5_file"] = hdf5_file
        self.d_config["batch_size"] = batch_size
        self.d_config["prob"] = prob
        self.d_config["dset"] = dset
        self.d_config["num_cached"] = num_cached
        self.d_config["maxproc"] = maxproc
        self.d_config["random_augm"] = random_augm
        self.d_config["data_shape"] = self.X_shape
        self.d_config["transforms"] = self.d_transform
        self.d_config["do3D"] = self.do3D

    def _transform(self, X, tf_name, idx):
        """ Perform a specified transformation to augment the data"""

        # Get the type of the transform
        tf_type = self.d_transform[tf_name]["tf_type"]

        if tf_type == "h_flip":
            X[idx] = X[idx, :, :, ::-1]
        elif tf_type == "v_flip":
            X[idx] = X[idx, :, ::-1, :]
        else:
            num_chan, height, width = self.X_shape[-3:]
            # Need to transpose the original data to use OpenCV functions
            Xtf = X[idx].transpose(1, 2, 0)
            # Fixed angle rotations
            if tf_type == "fixed_rot":
                angle = self.d_transform[tf_name]["angle"]
                rotM = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
                Xtf = cv2.warpAffine(Xtf, rotM, (width, height))
                Xtf = Xtf.reshape((height, width, num_chan))
                X[idx] = Xtf.transpose(2, 0, 1)
            # Random angle rotations
            elif tf_type == "random_rot":
                max_angle = self.d_transform[tf_name]["angle"]
                angle = np.random.uniform(-max_angle, max_angle)
                rotM = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
                Xtf = cv2.warpAffine(Xtf, rotM, (width, height))
                Xtf = Xtf.reshape((height, width, num_chan))
                X[idx] = Xtf.transpose(2, 0, 1)
            # Fixed translation
            elif tf_type == "fixed_tr":
                tr_x = self.d_transform[tf_name]["tr_x"]
                tr_y = self.d_transform[tf_name]["tr_y"]
                trM = np.float32([[1, 0, tr_x],[0, 1, tr_y]])
                Xtf = cv2.warpAffine(Xtf, trM, (width, height))
                Xtf = Xtf.reshape((height, width, num_chan))
                X[idx] = Xtf.transpose(2, 0, 1)
            # Random translations
            elif tf_type == "random_tr":
                max_tr_x = self.d_transform[tf_name]["tr_x"]
                max_tr_y = self.d_transform[tf_name]["tr_y"]
                tr_tr_x = np.random.uniform(-max_tr_x, max_tr_x)
                tr_tr_y = np.random.uniform(-max_tr_y, max_tr_y)
                trM = np.float32([[1, 0, tr_tr_x],[0, 1, tr_tr_y]])
                Xtf = cv2.warpAffine(Xtf, trM, (width, height))
                Xtf = Xtf.reshape((height, width, num_chan))
                X[idx] = Xtf.transpose(2, 0, 1)
            # Fixed crop and resize
            elif tf_type == "fixed_crop":
                pos_x = self.d_transform[tf_name]["pos_x"]
                pos_y = self.d_transform[tf_name]["pos_y"]
                crop_size_x = self.d_transform[tf_name]["crop_size_x"]
                crop_size_y = self.d_transform[tf_name]["crop_size_y"]
                # OpenCV conventions:
                # x axis == width, y axis == height
                # but opencv reads images as (height, width, n_channels)
                # That's why the first slice below is a y-axis slice
                Xtf = Xtf[pos_y:pos_y + crop_size_y,
                          pos_x:pos_x + crop_size_x, :]
                # Resize
                Xtf = cv2.resize(Xtf,(width, height), interpolation=cv2.INTER_CUBIC)
                Xtf = Xtf.reshape((height, width, num_chan))
                X[idx] = Xtf.transpose(2, 0, 1)
            # Random crop and resize
            elif tf_type == "random_crop":
                min_crop_size = self.d_transform[tf_name]["min_crop_size"]
                max_crop_size = self.d_transform[tf_name]["max_crop_size"]
                crop_size_x = np.random.randint(min_crop_size, max_crop_size)
                crop_size_y = np.random.randint(min_crop_size, max_crop_size)
                pos_x = np.random.randint(0, 1 + width - crop_size_x)  # +1 because npy slices
                pos_y = np.random.randint(0, 1 + height - crop_size_y)  # are non inclusive
                # OpenCV conventions:
                # x axis == width, y axis == height
                # but opencv reads images as (height, width, n_channels)
                # That's why the first slice below is a y-axis slice
                Xtf = Xtf[pos_y:pos_y + crop_size_y,
                          pos_x:pos_x + crop_size_x, :]
                # Resize
                Xtf = cv2.resize(Xtf,(width, height), interpolation=cv2.INTER_CUBIC)
                Xtf = Xtf.reshape((height, width, num_chan))
                X[idx] = Xtf.transpose(2, 0, 1)
            # Fixed blur
            elif tf_type == "fixed_blur":
                kernel_size = self.d_transform[tf_name]["kernel_size"]
                blur = (kernel_size, kernel_size)
                Xtf = cv2.blur(Xtf, blur)
                Xtf = Xtf.reshape((height, width, num_chan))
                X[idx] = Xtf.transpose(2, 0, 1)
            # Random blur
            elif tf_type == "random_blur":
                max_kernel_size = self.d_transform[tf_name]["kernel_size"]
                kernel_size = np.random.randint(1, max_kernel_size)
                blur = (kernel_size, kernel_size)
                Xtf = cv2.blur(Xtf, blur)
                Xtf = Xtf.reshape((height, width, num_chan))
                X[idx] = Xtf.transpose(2, 0, 1)
            # Fixed dilation
            elif tf_type == "fixed_dilate":
                kernel_size = self.d_transform[tf_name]["kernel_size"]
                kernel = np.ones((kernel_size,
                                  kernel_size), np.uint8)
                Xtf = cv2.dilate(Xtf, kernel, iterations=1)
                Xtf = Xtf.reshape((height, width, num_chan))
                X[idx] = Xtf.transpose(2, 0, 1)
            # Random dilation
            elif tf_type == "random_dilate":
                max_kernel_size = self.d_transform[tf_name]["kernel_size"]
                kernel_size = np.random.randint(1, max_kernel_size)
                kernel = np.ones((kernel_size,
                                  kernel_size), np.uint8)
                Xtf = cv2.dilate(Xtf, kernel, iterations=1)
                Xtf = Xtf.reshape((height, width, num_chan))
                X[idx] = Xtf.transpose(2, 0, 1)
            # Fixed erosion
            elif tf_type == "fixed_erode":
                kernel_size = self.d_transform[tf_name]["kernel_size"]
                kernel = np.ones((kernel_size,
                                  kernel_size), np.uint8)
                Xtf = cv2.erode(Xtf, kernel, iterations=1)
                Xtf = Xtf.reshape((height, width, num_chan))
                X[idx] = Xtf.transpose(2, 0, 1)
            # Random erosion
            elif tf_type == "random_erode":
                max_kernel_size = self.d_transform[tf_name]["kernel_size"]
                kernel_size = np.random.randint(1, max_kernel_size)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                Xtf = cv2.erode(Xtf, kernel, iterations=1)
                Xtf = Xtf.reshape((height, width, num_chan))
                X[idx] = Xtf.transpose(2, 0, 1)

            elif tf_type == "hist_equal":
                Xtf[:, :, 0] = cv2.equalizeHist(Xtf[:, :, 0])
                Xtf[:, :, 1] = cv2.equalizeHist(Xtf[:, :, 1])
                Xtf[:, :, 2] = cv2.equalizeHist(Xtf[:, :, 2])
                X[idx] = Xtf.transpose(2, 0, 1)

            elif tf_type == "random_occlusion":
                occ_size_x = self.d_transform[tf_name]["occ_size_x"]
                occ_size_y = self.d_transform[tf_name]["occ_size_y"]
                # Random pos
                x_start = np.random.randint(0, Xtf.shape[1] - occ_size_x)
                y_start = np.random.randint(0, Xtf.shape[0] - occ_size_y)
                Xtf[y_start: y_start + occ_size_y, x_start: x_start + occ_size_x, :] = 0
                X[idx] = Xtf.transpose(2, 0, 1)
            elif tf_type == "random_mask":
                arr_mask = self.d_transform[tf_name]["arr_mask"]
                mask_index = np.random.randint(0, arr_mask.shape[0])
                cmap = arr_mask[mask_index].transpose(1,2,0)
                Xtf[cmap < np.percentile(cmap, 50)] = 0
                # cmap = cmap < np.median(cmap)
                # for k in range(Xtf.shape[0]):
                #     for l in range(Xtf.shape[1]):
                #         if cmap[k, l]:
                #             Xtf[k, l, :] = 0
                X[idx] = Xtf.transpose(2, 0, 1)

        return X

    def _augment(self, X):
        """
        Iterate over all specified transformations and augment the data
        """

        # If transformations are to be selected randomly,
        # select a random chunk of self.transformations
        self.transformations = np.array(list(self.d_transform.keys()))  # add list() for python3
        if self.random_augm:
            max_num_augm = len(self.transformations)
            assert self.random_augm <= max_num_augm,\
                "random_augm exceeds the max number of transformations"
            augm_indices = np.random.choice(max_num_augm,
                                            self.random_augm,
                                            replace=False)
            self.transformations = self.transformations[augm_indices]

        if len(self.transformations) == 0:
            return X

        # Select which indices will be transformed and which tf
        # to apply
        tf_size = int(self.prob * self.batch_size)
        idx_tf = np.random.choice(self.batch_size, tf_size, replace=False)
        list_tf = self.transformations[np.random.choice(len(self.transformations),
                                                        tf_size,
                                                        replace=True)]

        for idx, tf in zip(idx_tf, list_tf):
            X = self._transform(X, tf, idx)
        return X

    def get_config(self):

        for key in self.d_config["transforms"].keys():
            if "random_mask" in key:
                self.d_config["transforms"].pop(key, None)
        return self.d_config

    def add_transform(self, tf_type, **kwargs):
        """ Add a new transformation for data augmentation

        The valid tf_type and corresponding keyword arguments are as follow:

        tf_type = "h_flip",        kwargs= {},
        tf_type = "v_flip",        kwargs= {},
        tf_type = "fixed_rot",     kwargs= {"angle": float},
        tf_type = "random_rot",    kwargs= {"angle": float},
        tf_type = "fixed_tr",      kwargs= {"tr_x": float, "tr_y": float},
        tf_type = "random_tr",     kwargs= {"tr_x": float, "tr_y": float},
        tf_type = "fixed_blur",    kwargs= {"kernel_size": float},
        tf_type = "random_blur",   kwargs= {"kernel_size": float},
        tf_type = "fixed_erode",   kwargs= {"kernel_size": float},
        tf_type = "random_erode",  kwargs= {"kernel_size": float},
        tf_type = "fixed_dilate",  kwargs= {"kernel_size": float},
        tf_type = "random_dilate", kwargs= {"kernel_size": float}
        tf_type = "hist_equal",    kwargs= {},
        tf_type = "random_occlusion",    kwargs= {"occ_size_x": int, "occ_size_y": int},
        tf_type = "random_mask",    kwargs= {"arr_mask": np array}
        """

        # Update self.d_transform
        list_keys = [int(k.split("_")[-1]) for k in self.d_transform.keys() if tf_type in k]

        kwargs["tf_type"] = tf_type

        if len(list_keys) == 0:
            tf_name = tf_type + "_0"
            self.d_transform[tf_type + "_0"] = kwargs
        else:
            index = str(list_keys[-1] + 1)
            tf_name = tf_type + "_%s" % index
            self.d_transform[tf_type + "_%s" % index] = kwargs

        self.d_config["transforms"][tf_name] = kwargs

        # Check parameters are valid
        # For OpenCV transformation involving convolution by a kernel
        # Check the kernel_size type and size
        if "kernel_size" in self.d_transform[tf_name]:
            kernel_size = self.d_transform[tf_name]["kernel_size"]
            check1 = isinstance(kernel_size, int)
            check2 = kernel_size > 0
            if not check1 and check2:
                raise AssertionError("kernel_size must be an int > 0")
        # For OpenCV transformations involving a fixed crop
        # Check the crop size and position fits the image
        if "fixed_crop" in self.d_transform[tf_name]:
            height, width = self.X_shape[-2:]
            pos_x = self.d_transform[tf_name]["pos_x"]
            pos_y = self.d_transform[tf_name]["pos_y"]
            crop_size_x = self.d_transform[tf_name]["crop_size_x"]
            crop_size_y = self.d_transform[tf_name]["crop_size_y"]
            check1 = pos_x <= width - crop_size_x
            check1 = pos_y <= height - crop_size_y
            if not check1 and check2:
                raise AssertionError("Chosen crop params don't fit image size")

    def gen_batch(self, fold=None):
        """ Use multiprocessing to generate batches in parallel. """
        try:
            queue = multiprocessing.Queue(maxsize=self.num_cached)

            # define producer (putting items into queue)
            def producer():

                try:
                    # Load the data from HDF5 file
                    with h5py.File(self.hdf5_file, "r") as hf:
                        num_chan, height, width = self.X_shape[-3:]
                        if fold is not None:
                            # Access train index for the given fold
                            idx_train = hf["train_fold%s" % fold][:]
                            # Select idx at random for the batch
                            idx_train_batch = np.random.choice(
                                idx_train, self.batch_size, replace=False)
                            # Sort for H5py
                            idx_train_batch = np.sort(idx_train_batch)
                            # Get X and y
                            X = hf["%s_data" % self.dset][idx_train_batch, :, :, :]
                            y = hf["%s_label" % self.dset][:][idx_train_batch].astype(int)
                        else:
                            # Select idx at random for the batch
                            idx_start = np.random.randint(0, self.X_shape[0] - self.batch_size)
                            idx_end = idx_start + self.batch_size
                            # Get X and y
                            X = hf["%s_data" % self.dset][idx_start: idx_end, :, :, :]
                            y = hf["%s_label" % self.dset][:][idx_start: idx_end]

                        # Augment the data (i.e. randomly distort some of the images in X)
                        X = self._augment(X)
                        # Put the data in a queue
                        queue.put((X, y))
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

    def gen_batch_colorful(self, X, nn_finder, nb_q, prior_factor):
        """Generate batch, assuming X_color, X_black are loaded in memory in the main program"""

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
            # Y_batch = Y_batch.transpose((0,3,1,2))

            yield X_batch_black, X_batch_color, Y_batch

    def gen_batch_pretraining(self, X_color, X_black, small_v=False):
        """Generate batch, assuming X_color, X_black are loaded in memory in the main program"""

        while True:
            # Select idx at random for the batch
            if not small_v:
                idx = np.random.choice(X_color.shape[0], self.batch_size, replace=False)
            else:
                idx = np.random.choice(X_color.shape[0], self.batch_size / 8, replace=False)
                idx = np.repeat(idx, 8)
                idx = np.ravel(idx)

            # Get X and y
            X_batch_color = X_color[idx]
            X_batch_black = X_black[idx]

            yield X_batch_color, X_batch_black

    def gen_batch_inmemory(self, X_color, X_black, batch_size=None):
        """Generate batch, assuming X_color, X_black are loaded in memory in the main program"""

        if not batch_size:
            batch_size = self.batch_size

        while True:
            # Select idx at random for the batch
            idx_color = np.random.choice(X_color.shape[0], batch_size, replace=False)
            idx_black = np.random.choice(X_black.shape[0], batch_size, replace=False)

            # Get X and y
            X_batch_color = X_color[idx_color]
            X_batch_black = X_black[idx_black]
            X_batch_color_ori = X_color[idx_black]

            X_batch_color = self._augment(X_batch_color)
            X_batch_black = self._augment(X_batch_black)
            yield X_batch_color, X_batch_black, X_batch_color_ori
