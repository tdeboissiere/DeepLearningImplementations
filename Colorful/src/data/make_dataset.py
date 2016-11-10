import os
import cv2
import h5py
import parmap
import argparse
import numpy as np
import cPickle as pickle
from skimage import color
from tqdm import tqdm as tqdm
import sklearn.neighbors as nn
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.signal import gaussian, convolve


def format_image(img_path, size):
    """
    Load img with opencv and reshape
    """

    img_color = cv2.imread(img_path)
    img_color = img_color[:, :, ::-1]
    img_black = cv2.imread(img_path, 0)

    img_color = cv2.resize(img_color, (size, size), interpolation=cv2.INTER_AREA)
    img_black = cv2.resize(img_black, (size, size), interpolation=cv2.INTER_AREA)

    img_lab = color.rgb2lab(img_color)

    img_lab = img_lab.reshape((1, size, size, 3)).transpose(0, 3, 1, 2)
    img_color = img_color.reshape((1, size, size, 3)).transpose(0, 3, 1, 2)
    img_black = img_black.reshape((1, size, size, 1)).transpose(0, 3, 1, 2)

    return img_color, img_lab, img_black


def build_HDF5(dset_type, size=64):
    """
    Gather the data in a single HDF5 file.
    """

    # Read evaluation file, build it if it does not exist
    # In evaluation status, "0" represents training image, "1" represents
    # validation image, "2" represents testing image;
    d_eval = {}
    try:
        with open(data_dir + "d_eval.pickle", "r") as fd:
            print "Loading d_eval"
            d_eval = pickle.load(fd)
    except:
        with open(raw_dir + "Eval/list_eval_partition.txt", "r") as f:
            lines = f.readlines()
            for celeb in lines:
                celeb = celeb.rstrip().split()
                img = celeb[0]
                attrs = int(celeb[1])
                d_eval[img] = attrs
        with open(data_dir + "d_eval.pickle", "w") as fd:
            pickle.dump(d_eval, fd)

    # Get the list of jpg files
    list_img = []
    if dset_type == "training":
        for img in d_eval.keys():
            if d_eval[img] == 0:
                list_img.append(os.path.join(raw_dir, "img_align_celeba", img))
    elif dset_type == "validation":
        for img in d_eval.keys():
            if d_eval[img] == 1:
                list_img.append(os.path.join(raw_dir, "img_align_celeba", img))
    elif dset_type == "test":
        for img in d_eval.keys():
            if d_eval[img] == 2:
                list_img.append(os.path.join(raw_dir, "img_align_celeba", img))

    # Shuffle images
    np.random.seed(20)
    p = np.random.permutation(len(list_img))
    list_img = np.array(list_img)[p]

    # Put train data in HDF5
    hdf5_file = os.path.join(data_dir, "%s_%s_data.h5" % (dset_type, size))
    with h5py.File(hdf5_file, "w") as hfw:

        data_color = hfw.create_dataset("%s_color_data" % dset_type,
                                        (0, 3, size, size),
                                        maxshape=(None, 3, size, size),
                                        dtype=np.uint8)

        data_lab = hfw.create_dataset("%s_lab_data" % dset_type,
                                      (0, 3, size, size),
                                      maxshape=(None, 3, size, size),
                                      dtype=np.float64)

        data_black = hfw.create_dataset("%s_black_data" % dset_type,
                                        (0, 1, size, size),
                                        maxshape=(None, 1, size, size),
                                        dtype=np.uint8)

        num_files = len(list_img)
        chunk_size = 1000
        num_chunks = num_files / chunk_size
        arr_chunks = np.array_split(np.arange(num_files), num_chunks)

        for chunk_idx in tqdm(arr_chunks):

            list_img_path = list_img[chunk_idx].tolist()
            output = parmap.map(format_image, list_img_path, size, parallel=True)

            arr_img_color = np.vstack([o[0] for o in output if o[0].shape[0] > 0])
            arr_img_lab = np.vstack([o[1] for o in output if o[0].shape[0] > 0])
            arr_img_black = np.vstack([o[2] for o in output if o[0].shape[0] > 0])

            # Resize HDF5 dataset
            data_color.resize(data_color.shape[0] + arr_img_color.shape[0], axis=0)
            data_lab.resize(data_lab.shape[0] + arr_img_lab.shape[0], axis=0)
            data_black.resize(data_black.shape[0] + arr_img_black.shape[0], axis=0)

            data_color[-arr_img_color.shape[0]:] = arr_img_color.astype(np.uint8)
            data_lab[-arr_img_lab.shape[0]:] = arr_img_lab.astype(np.float64)
            data_black[-arr_img_black.shape[0]:] = arr_img_black.astype(np.uint8)


def compute_color_prior(dset_type, size=64, do_plot=False):

    # Load the gamut points location
    q_ab = np.load(os.path.join(data_dir, "pts_in_hull.npy"))

    if do_plot:
        plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0])
        for i in range(q_ab.shape[0]):
            ax.scatter(q_ab[:, 0], q_ab[:, 1])
            ax.annotate(str(i), (q_ab[i, 0], q_ab[i, 1]), fontsize=6)
            ax.set_xlim([-110,110])
            ax.set_ylim([-110,110])

    with h5py.File(os.path.join(data_dir, "%s_%s_data.h5" % (dset_type, size)), "a") as hf:
        X_ab = hf["training_lab_data"][:100000][:, 1:, :, :]
        npts, c, h, w = X_ab.shape
        X_a = np.ravel(X_ab[:, 0, :, :])
        X_b = np.ravel(X_ab[:, 1, :, :])
        X_ab = np.vstack((X_a, X_b)).T

        if do_plot:
            plt.hist2d(X_ab[:, 0], X_ab[:, 1], bins=100, norm=LogNorm())
            plt.xlim([-110, 110])
            plt.ylim([-110, 110])
            plt.colorbar()
            plt.show()
            plt.clf()
            plt.close()

        # Create nearest neighbord instance with index = q_ab
        NN = 1
        nearest = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(q_ab)
        # Find index of nearest neighbor for X_ab
        dists, ind = nearest.kneighbors(X_ab)

        # We now count the number of occurrences of each color
        ind = np.ravel(ind)
        counts = np.bincount(ind)
        idxs = np.nonzero(counts)[0]
        prior_prob = np.zeros((313))
        for i in range(313):
            prior_prob[idxs] = counts[idxs]

        # We turn this into a color probability
        prior_prob = prior_prob / (1.0 * np.sum(prior_prob))

        # Save
        np.save(os.path.join(data_dir, "%s_%s_prior_prob.npy" % (dset_type, size), prior_prob))

        if do_plot:
            plt.hist(prior_prob, bins=100)
            plt.yscale("log")
            plt.show()


def smooth_color_prior(dset_type, size=64, sigma=5, do_plot=False):

    prior_prob = np.load(os.path.join(data_dir, "training_%s_prior_prob.npy" % size))
    # add an epsilon to prior prob to avoid 0 vakues and possible NaN
    prior_prob += 1E-3 * np.min(prior_prob)
    # renormalize
    prior_prob = prior_prob / (1.0 * np.sum(prior_prob))

    # Smooth with gaussian
    f = interp1d(np.arange(prior_prob.shape[0]),prior_prob)
    xx = np.linspace(0,prior_prob.shape[0] - 1, 1000)
    yy = f(xx)
    window = gaussian(2000, sigma)  # 2000 pts in the window, sigma=5
    smoothed = convolve(yy, window / window.sum(), mode='same')
    fout = interp1d(xx,smoothed)
    prior_prob_smoothed = np.array([fout(i) for i in range(prior_prob.shape[0])])
    prior_prob_smoothed = prior_prob_smoothed / np.sum(prior_prob_smoothed)

    # Save
    file_name = os.path.join(data_dir, "%s_%s_prior_prob_smoothed.npy" % (dset_type, size))
    np.save(file_name, prior_prob_smoothed)

    if do_plot:
        plt.plot(prior_prob)
        plt.plot(prior_prob_smoothed, "g--")
        plt.plot(xx, smoothed, "r-")
        plt.yscale("log")
        plt.show()


def compute_prior_factor(dset_type, size=64, gamma=0.5, alpha=1, do_plot=False):

    file_name = os.path.join(data_dir, "%s_%s_prior_prob_smoothed.npy" % (dset_type, size))
    prior_prob_smoothed = np.load(file_name)

    u = np.ones_like(prior_prob_smoothed)
    u = u / np.sum(1.0 * u)

    prior_factor = (1 - gamma) * prior_prob_smoothed + gamma * u
    prior_factor = np.power(prior_factor, -alpha)

    # renormalize
    prior_factor = prior_factor / (np.sum(prior_factor * prior_prob_smoothed))

    file_name = "../../data/processed/%s_%s_prior_factor.npy" % (dset_type, size)
    np.save(file_name, prior_factor)

    if do_plot:
        plt.plot(prior_factor)
        plt.yscale("log")
        plt.show()


def check_HDF5(dset_type):
    """
    Plot images with landmarks to check the processing
    """

    # Get processed data directory
    data_dir = os.path.expanduser(os.environ.get("DATA_DIR"))

    # Get hdf5 file
    hdf5_file = os.path.join(data_dir, "%s_64_data.h5" % dset_type)

    with h5py.File(hdf5_file, "r") as hf:
        data_color = hf["%s_color_data" % dset_type]
        data_lab = hf["%s_lab_data" % dset_type]
        data_black = hf["%s_black_data" % dset_type]
        for i in range(data_color.shape[0]):
            fig = plt.figure()
            gs = gridspec.GridSpec(3, 1)
            for k in range(3):
                ax = plt.subplot(gs[k])
                if k == 0:
                    img = data_color[i, :, :, :].transpose(1,2,0)
                    ax.imshow(img)
                elif k == 1:
                    img = data_lab[i, :, :, :].transpose(1,2,0)
                    img = color.lab2rgb(img)
                    ax.imshow(img)
                elif k == 2:
                    img = data_black[i, 0, :, :] / 255.
                    ax.imshow(img, cmap="gray")
            gs.tight_layout(fig)
            plt.show()
            plt.clf()
            plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build dataset')
    parser.add_argument('list_datasets', type=str, nargs='+',
                        help='List of dataset names. Choose training, validation or test')
    parser.add_argument('--img_size', default=64, type=int,
                        help='Desired Width == Height')
    parser.add_argument('--do_plot', default=False, type=bool,
                        help='Whether to visualize statistics when computing color prior')
    parser.add_argument('--data_dir', default="../../data/processed", type=str,
                        help='Path where to save the processed data')
    parser.add_argument('--raw_dir', default="../../data/raw", type=str,
                        help='Path where the raw Celeba data was saved in Step 1')

    args = parser.parse_args()

    data_dir = args.raw_dir
    raw_dir = args.data_dir

    for d in [raw_dir, data_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    for dset_type in args.list_datasets:
        assert dset_type in ["training", "validation", "test"]

        build_HDF5(dset_type, size=args.img_size)
        compute_color_prior("training")
        smooth_color_prior("training")
        compute_prior_factor("training", do_plot=True)
