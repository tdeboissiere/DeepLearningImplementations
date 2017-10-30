from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import h5py

import matplotlib.pylab as plt


def normalization(X):

    return X / 127.5 - 1


def inverse_normalization(X):

    return (X + 1.) / 2.


def get_nb_patch(img_dim, patch_size, image_data_format):

    assert image_data_format in ["channels_first", "channels_last"], "Bad image_data_format"

    if image_data_format == "channels_first":
        assert img_dim[1] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[2] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[1] // patch_size[0]) * (img_dim[2] // patch_size[1])
        img_dim_disc = (img_dim[0], patch_size[0], patch_size[1])

    elif image_data_format == "channels_last":
        assert img_dim[0] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[1] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1])
        img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1])

    return nb_patch, img_dim_disc


def extract_patches(X, image_data_format, patch_size):

    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        X = X.transpose(0,2,3,1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)

    return list_X


def load_data(dset, image_data_format):

    with h5py.File("../../data/processed/%s_data.h5" % dset, "r") as hf:

        X_full_train = hf["train_data_full"][:].astype(np.float32)
        X_full_train = normalization(X_full_train)

        X_sketch_train = hf["train_data_sketch"][:].astype(np.float32)
        X_sketch_train = normalization(X_sketch_train)

        if image_data_format == "channels_last":
            X_full_train = X_full_train.transpose(0, 2, 3, 1)
            X_sketch_train = X_sketch_train.transpose(0, 2, 3, 1)

        X_full_val = hf["val_data_full"][:].astype(np.float32)
        X_full_val = normalization(X_full_val)

        X_sketch_val = hf["val_data_sketch"][:].astype(np.float32)
        X_sketch_val = normalization(X_sketch_val)

        if image_data_format == "channels_last":
            X_full_val = X_full_val.transpose(0, 2, 3, 1)
            X_sketch_val = X_sketch_val.transpose(0, 2, 3, 1)

        return X_full_train, X_sketch_train, X_full_val, X_sketch_val


def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        yield X1[idx], X2[idx]


def get_disc_batch(X_full_batch, X_sketch_batch, generator_model, batch_counter, patch_size,
                   image_data_format, label_smoothing=False, label_flipping=0):

    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator_model.predict(X_sketch_batch)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = X_full_batch
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc
    X_disc = extract_patches(X_disc, image_data_format, patch_size)

    return X_disc, y_disc


def plot_generated_batch(X_full, X_sketch, generator_model, batch_size, image_data_format, suffix):

    # Generate images
    X_gen = generator_model.predict(X_sketch)

    X_sketch = inverse_normalization(X_sketch)
    X_full = inverse_normalization(X_full)
    X_gen = inverse_normalization(X_gen)

    Xs = X_sketch[:8]
    Xg = X_gen[:8]
    Xr = X_full[:8]

    if image_data_format == "channels_last":
        X = np.concatenate((Xs, Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] // 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=0)

    if image_data_format == "channels_first":
        X = np.concatenate((Xs, Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] // 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=1)
        Xr = Xr.transpose(1,2,0)

    if Xr.shape[-1] == 1:
        plt.imshow(Xr[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xr)
    plt.axis("off")
    plt.savefig("../../figures/current_batch_%s.png" % suffix)
    plt.clf()
    plt.close()
