import os
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt

FLAGS = tf.app.flags.FLAGS


def save_image(data, data_format, e, suffix=None):
    """Saves a picture showing the current progress of the model"""

    X_G, X_real = data

    Xg = X_G[:8]
    Xr = X_real[:8]

    if data_format == "NHWC":
        X = np.concatenate((Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] / 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=0)

    if data_format == "NCHW":
        X = np.concatenate((Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] / 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=1)
        Xr = Xr.transpose(1,2,0)

    if Xr.shape[-1] == 1:
        plt.imshow(Xr[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xr)
    plt.axis("off")
    if suffix is None:
        plt.savefig(os.path.join(FLAGS.fig_dir, "current_batch_%s.png" % e))
    else:
        plt.savefig(os.path.join(FLAGS.fig_dir, "current_batch_%s_%s.png" % (suffix, e)))
    plt.clf()
    plt.close()


def get_stacked_tensor(X1, X2):

    X = tf.concat((X1[:16], X2[:16]), axis=0)
    list_rows = []
    for i in range(8):
        Xr = tf.concat([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
        list_rows.append(Xr)

    X = tf.concat(list_rows, axis=1)
    X = tf.transpose(X, (1,2,0))
    X = tf.expand_dims(X, 0)

    return X
