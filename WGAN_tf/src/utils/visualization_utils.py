import os
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import matplotlib as mp

FLAGS = tf.app.flags.FLAGS


def format_plot(X, epoch=None, title=None, figsize=(15, 10)):

    plt.figure(figsize=figsize)

    if X.shape[-1] == 1:
        plt.imshow(X[:, :, 0], cmap="gray")
    else:
        plt.imshow(X)

    plt.axis("off")
    plt.gca().xaxis.set_major_locator(mp.ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(mp.ticker.NullLocator())

    if epoch is not None and title is None:
        save_path = os.path.join(FLAGS.fig_dir, "current_batch_%s.png" % epoch)
    elif epoch is not None and title is not None:
        save_path = os.path.join(FLAGS.fig_dir, "%s_%s.png" % (title, epoch))
    elif title is not None:
        save_path = os.path.join(FLAGS.fig_dir, "%s.png" % title)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()


def save_image(X1, X2, e=None, title=None):

    if FLAGS.data_format == "NCHW":
        X1 = X1.transpose((0, 2, 3, 1))
        X2 = X2.transpose((0, 2, 3, 1))

    Xup = X1[:32]
    Xdown = X2[:32]

    n_cols = 8

    list_rows_f = []
    for i in range(Xup.shape[0] // n_cols):
        Xrow = np.concatenate([Xup[k] for k in range(n_cols * i, n_cols * (i + 1))], axis=1)
        list_rows_f.append(Xrow)
    list_rows_r = []
    for i in range(Xup.shape[0] // n_cols):
        Xrow = np.concatenate([Xdown[k] for k in range(n_cols * i, n_cols * (i + 1))], axis=1)
        list_rows_r.append(Xrow)

    Xup = np.concatenate(list_rows_f, axis=0)
    Xdown = np.concatenate(list_rows_r, axis=0)

    X_ones = 255 * np.ones_like(Xup, dtype=np.uint8)
    X_ones = X_ones[:5, :, :]

    X = np.concatenate((Xup, X_ones, Xdown), axis=0)

    format_plot(X, epoch=e, title=title)
