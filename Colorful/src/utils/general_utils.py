import os
import numpy as np
from skimage import color
import matplotlib.pylab as plt


def remove_files(files):
    """
    Remove files from disk

    args: files (str or list) remove all files in 'files'
    """

    if isinstance(files, (list, tuple)):
        for f in files:
            if os.path.isfile(os.path.expanduser(f)):
                os.remove(f)
    elif isinstance(files, str):
        if os.path.isfile(os.path.expanduser(files)):
            os.remove(files)


def create_dir(dirs):
    """
    Create directory

    args: dirs (str or list) create all dirs in 'dirs'
    """

    if isinstance(dirs, (list, tuple)):
        for d in dirs:
            if not os.path.exists(os.path.expanduser(d)):
                os.makedirs(d)
    elif isinstance(dirs, str):
        if not os.path.exists(os.path.expanduser(dirs)):
            os.makedirs(dirs)


def setup_logging(model_name):

    model_dir = "../../models"
    # Output path where we store experiment log and weights
    model_dir = os.path.join(model_dir, model_name)

    fig_dir = "../../figures"

    # Create if it does not exist
    create_dir([model_dir, fig_dir])


def plot_batch(color_model, q_ab, X_batch_black, X_batch_color, batch_size, h, w, nb_q, epoch):

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

    list_img = []
    for i, img in enumerate(X_colorized[:min(32, batch_size)]):
        arr = np.concatenate([X_batch_color[i], np.repeat(X_batch_black[i] / 100., 3, axis=0), img], axis=2)
        list_img.append(arr)

    plt.figure(figsize=(20,20))
    list_img = [np.concatenate(list_img[4 * i: 4 * (i + 1)], axis=2) for i in range(len(list_img) / 4)]
    arr = np.concatenate(list_img, axis=1)
    plt.imshow(arr.transpose(1,2,0))
    ax = plt.gca()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.tight_layout()
    plt.savefig("../../figures/fig_epoch%s.png" % epoch)
    plt.clf()
    plt.close()


def plot_batch_eval(color_model, q_ab, X_batch_black, X_batch_color, batch_size, h, w, nb_q, T):

    # Format X_colorized
    X_colorized = color_model.predict(X_batch_black / 100.)[:, :, :, :-1]
    X_colorized = X_colorized.reshape((batch_size * h * w, nb_q))

    # Reweight probas
    X_colorized = np.exp(np.log(X_colorized) / T)
    X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

    # Reweighted
    q_a = q_ab[:, 0].reshape((1, 313))
    q_b = q_ab[:, 1].reshape((1, 313))

    X_a = np.sum(X_colorized * q_a, 1).reshape((batch_size, 1, h, w))
    X_b = np.sum(X_colorized * q_b, 1).reshape((batch_size, 1, h, w))

    X_colorized = np.concatenate((X_batch_black, X_a, X_b), axis=1).transpose(0, 2, 3, 1)
    X_colorized = [np.expand_dims(color.lab2rgb(im), 0) for im in X_colorized]
    X_colorized = np.concatenate(X_colorized, 0).transpose(0, 3, 1, 2)

    X_batch_color = [np.expand_dims(color.lab2rgb(im.transpose(1, 2, 0)), 0) for im in X_batch_color]
    X_batch_color = np.concatenate(X_batch_color, 0).transpose(0, 3, 1, 2)

    list_img = []
    for i, img in enumerate(X_colorized[:min(32, batch_size)]):
        arr = np.concatenate([X_batch_color[i], np.repeat(X_batch_black[i] / 100., 3, axis=0), img], axis=2)
        list_img.append(arr)

    plt.figure(figsize=(20,20))
    list_img = [np.concatenate(list_img[4 * i: 4 * (i + 1)], axis=2) for i in range(len(list_img) / 4)]
    arr = np.concatenate(list_img, axis=1)
    plt.imshow(arr.transpose(1,2,0))
    ax = plt.gca()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.tight_layout()
    plt.show()
