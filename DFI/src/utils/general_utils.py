import os
import numpy as np
from keras import backend as K


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


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def deprocess_image(x, img_nrows, img_ncols):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def color_correction(x, img_nrows, img_ncols, X_source):

    # save current generated image
    img = deprocess_image(x.copy(), img_nrows, img_ncols).astype(np.float64)
    X_sourceT = X_source[0].copy().transpose(1,2,0).astype(np.float64)
    # Color correction
    for k in range(3):
        mean, std = np.mean(X_sourceT[:, :, k]), np.std(X_sourceT[:, :, k])
        img[:, :, k] *= std / np.std(img[:, :, k])
        img[:, :, k] += mean - np.mean(img[:, :, k])

    img = img.clip(0, 255)

    return img