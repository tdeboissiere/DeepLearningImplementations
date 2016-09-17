import keras.backend as K
import numpy as np
import matplotlib.pylab as plt
import cPickle as pickle
import sys
import os


def find_top9_mean_act(data, Dec, target_layer, feat_map, batch_size=32):
    """
    Find images with highest mean activation

    args:  data (numpy array) the image data
           shape : (n_samples, n_channels, img_dim1, img_dim2)

           Dec (DeconvNet) instance of the DeconvNet class

           target_layer (str) Layer name we want to visualise

           feat_map (int) index of the filter to visualise

           batch_size (int) batch size

    returns: top9 (numpy array) index of the top9 images that activate feat_map
    """

    # Theano function to get the layer output
    T_in, T_out = Dec[Dec.model.layers[0].name].input, Dec[target_layer].output
    get_activation = K.function([T_in], T_out)

    list_max = []
    # Loop over batches and store the max activation value for each
    # image in data for the target layer and target feature map
    for nbatch in range(data.shape[0] / batch_size):
        sys.stdout.write("\rProcessing batch %s/%s" %
                         (nbatch + 1, len(range(data.shape[0] / batch_size))))
        sys.stdout.flush()
        X = data[nbatch * batch_size: (nbatch + 1) * batch_size]
        Dec.model.predict(X)
        X_activ = get_activation([X])[:, feat_map, :, :]
        X_sum = np.sum(X_activ, axis=(1,2))
        list_max += X_sum.tolist()
    # Only keep the top 9 activations
    list_max = np.array(list_max)
    i_sort = np.argsort(list_max)
    top9 = i_sort[-9:]
    print
    return top9


def get_deconv_images(d_act_path, d_deconv_path, data, Dec):
    """
    Deconvolve images specified in d_act. Then pickle these images
    for future use

    args:  d_act_path (str) path to the dict that for each target layer
                       and for a selection of feature maps, holds the index
                       of the top9 images activating said feature maps

            d_deconv_path (str) path to the dict that for each target layer
                       and for a selection of feature maps, holds the deconv
                       result of the top9 images activating said feature maps

           data (numpy array) the image data
           shape : (n_samples, n_channels, img_dim1, img_dim2)

           Dec (DeconvNet) instance of the DeconvNet class
    """

    # Load d_act
    with open(d_act_path, 'r') as f:
        d_act = pickle.load(f)

    # Get the list of target layers
    list_target = d_act.keys()

    # Store deconv images in d_deconv
    d_deconv = {}

    # Iterate over target layers and feature maps
    # and store the deconv image
    for target_layer in list_target:
        list_feat_map = d_act[target_layer].keys()
        for feat_map in list_feat_map:
            top9 = d_act[target_layer][feat_map]
            X = data[top9]
            X_out = Dec.get_deconv(X, target_layer, feat_map=feat_map)
            key = target_layer + "_feat_" + str(feat_map)
            d_deconv[key] = X_out

    np.savez("./Data/dict_top9_deconv.npz", **d_deconv)


def format_array(arr):
    """
    Utility to format array for tiled plot

    args: arr (numpy array)
            shape : (n_samples, n_channels, img_dim1, img_dim2)
    """
    n_channels = arr.shape[1]
    len_arr = arr.shape[0]
    assert (n_channels == 1 or n_channels == 3), "n_channels should be 1 (Greyscale) or 3 (Color)"
    if n_channels == 1:
        arr = np.repeat(arr, 3, axis=1)

    shape1, shape2 = arr.shape[-2:]
    arr = np.transpose(arr, [1, 0, 2, 3])
    arr = arr.reshape([3, len_arr, shape1 * shape2]).astype(np.float64)
    arr = tuple([arr[i] for i in xrange(3)] + [None])
    return arr, shape1, shape2


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            # Convert to uint to make it look like an image indeed
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                out_array[:, :, i] = np.zeros(out_shape,
                                              dtype='uint8'
                                              if output_pixel_vals else out_array.dtype
                                              ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] \
                        = this_img * (255 if output_pixel_vals else 1)
        return out_array


def plot_max_activation(d_act_path, d_deconv_path, data, target_layer, save=False):
    """
    Plot original images (or cropped versions) and the deconvolution result
    for images specified in d_act_path / d_dedeconv_path

    args:  d_act_path (str) path to the dict that for each target layer
                       and for a selection of feature maps, holds the index
                       of the top9 images activating said feature maps

            d_deconv_path (str) path to the dict that for each target layer
                       and for a selection of feature maps, holds the deconv
                       result of the top9 images activating said feature maps

           data (numpy array) the image data
           shape : (n_samples, n_channels, img_dim1, img_dim2)

           target_layer (str) name of the layer we want to visualise

           save (bool) whether or not to save the result to a Figures folder
    """

    # Load d_deconv
    d_deconv = {}
    arr_deconv = np.load(d_deconv_path)
    for key in arr_deconv.keys():
        layer, fmap = key.split("_feat_")
        fmap = int(fmap)
        try:
            d_deconv[layer][fmap] = arr_deconv[key]
        except KeyError:
            d_deconv[layer] = {fmap: arr_deconv[key]}

    # Load d_act
    with open(d_act_path, 'r') as f:
        d_act = pickle.load(f)

    # Get the list of feature maps
    list_feat_map = d_act[target_layer].keys()

    # We'll crop images to identify the region
    # the neuron has activated on
    # dict to store cropped images
    d_crop_deconv = {i: [] for i in list_feat_map}
    d_crop_ori = {i: [] for i in list_feat_map}

    # This will hold the image dimensions
    max_delta_x = 0
    max_delta_y = 0

    # To crop images:
    # First loop to get the largest image size required (bounding box)
    for feat_map in list_feat_map:
        X_deconv = d_deconv[target_layer][feat_map]
        for k in range(X_deconv.shape[0]):
            arr = np.argwhere(np.max(X_deconv[k], axis=0))
            try:
                (ystart, xstart), (ystop, xstop) = arr.min(0), arr.max(0) + 1
            except ValueError:
                print "Encountered a dead filter"
                return
            delta_x = xstop - xstart
            delta_y = ystop - ystart
            if delta_x > max_delta_x:
                max_delta_x = delta_x
            if delta_y > max_delta_y:
                max_delta_y = delta_y

    # Then loop to crop all images to the same size
    for feat_map in range(len(list_feat_map)):
        X_deconv = d_deconv[target_layer][feat_map]
        X_ori = data[d_act[target_layer][feat_map]]

        for k in range(X_deconv.shape[0]):
            arr = np.argwhere(np.max(X_deconv[k], axis=0))
            try:
                (ystart, xstart), (ystop, xstop) = arr.min(0), arr.max(0) + 1
            except ValueError:
                print "Encountered a dead filter"
                return
            # Specific case to avoid array boundary issues
            y_min, y_max = ystart, ystart + max_delta_y
            if y_max >= X_deconv[k].shape[-2]:
                y_min = y_min - (y_max - X_deconv[k].shape[-2])
                y_max = X_deconv[k].shape[-2]

            x_min, x_max = xstart, xstart + max_delta_x
            if x_max >= X_deconv[k].shape[-1]:
                x_min = x_min - (x_max - X_deconv[k].shape[-1])
                x_max = X_deconv[k].shape[-1]

            # Store the images in the dict
            arr_deconv = X_deconv[k, :, y_min: y_max, x_min: x_max]
            d_crop_deconv[feat_map].append(arr_deconv)

            arr_ori = X_ori[k, :, y_min: y_max, x_min: x_max]
            d_crop_ori[feat_map].append(arr_ori)

        d_crop_deconv[feat_map] = np.array(d_crop_deconv[feat_map])
        d_crop_ori[feat_map] = np.array(d_crop_ori[feat_map])

    # List to hold the images in the tiled plot
    list_input_img = []
    list_output_img = []

    # Loop over the feat maps to fill the lists above
    for feat_map in list_feat_map:

        arr_ori = d_crop_ori[feat_map]
        arr_deconv = d_crop_deconv[feat_map]

        arr_ori, shape1, shape2 = format_array(arr_ori)
        arr_deconv, shape1, shape2 = format_array(arr_deconv)

        input_map = tile_raster_images(arr_ori, img_shape=(shape1, shape2), tile_shape=(3, 3),
                                       tile_spacing=(1,1), scale_rows_to_unit_interval=True,
                                       output_pixel_vals=True)

        output_map = tile_raster_images(arr_deconv, img_shape=(shape1, shape2), tile_shape=(3, 3),
                                        tile_spacing=(1,1), scale_rows_to_unit_interval=True,
                                        output_pixel_vals=True)

        list_input_img.append(input_map)
        list_output_img.append(output_map)

    # Format the arrays for the plot
    arr_ori1 = np.vstack(list_input_img[::2])
    arr_ori2 = np.vstack(list_input_img[1::2])
    arr_dec1 = np.vstack(list_output_img[::2])
    arr_dec2 = np.vstack(list_output_img[1::2])
    arr_ori = np.hstack((arr_ori1, arr_dec1))
    arr_dec = np.hstack((arr_ori2, arr_dec2))
    arr_full = np.hstack((arr_dec, arr_ori))

    # RGB/GBR reordering
    arr_full_copy = arr_full.copy()
    arr_full[:, :, 0] = arr_full_copy[:, :, 2]
    arr_full[:, :, 1] = arr_full_copy[:, :, 1]
    arr_full[:, :, 2] = arr_full_copy[:, :, 0]
    del arr_full_copy

    # Plot and prettify
    plt.imshow(arr_full, aspect='auto')
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    labelbottom='off', right='off', left='off', labelleft='off')
    plt.xlabel(target_layer, fontsize=26)
    # plt.axis("off")
    plt.tight_layout()
    if save:
        if not os.path.exists("./Figures/"):
            os.makedirs("./Figures")
        plt.savefig("./Figures/%s.png" % target_layer, format='png', dpi=200)
    else:
        plt.show()
        raw_input()


def plot_deconv(img_index, data, Dec, target_layer, feat_map, save=False):
    """
    Plot original images (or cropped versions) and the deconvolution result
    for images specified in img_index, for the target layer and feat_map
    specified in the arguments

    args:  img_index (list/arr) array or list of index. These are the indices
                                of the images we want to plot

           data (numpy array) the image data
           shape : (n_samples, n_channels, img_dim1, img_dim2)

           Dec (DeconvNet) instance of the DeconvNet class

           target_layer (str) name of the layer we want to visualise

           feat_map (int) index of the filter to visualise
    """

    num_img = len(img_index)
    assert np.isclose(np.sqrt(num_img), int(
        np.sqrt(num_img))), "len(img_index) must be a perfect square"
    mosaic_size = int(np.sqrt(num_img))

    X_ori = data[img_index]
    X_deconv = Dec.get_deconv(data[img_index], target_layer, feat_map=feat_map)

    max_delta_x = 0
    max_delta_y = 0

    # To crop images:
    # First loop to get the largest image size required (bounding box)
    for k in range(X_deconv.shape[0]):
        arr = np.argwhere(np.max(X_deconv[k], axis=0))
        try:
            (ystart, xstart), (ystop, xstop) = arr.min(0), arr.max(0) + 1
        except ValueError:
            print "Encountered a dead filter, retry with different img/filter"
            return
        delta_x = xstop - xstart
        delta_y = ystop - ystart
        if delta_x > max_delta_x:
            max_delta_x = delta_x
        if delta_y > max_delta_y:
            max_delta_y = delta_y

    list_deconv = []
    list_ori = []

    # Then loop to crop all images to the same size
    for k in range(X_deconv.shape[0]):
        arr = np.argwhere(np.max(X_deconv[k], axis=0))
        try:
            (ystart, xstart), (ystop, xstop) = arr.min(0), arr.max(0) + 1
        except ValueError:
            print "Encountered a dead filter, retry with different img/filter"
            return
        # Specific case to avoid array boundary issues
        y_min, y_max = ystart, ystart + max_delta_y
        if y_max >= X_deconv[k].shape[-2]:
            y_min = y_min - (y_max - X_deconv[k].shape[-2])
            y_max = X_deconv[k].shape[-2]

        x_min, x_max = xstart, xstart + max_delta_x
        if x_max >= X_deconv[k].shape[-1]:
            x_min = x_min - (x_max - X_deconv[k].shape[-1])
            x_max = X_deconv[k].shape[-1]

        # Store the images in the dict
        arr_deconv = X_deconv[k, :, y_min: y_max, x_min: x_max]
        arr_ori = X_ori[k, :, y_min: y_max, x_min: x_max]

        list_ori.append(arr_ori)
        list_deconv.append(arr_deconv)

    arr_deconv = np.array(list_deconv)
    arr_ori = np.array(list_ori)

    arr_ori, shape1, shape2 = format_array(arr_ori)
    arr_deconv, _, _ = format_array(arr_deconv)

    input_map = tile_raster_images(arr_ori, img_shape=(shape1, shape2),
                                   tile_shape=(mosaic_size, mosaic_size),
                                   tile_spacing=(1, 1), scale_rows_to_unit_interval=True,
                                   output_pixel_vals=True)

    output_map = tile_raster_images(arr_deconv, img_shape=(shape1, shape2),
                                    tile_shape=(mosaic_size, mosaic_size),
                                    tile_spacing=(1, 1), scale_rows_to_unit_interval=True,
                                    output_pixel_vals=True)

    arr_full = np.append(input_map, output_map, axis=1)

    # RGB/GBR reordering
    arr_full_copy = arr_full.copy()
    arr_full[:, :, 0] = arr_full_copy[:, :, 2]
    arr_full[:, :, 1] = arr_full_copy[:, :, 1]
    arr_full[:, :, 2] = arr_full_copy[:, :, 0]
    del arr_full_copy

    # Plot and prettify
    plt.imshow(arr_full, aspect='auto')
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    labelbottom='off', right='off', left='off', labelleft='off')
    plt.xlabel(target_layer + "Filter: %s" % feat_map, fontsize=26)
    # plt.axis("off")
    plt.tight_layout()
    if save:
        if not os.path.exists("./Figures/"):
            os.makedirs("./Figures")
        plt.savefig("./Figures/sample_%s.png" % target_layer, format='png', dpi=200)
    else:
        plt.show()
        raw_input()
