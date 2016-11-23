import os
import cv2
import sys
import glob
import time
import h5py
import numpy as np
import pandas as pd
import keras.backend as K
from scipy.misc import imsave
from keras.models import Model
import sklearn.neighbors as nn
from scipy.optimize import fmin_l_bfgs_b
# Utils
sys.path.append("../utils")
import general_utils


def launch_dfi(**kwargs):

    data_file = "../../data/processed/lfw_200_data.h5"
    attributes_file = "../../data/processed/lfw_processed_attributes.csv"

    nb_neighbors = 100
    alpha = 4
    weight_reverse_mapping = 1
    weight_total_variation = 1E3

    nb_neighbors = kwargs["nb_neighbors"]
    alpha = kwargs["alpha"]
    attributes_file = kwargs["attributes_file"]
    data_file = kwargs["data_file"]
    normalize_w = kwargs["normalize_w"]
    keras_model_path = kwargs["keras_model_path"]
    weight_reverse_mapping = kwargs["weight_reverse_mapping"]
    weight_total_variation = kwargs["weight_total_variation"]

    # Import vgg19 model
    sys.path.append(keras_model_path)
    from vgg19 import VGG19

    list_img = glob.glob("../../figures/*.png")
    for f in list_img:
        os.remove(f)

    # Load and rescale data
    with h5py.File(data_file, "r") as hf:
        X = hf["data"][:]

        img_nrows, img_ncols = X.shape[-2:]

        df_attrs = pd.read_csv(attributes_file)
        list_col_labels = [c for c in df_attrs.columns.values
                           if c not in ["person", "imagenum", "image_path"]]

        # Select middle aged white male no eyeglasses no mustache Fully visible forehead
        cond = (df_attrs["Male"] > 0.4)
        cond = cond & (df_attrs["Middle Aged"] > -0.1)
        cond = cond & (df_attrs["No Eyewear"] > 0)
        cond = cond & (df_attrs["Mustache"] < 0)
        cond = cond & (df_attrs["Fully Visible Forehead"] > 0)
        idx_middle = np.where(cond)[0]
        # Select aged white male no eyeglasses no mustache Fully visible forehead
        cond = (df_attrs["Male"] > 0.4)
        cond = cond & (df_attrs["Senior"] > -0.3)
        cond = cond & (df_attrs["No Eyewear"] > 0)
        cond = cond & (df_attrs["Mustache"] < 0)
        cond = cond & (df_attrs["Fully Visible Forehead"] > 0)
        idx_aged = np.where(cond)[0]
        idx_aged = idx_aged[~np.in1d(idx_aged, idx_middle)]

        # Our source image is the first middle  aged man
        source_idx = idx_middle[0]
        X_source = np.expand_dims(X[source_idx], 0).astype(np.float64)
        cv2.imwrite("../../figures/0000.png", X_source[0, ::-1, :, :].transpose(1,2,0))

        # Pick the first middle aged man and find its neighbors wrt list_col_labels features
        nearest = nn.NearestNeighbors(n_neighbors=10000, algorithm='ball_tree')
        X_neighb = df_attrs[list_col_labels].values
        nearest.fit(X_neighb)
        _, neighb_idx = nearest.kneighbors(X_neighb[source_idx].reshape(1, -1))
        # Flatten the array of index
        neighb_idx = np.ravel(neighb_idx)

        # Now look for the nearest neighbors that are middle aged men
        neighb_middle = np.in1d(neighb_idx, idx_middle[1:])
        neighb_middle = neighb_idx[neighb_middle]
        neighb_middle = neighb_middle[:min(nb_neighbors, neighb_middle.shape[0])]

        # Now look for the nearest neighbors that are senior men
        neighb_aged = np.in1d(neighb_idx, idx_aged[1:])
        neighb_aged = neighb_idx[neighb_aged]
        neighb_aged = neighb_aged[:min(nb_neighbors, neighb_aged.shape[0])]

        neighb_middle = np.sort(neighb_middle).tolist()  # sort for hdf5 indexing
        neighb_aged = np.sort(neighb_aged).tolist()  # sort for hdf5 indexing

        # Get the deep features, flatten and concatenate
        X_VGG_middle = [hf["data_VGG_%s" % l][neighb_middle].reshape(len(neighb_middle), -1) for l in range(3)]
        X_VGG_aged = [hf["data_VGG_%s" % l][neighb_aged].reshape(len(neighb_aged), -1) for l in range(3)]

        X_VGG_middle = np.concatenate(X_VGG_middle, 1)
        X_VGG_aged = np.concatenate(X_VGG_aged, 1)

    # Get the mean VGG feature for middle aged and senior men
    phi_middle = np.mean(X_VGG_middle, 0)
    phi_aged = np.mean(X_VGG_aged, 0)

    # Compute w that goes from middle aged to senior men
    w = phi_aged - phi_middle

    # Normalize (optional)
    if normalize_w:
        w = w / np.linalg.norm(w)

    # get tensor representations of our images as well as alpha and w
    base_image = K.variable(general_utils.preprocess_input(X_source.copy()))
    alpha = K.variable(alpha)
    w = K.variable(w)

    # this will contain our generated image
    generated_image = K.placeholder((1, 3, img_nrows, img_ncols))

    # combine the 2 images into a single Keras tensor
    input_tensor = K.concatenate([base_image,
                                  generated_image], axis=0)

    # build the VGG19 network with our 3 images as input
    base_model = VGG19(weights='imagenet', include_top=False)
    list_output = ["block3_conv1", "block4_conv1", "block5_conv1"]
    list_output = [base_model.get_layer(l).output for l in list_output]
    model = Model(input=base_model.input, output=list_output)
    print('Model loaded.')

    def reverse_mapping_loss(phi_x, phi_z, alpha, w):

        vec = phi_x + alpha * w - phi_z
        return K.sum(K.square(vec))

    def total_variation_loss(x):
        assert K.ndim(x) == 4
        if K.image_dim_ordering() == 'th':
            a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
            b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
        else:
            a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
            b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
        return K.sum(a + b)

    # combine these loss functions into a single scalar
    loss = K.variable(0.)
    # Compute features
    VGG_features = model(input_tensor)
    base_image_features = [K.expand_dims(K.flatten(f[0, :, :, :]), 0) for f in VGG_features]
    generated_image_features = [K.expand_dims(K.flatten(f[1, :, :, :]), 0) for f in VGG_features]

    phi_x = K.concatenate(base_image_features)
    phi_z = K.concatenate(generated_image_features)

    loss_reverse_mapping = reverse_mapping_loss(phi_x, phi_z, alpha, w)
    loss_TV = total_variation_loss(generated_image)

    loss += weight_reverse_mapping * loss_reverse_mapping
    loss += weight_total_variation * loss_TV

    # get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, generated_image)

    outputs = [loss, loss_reverse_mapping, loss_TV]
    if type(grads) in {list, tuple}:
        outputs += grads
    else:
        outputs.append(grads)

    f_outputs = K.function([generated_image], outputs)

    def eval_loss_and_grads(x):
        if K.image_dim_ordering() == 'th':
            x = x.reshape((1, 3, img_nrows, img_ncols))
        else:
            x = x.reshape((1, img_nrows, img_ncols, 3))
        outs = f_outputs([x])
        loss_value = outs[0]
        loss_reverse = outs[1]
        loss_TV = outs[2]
        if len(outs[3:]) == 1:
            grad_values = outs[3].flatten().astype('float64')
        else:
            grad_values = np.array(outs[3:]).flatten().astype('float64')
        return loss_value, loss_reverse, loss_TV, grad_values

    class Evaluator(object):
        def __init__(self):
            self.loss_value = None
            self.loss_TV = None
            self.grad_values = None
            self.grads_values = None

        def loss(self, x):
            assert self.loss_value is None
            loss_value, loss_reverse, loss_TV, grad_values = eval_loss_and_grads(x)
            self.loss_value = loss_value
            self.loss_reverse = loss_reverse
            self.loss_TV = loss_TV
            self.grad_values = grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            # self.loss_reverse = None
            # self.loss_TV = None
            self.grad_values = None
            return grad_values

    evaluator = Evaluator()

    # run scipy-based optimization (L-BFGS) over the pixels of the generated image
    # x = np.random.uniform(0, 255, (1, 3, img_nrows, img_ncols)) - 128. # random initialization
    x = general_utils.preprocess_input(X_source.copy())  # better results when initializing with source image

    for i in range(1000):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        loss_reverse, loss_TV = evaluator.loss_reverse * 1E-7, evaluator.loss_TV * 1E-7
        print('Current loss value (x1E-7):', min_val * 1E-7, loss_reverse, loss_TV)
        # save current generated image
        img = general_utils.color_correction(x, img_nrows, img_ncols, X_source)

        fname = '../../figures/at_iteration_%d.png' % i
        if i % 5 == 0:
            imsave(fname, img)
            print('Image saved as', fname)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))


if __name__ == '__main__':
    launch_dfi()
