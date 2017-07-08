import numpy as np
from tqdm import tqdm
import lasagne
import theano


def create_dataset(npts):
    """ Sample data uniformly in a [-5,5] x [-5, 5] window"""

    # Create data
    np.random.seed(20)  # set seed for reproducibility
    X = np.random.uniform(-5,5, (npts, 2)).astype(np.float32)

    return X


def get_list_batches(npts, batch_size):
    """Create batches (i.e a list of index) such that an array of size npts
    is split in batches of size batch_size"""

    num_elem = npts
    num_batches = num_elem / batch_size
    # list_batches is a list of array. Each array contains the indeces of the batch elements.
    list_batches = np.array_split(np.arange(num_elem), num_batches)

    return list_batches


def train_network(train_fn, X, list_batches, nb_epoch):

    # Store train loss for each epoch
    list_loss = []

    for epoch in tqdm(range(nb_epoch), desc="Training normally"):

        # Store train loss for each batch
        epoch_losses = []

        # Loop over batches
        for batch_idxs in list_batches:
            X_batch = X[batch_idxs]
            epoch_losses.append(train_fn(X_batch))

        list_loss.append(np.mean(epoch_losses))

    return list_loss


def train_network_sobolev(train_fn, X, list_batches, nb_epoch):

    # Store train loss for each epoch
    list_loss = []
    list_loss_J = []

    for epoch in tqdm(range(nb_epoch), desc="Training with Sobolev"):

        epoch_losses = []
        epoch_losses_J = []

        for batch_idxs in list_batches:
            X_batch = X[batch_idxs]
            loss, J_loss = train_fn(X_batch)

            epoch_losses.append(loss)
            epoch_losses_J.append(J_loss)

        list_loss.append(np.mean(epoch_losses))
        list_loss_J.append(np.mean(epoch_losses_J))

    return list_loss, list_loss_J


def get_prediction_fn(input_var, network):

    # Create a prediction function
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_fn = theano.function([input_var], test_prediction)

    return predict_fn
