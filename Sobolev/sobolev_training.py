import lasagne
import theano
import theano.tensor as T
from lasagne.nonlinearities import leaky_rectify, linear
import utils
import plot_results


def create_student_model(input_var):

    # create a small convolutional neural network
    network = lasagne.layers.InputLayer((None, 2), input_var)
    network = lasagne.layers.DenseLayer(network, 256, nonlinearity=leaky_rectify)
    network = lasagne.layers.DenseLayer(network, 256, nonlinearity=leaky_rectify)
    network = lasagne.layers.DenseLayer(network, 1, nonlinearity=linear)

    return network


def standard_train(args, input_var, network, tang_output):
    """Train a network normally, output the network, a function to make predictions and training losses"""

    # Create data
    X = utils.create_dataset(args.npts)

    # Create a list of batches (a list of batch idxs splitting X in batches of size batch_size)
    list_batches = utils.get_list_batches(args.npts, args.batch_size)

    # create loss function
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, tang_output)
    loss = loss.mean()

    # create parameter update expressions
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=args.learning_rate, momentum=0.9)

    # compile training function that updates parameters and returns training loss
    train_fn = theano.function([input_var], loss, updates=updates)

    # train network
    list_loss = utils.train_network(train_fn, X, list_batches, args.nb_epoch)

    # Create a prediction function to evaluate after training
    predict_fn = utils.get_prediction_fn(input_var, network)

    return network, predict_fn, list_loss


def sobolev_train(args, input_var, network, tang_output):
    """Train a network with Sobolev, output the network, a function to make predictions and training losses"""

    # Create data
    X = utils.create_dataset(args.npts)

    # Create a list of batches (a list of batch idxs splitting X in batches of size batch_size)
    list_batches = utils.get_list_batches(args.npts, args.batch_size)

    # create loss function
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, tang_output)
    loss = loss.mean()

    # Add jacobian (J) output for Sobolev training
    J_teacher = theano.gradient.jacobian(tang_output.flatten(), input_var)
    J_student = theano.gradient.jacobian(prediction.flatten(), input_var)

    loss_sobolev = lasagne.objectives.squared_error(J_teacher.flatten(), J_student.flatten())
    loss_sobolev = args.sobolev_weight * loss_sobolev.mean()

    loss_total = loss + loss_sobolev

    # create parameter update expressions
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss_total, params, learning_rate=args.learning_rate, momentum=0.9)

    # compile training function that updates parameters and returns training loss
    train_fn = theano.function([input_var], [loss, loss_sobolev], updates=updates)

    # train network
    list_loss, list_loss_J = utils.train_network_sobolev(train_fn, X, list_batches, args.nb_epoch)

    # Create a prediction function to evaluate after training
    predict_fn = utils.get_prediction_fn(input_var, network)

    return network, predict_fn, list_loss, list_loss_J


def launch_experiments(args):

    # create Theano variables for input and target minibatch
    input_var = T.fmatrix('X')

    # Create tang function
    # need abs to avoid nan with pow on the GPU
    f0 = 0.5 * (T.pow(abs(input_var), 4) - 16 * T.pow(abs(input_var), 2) + 5 * input_var)
    tang_output = f0.sum(axis=-1)

    # Create and train network normally
    network = create_student_model(input_var)
    out = standard_train(args, input_var, network, tang_output)

    # Create and train network with Sobolev
    network_sobolev = create_student_model(input_var)
    out_sobolev = sobolev_train(args, input_var, network_sobolev, tang_output)

    # Now plot and compare outputs
    plot_results.plot_results(args, out, out_sobolev)
