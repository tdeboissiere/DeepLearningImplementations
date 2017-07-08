import numpy as np
import json
import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as dset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import models


def train(args):
    """Train NN for eye tracking regression

    Args:
        args: CLI arguments
    """

    # Setup experiment ID
    exp_ID = "%s_depth_%s_opt_%s_drop_%s_bn_%s" % (args.model,
                                                   args.n_inner_layers + 2,
                                                   args.optimizer,
                                                   args.dropout,
                                                   args.batchnorm)

    ###########################
    # Data and normalization
    ###########################

    # Get data
    train_set = dset.MNIST(root=".", train=True)
    test_set = dset.MNIST(root=".", train=False)

    train_data = train_set.train_data.numpy().astype(np.float32)
    test_data = test_set.test_data.numpy().astype(np.float32)

    # Flatten
    train_data = np.reshape(train_data, (-1, 784))
    test_data = np.reshape(test_data, (-1, 784))

    # Stack
    data = np.concatenate((train_data, test_data), axis=0)
    # Scaler
    scaler = StandardScaler()
    scaler.fit(data)

    # Normalize data
    train_data = scaler.transform(train_data).astype(np.float32)
    test_data = scaler.transform(test_data).astype(np.float32)

    train_target = train_set.train_labels.numpy()
    test_target = test_set.test_labels.numpy()

    ###########################
    # Neural Net
    ###########################

    if args.hidden_dim == -1:
        hidden_dim = 784

    if args.model == "RELUNet":
        model = models.RELUNet(args.n_inner_layers, 784, hidden_dim, 10, dropout=args.dropout, batchnorm=args.batchnorm)
    elif args.model == "SELUNet":
        model = models.SELUNet(args.n_inner_layers, 784, hidden_dim, 10, dropout=args.dropout)

    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    loss_fn = torch.nn.CrossEntropyLoss(size_average=True)

    if args.use_cuda:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    # Get a list of batch idxs
    n_samples = train_target.shape[0]
    num_batches = n_samples // args.batch_size
    list_batches = np.array_split(np.arange(n_samples), num_batches)

    # Initialize train loss to monitor
    train_loss = np.inf
    # Dict to save losses
    d_loss = {"train_loss": []}

    for e in tqdm(range(args.nb_epoch), desc="Training"):

        # List of train losses for current epoch
        list_train_loss = []

        for batch_idxs in tqdm(list_batches, desc="Epoch: %s, TR loss: %.2g" % (e, train_loss)):

            optimizer.zero_grad()

            # Start and end index
            start, end = batch_idxs[0], batch_idxs[-1]

            # Get the data
            X = train_data[start: end + 1]
            y = train_target[start: end + 1]

            # Wrap to tensor
            X = torch.FloatTensor(X)
            y = torch.LongTensor(y)

            if args.use_cuda:
                X = X.cuda()
                y = y.cuda()

            # Wrap to variable
            X = Variable(X)
            y = Variable(y)

            # Forward pass
            y_pred = model(X, training=True)
            loss = loss_fn(y_pred, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            list_train_loss.append(loss.cpu().data.numpy()[0])

        # Update train loss
        train_loss = np.mean(list_train_loss)
        d_loss["train_loss"].append(float(train_loss))

        # Save
        with open("results/%s.json" % exp_ID, "w") as f:
            json.dump(d_loss, f)
