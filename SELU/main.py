from __future__ import print_function
import os
import argparse
import torchvision.datasets as dset
import train

# Training settings
parser = argparse.ArgumentParser(description='MNIST SELU experiments')

# Neural Net archi
parser.add_argument('--model', default="RELUNet", type=str, help="Model name, RELUNet or SELUNet")
parser.add_argument('--n_inner_layers', default=4, type=int, help="Number of inner hidden layers")
parser.add_argument('--hidden_dim', default=-1, type=int, help="Hidden layer dimension")
parser.add_argument('--dropout', default=0, type=float, help="Dropout rate")
# Training params
parser.add_argument('--use_cuda', action="store_true", help="Use CUDA")
parser.add_argument('--nb_epoch', default=100, type=int, help="Number of training epochs")
parser.add_argument('--batchnorm', action="store_true", help="Whether to use BN for RELUNet")
parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
parser.add_argument('--optimizer', default="SGD", type=str, help="Optimizer")
parser.add_argument('--learning_rate', default=1E-5, type=float, help="Learning rate")
args = parser.parse_args()


assert args.model in ["RELUNet", "SELUNet"]

# Download mnist if it does not exist
if not os.path.isfile("processed/training.pt"):
    dset.MNIST(root=".", download=True)

if not os.path.exists("results"):
    os.makedirs("results")

# Launch training
train.train(args)
