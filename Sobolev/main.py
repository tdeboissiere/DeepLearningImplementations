import argparse
import sobolev_training

# Training settings
parser = argparse.ArgumentParser(description='Sobolev experiments')

# Training params
parser.add_argument('--nb_epoch', default=100, type=int, help="Number of training epochs")
parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
parser.add_argument('--npts', default=20, type=int, help="Number of training points")
parser.add_argument('--learning_rate', default=1E-4, type=float, help="Learning rate")
parser.add_argument('--sobolev_weight', default=1, type=float, help="How much do we weight the Sobolev function")
args = parser.parse_args()


sobolev_training.launch_experiments(args)
