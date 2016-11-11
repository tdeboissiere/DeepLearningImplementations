import argparse
import train_colorful
import eval_colorful


def launch_training(**kwargs):

    # Launch training
    train_colorful.train(**kwargs)


def launch_eval(**kwargs):

    # Launch training
    eval_colorful.eval(**kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('mode', type=str, help="Choose train or eval")
    parser.add_argument('data_file', type=str, help="Path to HDF5 containing the data")
    parser.add_argument('--model_name', type=str, default="simple_colorful",
                        help="Model name. Choose simple_colorful or colorful")
    parser.add_argument('--training_mode', default="in_memory", type=str,
                        help=('Training mode. Choose in_memory to load all the data in memory and train.'
                              'Choose on_demand to load batches from disk at each step'))
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--n_batch_per_epoch', default=100, type=int, help="Number of training epochs")
    parser.add_argument('--nb_epoch', default=400, type=int, help="Number of batches per epoch")
    parser.add_argument('--nb_resblocks', default=3, type=int, help="Number of residual blocks for simple model")
    parser.add_argument('--nb_neighbors', default=10, type=int, help="Number of nearest neighbors for soft encoding")
    parser.add_argument('--epoch', default=0, type=int, help="Epoch at which weights were saved for evaluation")
    parser.add_argument('--T', default=0.1, type=float,
                        help=("Temperature to change color balance in evaluation phase."
                              "If T = 1: desaturated. If T~0 vivid"))

    args = parser.parse_args()

    # Set default params
    d_params = {"data_file": args.data_file,
                "batch_size": args.batch_size,
                "n_batch_per_epoch": args.n_batch_per_epoch,
                "nb_epoch": args.nb_epoch,
                "nb_resblocks":args.nb_resblocks,
                "training_mode": args.training_mode,
                "model_name": args.model_name,
                "nb_neighbors": args.nb_neighbors,
                "epoch": args.epoch,
                "T": args.T
                }

    assert args.mode in ["train", "eval"]
    assert args.training_mode in ["in_memory", "on_demand"]
    assert args.model_name in ["colorful", "simple_colorful"]

    if args.mode == "train":
        # Launch training
        launch_training(**d_params)

    if args.mode == "eval":
        # Launch evaluation
        launch_eval(**d_params)
