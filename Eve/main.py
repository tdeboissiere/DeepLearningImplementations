import os
import train
import argparse


def launch_training(model_name, **kwargs):

    # Launch training
    train.train(model_name, **d_params)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experiments for Eve')
    parser.add_argument('list_experiments', type=str, nargs='+',
                        help='List of experiment names. E.g. Eve SGD Adam --> will run a training session with each optimizer')
    parser.add_argument('--model_name', default='CNN', type=str,
                        help='Model name: CNN, Big_CNN or FCN')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--nb_epoch', default=30, type=int,
                        help='Number of epochs')
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help='Dataset, cifar10, cifar100 or mnist')

    args = parser.parse_args()

    list_dir = ["figures", "log"]
    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    for experiment_name in args.list_experiments:
        optimizer = experiment_name.split("_")[0]
        assert optimizer in ["Eve", "Adam", "SGD"], "Invalid optimizer"
        assert args.model_name in ["CNN", "Big_CNN", "FCN"], "Invalid model name"

        # Set default params
        d_params = {"optimizer": optimizer,
                    "experiment_name": experiment_name,
                    "batch_size": args.batch_size,
                    "nb_epoch": args.nb_epoch,
                    "dataset": args.dataset
                    }

        # Launch training
        launch_training(args.model_name, **d_params)
