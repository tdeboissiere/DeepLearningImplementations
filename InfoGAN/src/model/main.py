import os
import argparse


def launch_training(**kwargs):

    # Launch training
    train.train(**kwargs)


def launch_eval(**kwargs):

    # Launch training
    eval.eval(**kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('mode', type=str, help="train or eval")
    parser.add_argument('--backend', type=str, default="tensorflow", help="theano or tensorflow")
    parser.add_argument('--generator', type=str, default="upsampling", help="upsampling or deconv")
    parser.add_argument('--dset', type=str, default="mnist", help="mnist or celebA")
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--n_batch_per_epoch', default=2000, type=int, help="Number of training epochs")
    parser.add_argument('--nb_epoch', default=400, type=int, help="Number of batches per epoch")
    parser.add_argument('--epoch', default=10, type=int, help="Epoch at which weights were saved for evaluation")
    parser.add_argument('--nb_classes', default=2, type=int, help="Number of classes")
    parser.add_argument('--do_plot', default=False, type=bool, help="Debugging plot")
    parser.add_argument('--bn_mode', default=2, type=int, help="Batch norm mode")
    parser.add_argument('--img_dim', default=64, type=int, help="Image width == height (only specify for CelebA)")
    parser.add_argument('--noise_dim', default=64, type=int, help="noise dimension")
    parser.add_argument('--cont_dim', default=2, type=int, help="Latent continuous dimensions")
    parser.add_argument('--cat_dim', default=10, type=int, help="Latent categorical dimension")
    parser.add_argument('--noise_scale', default=0.5, type=float,
                        help="variance of the normal from which we sample the noise")
    parser.add_argument('--label_smoothing', action="store_true", help="smooth the positive labels when training D")
    parser.add_argument('--use_mbd', action="store_true", help="use mini batch disc")
    parser.add_argument('--label_flipping', default=0, type=float,
                        help="Probability (0 to 1.) to flip the labels when training D")

    args = parser.parse_args()

    assert args.mode in ["train", "eval"]
    assert args.dset in ["mnist", "celebA"]

    # Set the backend by modifying the env variable
    if args.backend == "theano":
        os.environ["KERAS_BACKEND"] = "theano"
    elif args.backend == "tensorflow":
        os.environ["KERAS_BACKEND"] = "tensorflow"

    # Import the backend
    import keras.backend as K

    # manually set dim ordering otherwise it is not changed
    if args.backend == "theano":
        image_dim_ordering = "th"
        K.set_image_dim_ordering(image_dim_ordering)
    elif args.backend == "tensorflow":
        image_dim_ordering = "tf"
        K.set_image_dim_ordering(image_dim_ordering)

    import train
    import eval

    # Set default params
    d_params = {"dset": args.dset,
                "generator": args.generator,
                "batch_size": args.batch_size,
                "n_batch_per_epoch": args.n_batch_per_epoch,
                "nb_epoch": args.nb_epoch,
                "model_name": "InfoGAN",
                "epoch": args.epoch,
                "nb_classes": args.nb_classes,
                "do_plot": args.do_plot,
                "image_dim_ordering": image_dim_ordering,
                "bn_mode": args.bn_mode,
                "img_dim": args.img_dim,
                "noise_dim": args.noise_dim,
                "cat_dim": args.cat_dim,
                "cont_dim": args.cont_dim,
                "label_smoothing": args.label_smoothing,
                "label_flipping": args.label_flipping,
                "noise_scale": args.noise_scale,
                "use_mbd": args.use_mbd,
                }

    if args.mode == "train":
        # Launch training
        launch_training(**d_params)

    if args.mode == "eval":
        # Launch eval
        launch_eval(**d_params)
