import argparse
import dfi


def launch_dfi(**kwargs):

    # Launch training
    dfi.launch_dfi(**kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument("keras_model_path", type=str, help="Path to keras deep-learning-models directory")
    parser.add_argument('--data_file', type=str, default="../../data/processed/lfw_200_data.h5",
                        help="Path to HDF5 containing the data")
    parser.add_argument('--attributes_file', type=str, default="../../data/processed/lfw_processed_attributes.csv",
                        help="Path to csv file containing the attributes")
    parser.add_argument('--nb_neighbors', default=100, type=int,
                        help="Number of nearest neighbors to compute VGG representation")
    parser.add_argument('--alpha', default=4, type=float, help="Interpolation coefficient")
    parser.add_argument('--weight_reverse_mapping', default=1., type=float, help="Weight of reverse mapping loss")
    parser.add_argument('--weight_total_variation', default=1E3, type=float, help="Weight of total variation loss")
    parser.add_argument('--normalize_w', default=False, type=bool, help="Whether to normalize w")

    args = parser.parse_args()

    # Set default params
    d_params = {"keras_model_path": args.keras_model_path,
                "data_file": args.data_file,
                "attributes_file": args.attributes_file,
                "nb_neighbors": args.nb_neighbors,
                "alpha": args.alpha,
                "weight_total_variation":args.weight_total_variation,
                "weight_reverse_mapping": args.weight_reverse_mapping,
                "normalize_w": args.normalize_w
                }

    launch_dfi(**d_params)
