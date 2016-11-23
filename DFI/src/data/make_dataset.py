import os
import cv2
import sys
import h5py
import parmap
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
import matplotlib.pylab as plt


def format_image(img_path, size):
    """
    Load img with opencv and reshape
    """

    img_color = cv2.imread(img_path)
    img_color = img_color[20:220, 20:220, :]  # crop to center around face (empirical values)

    img_color = cv2.resize(img_color, (size, size), interpolation=cv2.INTER_AREA)
    img_color = img_color.reshape((1, size, size, 3)).transpose(0, 3, 1, 2)
    img_color = img_color[:, ::-1, :, :]  # BGR to RGB

    return img_color


def parse_attibutes():

    attr_file = os.path.join(raw_dir, "lfw_attributes.txt")

    arr = []

    with open(attr_file, "r") as f:

        lines = f.readlines()
        list_col_names = lines[1].rstrip().split("\t")[1:]
        for l in lines[2:]:
            arr.append(l.rstrip().split("\t"))
        arr = np.array(arr)

        df = pd.DataFrame(arr, columns=list_col_names)
        col_float = df.columns.values[2:]
        for c in col_float:
            df[c] = df[c].astype(np.float32)
        df["imagenum"] = df.imagenum.apply(lambda x: x.zfill(4))
        df["person"] = df.person.apply(lambda x: "_".join(x.split(" ")))
        df["image_path"] = df.person + "/" + df.person + "_" + df.imagenum + ".jpg"
        df["image_path"] = df["image_path"].apply(lambda x: os.path.join(raw_dir, "lfw-deepfunneled", x))
        df.to_csv(os.path.join(data_dir, "lfw_processed_attributes.csv"), index=False)

        return df


def build_HDF5(size):
    """
    Gather the data in a single HDF5 file.
    """

    df_attr = parse_attibutes()
    list_col_labels = [c for c in df_attr.columns.values
                       if c not in ["person", "imagenum", "image_path"]]

    # Put train data in HDF5
    hdf5_file = os.path.join(data_dir, "lfw_%s_data.h5" % size)
    with h5py.File(hdf5_file, "w") as hfw:

        data_color = hfw.create_dataset("data",
                                        (0, 3, size, size),
                                        maxshape=(None, 3, size, size),
                                        dtype=np.uint8)

        label = hfw.create_dataset("labels", data=df_attr[list_col_labels].values)
        label.attrs["label_names"] = list_col_labels

        arr_img = df_attr.image_path.values

        num_files = len(arr_img)
        chunk_size = 1000
        num_chunks = num_files / chunk_size
        arr_chunks = np.array_split(np.arange(num_files), num_chunks)

        for chunk_idx in tqdm(arr_chunks):

            list_img_path = arr_img[chunk_idx].tolist()
            output = parmap.map(format_image, list_img_path, size, parallel=True)

            arr_img_color = np.concatenate(output, axis=0)

            # Resize HDF5 dataset
            data_color.resize(data_color.shape[0] + arr_img_color.shape[0], axis=0)

            data_color[-arr_img_color.shape[0]:] = arr_img_color.astype(np.uint8)


def compute_VGG19_features(keras_model_path, size, batch_size=32):

    sys.path.append(keras_model_path)
    from vgg19 import VGG19
    from imagenet_utils import preprocess_input
    from keras.models import Model

    # Load data
    hdf5_file = os.path.join(data_dir, "lfw_%s_data.h5" % size)
    with h5py.File(hdf5_file, "a") as hf:

        X = hf["data"][:].astype(np.float32)
        X = preprocess_input(X)

        base_model = VGG19(weights='imagenet', include_top=False)
        list_output = ["block3_conv1", "block4_conv1", "block5_conv1"]
        list_output = [base_model.get_layer(l).output for l in list_output]

        model = Model(input=base_model.input, output=list_output)
        vgg19_feat = model.predict(X, batch_size=batch_size, verbose=True)
        for i in range(len(vgg19_feat)):
            hf.create_dataset("data_VGG_%s" % str(i), data=vgg19_feat[i])


def check_HDF5(size):
    """
    Plot images with landmarks to check the processing
    """

    # Get hdf5 file
    hdf5_file = os.path.join(data_dir, "lfw_%s_data.h5" % size)

    with h5py.File(hdf5_file, "r") as hf:
        data_color = hf["data"]
        label = hf["labels"]
        attrs = label.attrs["label_names"]
        for i in range(data_color.shape[0]):
            plt.figure(figsize=(20, 10))
            img = data_color[i, :, :, :].transpose(1,2,0)[:, :, ::-1]
            # Get the 10 labels with highest values
            idx = label[i].argsort()[-10:]
            plt.xlabel(",  ".join(attrs[idx]), fontsize=12)
            plt.imshow(img)
            plt.show()
            plt.clf()
            plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build dataset')
    parser.add_argument("keras_model_path", type=str,
                        help="Path to keras deep-learning-models directory")
    parser.add_argument('--img_size', default=200, type=int,
                        help='Desired Width == Height')
    parser.add_argument('--do_plot', default=False, type=bool,
                        help='Whether to visualize saved images')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for VGG predictions')
    args = parser.parse_args()

    raw_dir = "../../data/raw"
    data_dir = "../../data/processed"

    for d in [raw_dir, data_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    build_HDF5(args.img_size)
    compute_VGG19_features(args.keras_model_path, args.img_size, args.batch_size)

    if args.do_plot:
        check_HDF5(args.img_size)
