import os
import cv2
import h5py
import glob
import parmap
import argparse
import numpy as np
from tqdm import tqdm as tqdm
import matplotlib.pylab as plt


def format_image(img_path, size):
    """
    Load img with opencv and reshape
    """

    img_color = cv2.imread(img_path)
    img_color = img_color[:, :, ::-1]

    # Slice image to center around face
    img_color = img_color[30:-30, 20:-20, :]

    img_color = cv2.resize(img_color, (size, size), interpolation=cv2.INTER_AREA)

    img_color = img_color.reshape((1, size, size, 3)).transpose(0, 3, 1, 2)

    return img_color


def build_HDF5(jpeg_dir, size=64):
    """
    Gather the data in a single HDF5 file.
    """

    # Put train data in HDF5
    hdf5_file = os.path.join(data_dir, "CelebA_%s_data.h5" % size)
    with h5py.File(hdf5_file, "w") as hfw:

            list_img = glob.glob(os.path.join(jpeg_dir, "*.jpg"))
            list_img = np.array(list_img)

            data_color = hfw.create_dataset("data",
                                            (0, 3, size, size),
                                            maxshape=(None, 3, size, size),
                                            dtype=np.uint8)

            num_files = len(list_img)
            chunk_size = 2000
            num_chunks = num_files / chunk_size
            arr_chunks = np.array_split(np.arange(num_files), num_chunks)

            for chunk_idx in tqdm(arr_chunks):

                list_img_path = list_img[chunk_idx].tolist()
                output = parmap.map(format_image, list_img_path, size, parallel=True)

                arr_img_color = np.concatenate(output, axis=0)

                # Resize HDF5 dataset
                data_color.resize(data_color.shape[0] + arr_img_color.shape[0], axis=0)

                data_color[-arr_img_color.shape[0]:] = arr_img_color.astype(np.uint8)


def check_HDF5(size=64):
    """
    Plot images with landmarks to check the processing
    """

    # Get hdf5 file
    hdf5_file = os.path.join(data_dir, "CelebA_%s_data.h5" % size)

    with h5py.File(hdf5_file, "r") as hf:
        data_color = hf["data"]
        for i in range(data_color.shape[0]):
            plt.figure()
            img = data_color[i, :, :, :].transpose(1,2,0)
            plt.imshow(img)
            plt.show()
            plt.clf()
            plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build dataset')
    parser.add_argument('jpeg_dir', type=str, help='path to celeba jpeg images')
    parser.add_argument('--img_size', default=64, type=int,
                        help='Desired Width == Height')
    parser.add_argument('--do_plot', default=False, type=bool,
                        help='Plot the images to make sure the data processing went OK')
    args = parser.parse_args()

    data_dir = "../../data/processed"

    build_HDF5(args.jpeg_dir, size=args.img_size)

    if args.do_plot:
        check_HDF5(args.img_size)
