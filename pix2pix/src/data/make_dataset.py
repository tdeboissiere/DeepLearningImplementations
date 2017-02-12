import os
import cv2
import h5py
import parmap
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm as tqdm
import matplotlib.pylab as plt


def format_image(img_path, size, nb_channels):
    """
    Load img with opencv and reshape
    """

    if nb_channels == 1:
        img = cv2.imread(img_path, 0)
        img = np.expand_dims(img, axis=-1)
    else:
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]  # GBR to RGB

    w = img.shape[1]

    # Slice image in 2 to get both parts
    img_full = img[:, :w / 2, :]
    img_sketch = img[:, w / 2:, :]

    img_full = cv2.resize(img_full, (size, size), interpolation=cv2.INTER_AREA)
    img_sketch = cv2.resize(img_sketch, (size, size), interpolation=cv2.INTER_AREA)

    if nb_channels == 1:
        img_full = np.expand_dims(img_full, -1)
        img_sketch = np.expand_dims(img_sketch, -1)

    img_full = np.expand_dims(img_full, 0).transpose(0, 3, 1, 2)
    img_sketch = np.expand_dims(img_sketch, 0).transpose(0, 3, 1, 2)

    return img_full, img_sketch


def build_HDF5(jpeg_dir, nb_channels, size=256):
    """
    Gather the data in a single HDF5 file.
    """

    # Put train data in HDF5
    file_name = os.path.basename(jpeg_dir.rstrip("/"))
    hdf5_file = os.path.join(data_dir, "%s_data.h5" % file_name)
    with h5py.File(hdf5_file, "w") as hfw:

        for dset_type in ["train", "test", "val"]:

            list_img = list(Path(jpeg_dir).glob('%s/*.jpg' % dset_type))
            list_img.extend(list(Path(jpeg_dir).glob('%s/*.png' % dset_type)))
            list_img = map(str, list_img)
            list_img = np.array(list_img)

            data_full = hfw.create_dataset("%s_data_full" % dset_type,
                                           (0, nb_channels, size, size),
                                           maxshape=(None, 3, size, size),
                                           dtype=np.uint8)

            data_sketch = hfw.create_dataset("%s_data_sketch" % dset_type,
                                             (0, nb_channels, size, size),
                                             maxshape=(None, 3, size, size),
                                             dtype=np.uint8)

            num_files = len(list_img)
            chunk_size = 100
            num_chunks = num_files / chunk_size
            arr_chunks = np.array_split(np.arange(num_files), num_chunks)

            for chunk_idx in tqdm(arr_chunks):

                list_img_path = list_img[chunk_idx].tolist()
                output = parmap.map(format_image, list_img_path, size, nb_channels, parallel=False)

                arr_img_full = np.concatenate([o[0] for o in output], axis=0)
                arr_img_sketch = np.concatenate([o[1] for o in output], axis=0)

                # Resize HDF5 dataset
                data_full.resize(data_full.shape[0] + arr_img_full.shape[0], axis=0)
                data_sketch.resize(data_sketch.shape[0] + arr_img_sketch.shape[0], axis=0)

                data_full[-arr_img_full.shape[0]:] = arr_img_full.astype(np.uint8)
                data_sketch[-arr_img_sketch.shape[0]:] = arr_img_sketch.astype(np.uint8)

def check_HDF5(jpeg_dir, nb_channels):
    """
    Plot images with landmarks to check the processing
    """

    # Get hdf5 file
    file_name = os.path.basename(jpeg_dir.rstrip("/"))
    hdf5_file = os.path.join(data_dir, "%s_data.h5" % file_name)

    with h5py.File(hdf5_file, "r") as hf:
        data_full = hf["train_data_full"]
        data_sketch = hf["train_data_sketch"]
        for i in range(data_full.shape[0]):
            plt.figure()
            img = data_full[i, :, :, :].transpose(1,2,0)
            img2 = data_sketch[i, :, :, :].transpose(1,2,0)
            img = np.concatenate((img, img2), axis=1)
            if nb_channels == 1:
                plt.imshow(img[:, :, 0], cmap="gray")
            else:
                plt.imshow(img)
            plt.show()
            plt.clf()
            plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build dataset')
    parser.add_argument('jpeg_dir', type=str, help='path to jpeg images')
    parser.add_argument('nb_channels', type=int, help='number of image channels')
    parser.add_argument('--img_size', default=256, type=int,
                        help='Desired Width == Height')
    parser.add_argument('--do_plot', action="store_true",
                        help='Plot the images to make sure the data processing went OK')
    args = parser.parse_args()

    data_dir = "../../data/processed"

    build_HDF5(args.jpeg_dir, args.nb_channels, size=args.img_size)

    if args.do_plot:
        check_HDF5(args.jpeg_dir, args.nb_channels)
