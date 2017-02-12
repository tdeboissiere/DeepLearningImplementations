# Building the data

# Step 1. Download facades dataset

- `git clone https://github.com/phillipi/pix2pix.git`
- `cd pix2pix`
- `bash ./datasets/download_dataset.sh facades`


You should have the following folder structure:

    ├── facades
        ├── train 
        ├── test 
        ├── val


# Step 2. Build HDF5 facades dataset

`python make_dataset.py`

positional arguments:

    jpeg_dir             path to jpeg images
    nb_channels          number of image channels

optional arguments:

    -h, --help           show this help message and exit
    --img_size IMG_SIZE  Desired Width == Height
    --do_plot            Plot the images to make sure the data processing went
                         OK



**Example:**

`python make_dataset.py /home/user/GitHub/pix2pix/datasets/facades 3 --img_size 256 --do_plot True`