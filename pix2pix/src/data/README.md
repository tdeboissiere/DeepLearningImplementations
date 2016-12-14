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
        ├── list_landmarks_celeba.txt


# Step 2. Build HDF5 CelebA dataset

`python make_dataset.py`

positional arguments:
  jpeg_dir             path to celeba jpeg images

optional arguments:
  -h, --help           show this help message and exit
  --img_size IMG_SIZE  Desired Width == Height
  --do_plot DO_PLOT    Plot the images to make sure the data processing went
                       OK


**Example:**

`python make_dataset.py --img_size 256 --do_plot True`