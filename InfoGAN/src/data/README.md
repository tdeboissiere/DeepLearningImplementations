# Building the data

# Step 1. Download CelebA

- Go to http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- In the Downloads section, select Align&Cropped images.
- In the dropbox page that follows, download the Anno, Eval and Img folders.
- Copy these folders to `Colorful/data/raw`.
- Extract the zip files.

You should have the following folder structure:

    ├── Anno
        ├── list_attr_celeba.txt  
        ├── list_bbox_celeba.txt  
        ├── list_landmarks_align_celeba.txt  
        ├── list_landmarks_celeba.txt
    ├── Eval
        ├──list_eval_partition.txt
    ├── img_align_celeba
        ├──lots of images

At this stage, annotations are not used.


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

`python make_dataset.py --img_size 64`