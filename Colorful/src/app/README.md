# Colorizer app

## Usage

`python webcam_colorizer.py`

positional arguments:
    data_file             Path to HDF5 containing the data
    epoch                 Epoch of saved weights

optional arguments:

    -h, --help            show this help message and exit
    --model_name MODEL_NAME Model name. Choose simple_colorful or colorful
    --T T                 Temperature to change color balance. If T = 1: desaturated. If T~0 vivid
    --out_h OUT_H         Width of ouput image
    --out_w OUT_W         Height of ouput image
    --video_path VIDEO_PATH Path to B&W video to colorize
