# Training and evaluating

## Training

`python main.py`


positional arguments:

    keras_model_path      Path to keras deep-learning-models directory

optional arguments:

    -h, --help            show this help message and exit
    --data_file DATA_FILE
                        Path to HDF5 containing the data
    --attributes_file ATTRIBUTES_FILE
                        Path to csv file containing the attributes
    --nb_neighbors NB_NEIGHBORS
                        Number of nearest neighbors to compute VGG
                        representation
    --alpha ALPHA         Interpolation coefficient
    --weight_reverse_mapping WEIGHT_REVERSE_MAPPING
                        Weight of reverse mapping loss
    --weight_total_variation WEIGHT_TOTAL_VARIATION
                        Weight of total variation loss
    --normalize_w NORMALIZE_W
                        Whether to normalize w

**Example:**

`python main.py /home/user/GitHub/deep-learning-models`

### Expected outputs:

- Create a copy of the source image in `figures`
- Saves the generated image every few iteration in `figures`

