# Training and evaluating

## Training

`python main.py`

positional arguments:

    mode                  Choose train or eval
    data_file             Path to HDF5 containing the data

optional arguments:

    -h, --help            show this help message and exit
    --model_name MODEL_NAME
                        Model name. Choose simple_colorful or colorful
    --training_mode TRAINING_MODE
                        Training mode. Choose in_memory to load all the data
                        in memory and train.Choose on_demand to load batches
                        from disk at each step
    --batch_size BATCH_SIZE
                        Batch size
    --n_batch_per_epoch N_BATCH_PER_EPOCH
                        Number of training epochs
    --nb_epoch NB_EPOCH   Number of batches per epoch
    --nb_resblocks NB_RESBLOCKS
                        Number of residual blocks for simple model
    --nb_neighbors NB_NEIGHBORS
                        Number of nearest neighbors for soft encoding
    --epoch EPOCH         Epoch at which weights were saved for evaluation
    --T T                 Temperature to change color balance in evaluation
                        phase.If T = 1: desaturated. If T~0 vivid

**Example:**

`python main.py train ../../data/processed/CelebA_32_data.h5`

### Expected outputs:

- Create a directory in Colorful/models where weights are saved
- Create a directory in Colorful/figures where figures are saved
- Plot examples of the colorization results at each epoch
- Plot model architecture.
- Save model weights every few epochs


## Evaluating

positional arguments:

    mode                  Choose train or eval
    data_file             Path to HDF5 containing the data

optional arguments:

    -h, --help            show this help message and exit
    --model_name MODEL_NAME
                        Model name. Choose simple_colorful or colorful
    --training_mode TRAINING_MODE
                        Training mode. Choose in_memory to load all the data
                        in memory and train.Choose on_demand to load batches
                        from disk at each step
    --batch_size BATCH_SIZE
                        Batch size
    --n_batch_per_epoch N_BATCH_PER_EPOCH
                        Number of training epochs
    --nb_epoch NB_EPOCH   Number of batches per epoch
    --nb_resblocks NB_RESBLOCKS
                        Number of residual blocks for simple model
    --nb_neighbors NB_NEIGHBORS
                        Number of nearest neighbors for soft encoding
    --epoch EPOCH         Epoch at which weights were saved for evaluation
    --T T                 Temperature to change color balance in evaluation
                        phase.If T = 1: desaturated. If T~0 vivid

**Example:**

`python main.py eval ../../data/processed/CelebA_64_data.h5 --epoch 10`

### Expected outputs:

Randomly sample images from the validation set and plot the color, colorized and b&w version.