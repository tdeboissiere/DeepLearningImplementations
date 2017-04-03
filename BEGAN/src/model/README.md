# Training WGAN

## Usage

`python main.py`


optional arguments:

    --run RUN             Which operation to run. [train|inference]
    --nb_epoch NB_EPOCH   Number of epochs
    --batch_size BATCH_SIZE
                          Number of samples per batch.
    --nb_batch_per_epoch NB_BATCH_PER_EPOCH
                          Number of batches per epoch
    --learning_rate LEARNING_RATE
                          Learning rate used for AdamOptimizer
    --h_dim H_DIM         AutoEncoder internal embedding dimension
    --z_dim Z_DIM         Noise distribution dimension dimension
    --nb_filters_D NB_FILTERS_D
                      Number of conv filters for D
    --nb_filters_G NB_FILTERS_G
                      Number of conv filters for G

    --random_seed RANDOM_SEED
                          Seed used to initialize rng.
    --max_to_keep MAX_TO_KEEP
                          Maximum number of model/session files to keep
    --gamma GAMMA
    --lambdak LAMBDAK     Proportional gain for k
    --use_XLA [USE_XLA]   Whether to use XLA compiler.
    --num_threads NUM_THREADS
                          Number of threads to fetch the data
    --capacity_factor CAPACITY_FACTOR
                          Number of batches to store in queue
    --data_format DATA_FORMAT
                          Tensorflow image data format.
    --celebA_path CELEBA_PATH
                          Path to celebA images
    --channels CHANNELS   Number of channels
    --central_fraction CENTRAL_FRACTION
                          Central crop as a fraction of total image
    --img_size IMG_SIZE   Image size
    --model_dir MODEL_DIR
                          Output folder where checkpoints are dumped.
    --log_dir LOG_DIR     Logs for tensorboard.
    --fig_dir FIG_DIR     Where to save figures.
    --raw_dir RAW_DIR     Where raw data is saved
    --data_dir DATA_DIR   Where processed data is saved





**Example:**

    python main.py --run train


## Results:

Figures are saved in the `figures` folder. They show real images, fake images and reconstruction. One set of images is saved at each epoch.

Call:

    tensorboard --logdir = /path/to/logs

for visualization.

## Implementation notes:

- Downsampling in D is done with strided convolutions
- The current implementation is designed for 64x64 images.
- Previous models / logs / figures are wiped out at each invocation of the code. THis behaviour can be modified in `src/utils/training_utils/setup_training_session()`