# Training GAN

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
    --noise_dim NOISE_DIM
                          Noise dimension for GAN generation
    --random_seed RANDOM_SEED
                          Seed used to initialize rng.
    --use_XLA [USE_XLA]   Whether to use XLA compiler.
    --nouse_XLA
    --num_threads NUM_THREADS
                          Number of threads to fetch the data
    --capacity_factor CAPACITY_FACTOR
                          Nuumber of batches to store in queue
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