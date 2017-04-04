
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def define_flags():

    ############
    # Run mode
    ############
    # INFERENCE TODO
    tf.app.flags.DEFINE_string('run', None, "Which operation to run. [train|inference]")

    ##########################
    # Training parameters
    ###########################
    tf.app.flags.DEFINE_integer('nb_epoch', 500, "Number of epochs")
    tf.app.flags.DEFINE_integer('batch_size', 16, "Number of samples per batch.")
    tf.app.flags.DEFINE_integer('nb_batch_per_epoch', 100, "Number of batches per epoch")
    tf.app.flags.DEFINE_float('learning_rate', 1E-4, "Learning rate used for AdamOptimizer")
    tf.app.flags.DEFINE_integer('h_dim', 128, "AutoEncoder internal embedding dimension")
    tf.app.flags.DEFINE_integer('z_dim', 128, "Noise distribution dimension dimension")
    tf.app.flags.DEFINE_integer('nb_filters_D', 64, "Number of conv filters for D")
    tf.app.flags.DEFINE_integer('nb_filters_G', 64, "Number of conv filters for G")
    tf.app.flags.DEFINE_integer('random_seed', 0, "Seed used to initialize rng.")
    tf.app.flags.DEFINE_integer('max_to_keep', 500, "Maximum number of model/session files to keep")
    tf.app.flags.DEFINE_float('gamma', 0.5, "")
    tf.app.flags.DEFINE_float('lambdak', 1E-3, "Proportional gain for k")

    ############################################
    # General tensorflow parameters parameters
    #############################################
    tf.app.flags.DEFINE_bool('use_XLA', False, "Whether to use XLA compiler.")
    tf.app.flags.DEFINE_integer('num_threads', 2, "Number of threads to fetch the data")
    tf.app.flags.DEFINE_float('capacity_factor', 32, "Number of batches to store in queue")

    ##########
    # Datasets
    ##########
    tf.app.flags.DEFINE_string('data_format', "NCHW", "Tensorflow image data format.")
    tf.app.flags.DEFINE_string('celebA_path', "../../data/raw/img_align_celeba", "Path to celebA images")
    tf.app.flags.DEFINE_integer('channels', 3, "Number of channels")
    tf.app.flags.DEFINE_float('central_fraction', 0.6, "Central crop as a fraction of total image")
    tf.app.flags.DEFINE_integer('img_size', 64, "Image size")

    ##############
    # Directories
    ##############
    tf.app.flags.DEFINE_string('model_dir', '../../models', "Output folder where checkpoints are dumped.")
    tf.app.flags.DEFINE_string('log_dir', '../../logs', "Logs for tensorboard.")
    tf.app.flags.DEFINE_string('fig_dir', '../../figures', "Where to save figures.")
    tf.app.flags.DEFINE_string('raw_dir', '../../data/raw', "Where raw data is saved")
    tf.app.flags.DEFINE_string('data_dir', '../../data/processed', "Where processed data is saved")
