import os
import sys
import random
import numpy as np
import tensorflow as tf
sys.path.append("../utils")
import logging_utils as lu

FLAGS = tf.app.flags.FLAGS


def setup_inference_session():

    assert FLAGS.run == "inference", "Wrong run flag"

    ##################
    # Create session
    ##################
    config = tf.ConfigProto()
    if FLAGS.use_XLA:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)
    # Print session parameters and flags
    lu.print_session("Inference")

    ###############################################
    # Initialize all RNGs with a deterministic seed
    ###############################################
    # with sess.graph.as_default():
    #     tf.set_random_seed(FLAGS.random_seed)

    # random.seed(FLAGS.random_seed)
    # np.random.seed(FLAGS.random_seed)

    ########################################
    # Check models directory, inference mode
    ########################################
    assert tf.gfile.Exists(FLAGS.model_dir), "Model directory (%s) does not exist" % FLAGS.model_dir
    list_files = tf.gfile.Glob(os.path.join(FLAGS.model_dir, "*"))
    assert len(list_files) > 0, "Model directory (%s) is empty" % FLAGS.model_dir

    return sess


def restore_session():

    # Get checkpoint
    checkpoint = tf.train.get_checkpoint_state(FLAGS.model_dir)
    checkpoint_path = checkpoint.model_checkpoint_path
    lu.print_checkpoint(checkpoint)

    # Retrieve the meta graph
    meta_graph = tf.train.import_meta_graph(checkpoint_path + '.meta')
    # Get default graph
    graph = tf.get_default_graph()
    lu.print_meta_graph(checkpoint_path + '.meta')

    # Setup sessions
    sess = setup_inference_session()
    # Restore from the checkpoint
    meta_graph.restore(sess, checkpoint_path)
    lu.print_restore()

    return sess, graph
