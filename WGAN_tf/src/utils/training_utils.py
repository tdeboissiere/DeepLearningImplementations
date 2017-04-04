import os
import sys
import random
import numpy as np
import tensorflow as tf
sys.path.append("../utils")
import logging_utils as lu

FLAGS = tf.app.flags.FLAGS


def setup_training_session():

    assert FLAGS.run == "train", "Wrong run flag"

    ##################
    # Create session
    ##################
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if FLAGS.use_XLA:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)
    # Print session parameters and flags
    lu.print_session("Training")

    ###############################################
    # Initialize all RNGs with a deterministic seed
    ###############################################
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)

    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    ##################################
    # Setup directories, training mode
    ##################################
    list_delete = []  # keep track of directory creation/deletion
    list_create = []  # keep track of directory creation/deletion
    for d in [FLAGS.log_dir, FLAGS.model_dir, FLAGS.fig_dir]:
        # Delete directory and its contents if it exists
        if tf.gfile.Exists(d):
            tf.gfile.DeleteRecursively(d)
            list_delete.append(d)
        # Recreate directory
        tf.gfile.MakeDirs(d)
        list_create.append(d)
    # Print directory creation / deletion
    lu.print_directories(list_delete, list_create)

    return sess


def initialize_session(sess):

    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)

    lu.print_initialize()

    return saver


def add_gradient_summary(list_gradvar):
    # Add summary for gradients
    for g,v in list_gradvar:
        if g is not None:
            tf.summary.histogram(v.name + "/gradient", g)


def manage_summaries(sess):

    writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    lu.print_summaries()

    return writer


def check_data(out, list_data):

    lu.print_check_data(out, list_data)
