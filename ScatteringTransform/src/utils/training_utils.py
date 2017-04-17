import tensorflow as tf
import random
import numpy as np
import sys
sys.path.append("../utils")
import logging_utils as lu


def setup_session():

    lu.print_session()

    FLAGS = tf.app.flags.FLAGS

    # Create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if FLAGS.use_XLA:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)

    # Setup directory to save model
    for d in [FLAGS.log_dir, FLAGS.model_dir, FLAGS.fig_dir]:
        # Clear directories by default
        if tf.gfile.Exists(d):
            tf.gfile.DeleteRecursively(d)
        tf.gfile.MakeDirs(d)

    # Initialize all RNGs with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)

    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    return sess


def initialize_session(sess):

    saver = tf.train.Saver()

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


def manage_queues(sess):

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)

    lu.print_queues()

    return coord


def manage_summaries(sess):

    FLAGS = tf.app.flags.FLAGS
    writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    lu.print_summaries()

    return writer


def check_data(out, list_data):

    lu.print_check_data(out, list_data)
