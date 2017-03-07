import os
import sys
import models
import numpy as np
from tqdm import tqdm
import tensorflow as tf
sys.path.append("../utils")
import visualization_utils as vu
import training_utils as tu
import data_utils as du
import objectives

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS


def train_model():

    # Setup session
    sess = tu.setup_session()

    # Placeholder for data and Mnist iterator
    mnist = input_data.read_data_sets(FLAGS.raw_dir, one_hot=True)
    # images = tf.constant(mnist.train.images)
    if FLAGS.data_format == "NHWC":
        X_real = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 28, 28, 1])
    else:
        X_real = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 1, 28, 28])

    with tf.device('/cpu:0'):
        imgs = mnist.train.images.astype(np.float32)
        npts = imgs.shape[0]
        if FLAGS.data_format == "NHWC":
            imgs = imgs.reshape((npts, 28, 28, 1))
        else:
            imgs = imgs.reshape((npts, 1, 28, 28))
        imgs = (imgs - 0.5) / 0.5
        # input_images = tf.constant(imgs)

    #     image = tf.train.slice_input_producer([input_images], num_epochs=FLAGS.nb_epoch)
    #     X_real = tf.train.batch(image, batch_size=FLAGS.batch_size, num_threads=8)

    #######################
    # Instantiate generator
    #######################
    list_filters = [256, 1]
    list_strides = [2] * len(list_filters)
    list_kernel_size = [3] * len(list_filters)
    list_padding = ["SAME"] * len(list_filters)
    output_shape = X_real.get_shape().as_list()[1:]
    G = models.Generator(list_filters, list_kernel_size, list_strides, list_padding, output_shape,
                         batch_size=FLAGS.batch_size, dset="mnist", data_format=FLAGS.data_format)

    ###########################
    # Instantiate discriminator
    ###########################
    list_filters = [32, 64]
    list_strides = [2] * len(list_filters)
    list_kernel_size = [3] * len(list_filters)
    list_padding = ["SAME"] * len(list_filters)
    D = models.Discriminator(list_filters, list_kernel_size, list_strides, list_padding,
                             FLAGS.batch_size, data_format=FLAGS.data_format)

    ###########################
    # Instantiate optimizers
    ###########################
    G_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, name='G_opt', beta1=0.5)
    D_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, name='D_opt', beta1=0.5)

    ###########################
    # Instantiate model outputs
    ###########################

    # noise_input = tf.random_normal((FLAGS.batch_size, FLAGS.noise_dim,), stddev=0.1)
    noise_input = tf.random_uniform((FLAGS.batch_size, FLAGS.noise_dim,), minval=-1, maxval=1)
    X_fake = G(noise_input)

    # output images
    X_G_output = du.unnormalize_image(X_fake)
    X_real_output = du.unnormalize_image(X_real)

    D_real = D(X_real)
    D_fake = D(X_fake, reuse=True)

    ###########################
    # Instantiate losses
    ###########################

    G_loss = objectives.binary_cross_entropy_with_logits(D_fake, tf.ones_like(D_fake))
    D_loss_real = objectives.binary_cross_entropy_with_logits(D_real, tf.ones_like(D_real))
    D_loss_fake = objectives.binary_cross_entropy_with_logits(D_fake, tf.zeros_like(D_fake))

    D_loss = D_loss_real + D_loss_fake

    # ######################################################################
    # # Some parameters need to be updated (e.g. BN moving average/variance)
    # ######################################################################
    # from tensorflow.python.ops import control_flow_ops
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     barrier = tf.no_op(name='update_barrier')
    # D_loss = control_flow_ops.with_dependencies([barrier], D_loss)
    # G_loss = control_flow_ops.with_dependencies([barrier], G_loss)

    ###########################
    # Compute gradient updates
    ###########################

    dict_G_vars = G.get_trainable_variables()
    G_vars = [dict_G_vars[k] for k in dict_G_vars.keys()]

    dict_D_vars = D.get_trainable_variables()
    D_vars = [dict_D_vars[k] for k in dict_D_vars.keys()]

    G_gradvar = G_opt.compute_gradients(G_loss, var_list=G_vars)
    G_update = G_opt.apply_gradients(G_gradvar, name='G_loss_minimize')

    D_gradvar = D_opt.compute_gradients(D_loss, var_list=D_vars)
    D_update = D_opt.apply_gradients(D_gradvar, name='D_loss_minimize')

    ##########################
    # Group training ops
    ##########################
    train_ops = [G_update, D_update]
    loss_ops = [G_loss, D_loss, D_loss_real, D_loss_fake]

    ##########################
    # Summary ops
    ##########################

    # Add summary for gradients
    tu.add_gradient_summary(G_gradvar)
    tu.add_gradient_summary(D_gradvar)

    # Add scalar symmaries
    tf.summary.scalar("G loss", G_loss)
    tf.summary.scalar("D loss real", D_loss_real)
    tf.summary.scalar("D loss fake", D_loss_fake)

    summary_op = tf.summary.merge_all()

    ############################
    # Start training
    ############################

    # Initialize session
    saver = tu.initialize_session(sess)

    # Start queues
    coord = tu.manage_queues(sess)

    # Summaries
    writer = tu.manage_summaries(sess)

    for e in tqdm(range(FLAGS.nb_epoch), desc="\nTraining progress"):

        t = tqdm(range(FLAGS.nb_batch_per_epoch), desc="Epoch %i" % e, mininterval=0.5)
        for batch_counter in t:

            # X_batch, _ = mnist.train.next_batch(FLAGS.batch_size)
            # if FLAGS.data_format == "NHWC":
            #     X_batch = np.reshape(X_batch, [-1, 28, 28, 1])
            # else:
            #     X_batch = np.reshape(X_batch, [-1, 1, 28, 28])
            # X_batch = (X_batch - 0.5) / 0.5

            X_batch = du.sample_batch(imgs, FLAGS.batch_size)
            output = sess.run(train_ops + loss_ops + [summary_op], feed_dict={X_real: X_batch})

            if batch_counter % (FLAGS.nb_batch_per_epoch // 20) == 0:
                writer.add_summary(output[-1], e * FLAGS.nb_batch_per_epoch + batch_counter)
            lossG, lossDreal, lossDfake = [output[2], output[4], output[5]]

            t.set_description('Epoch %i: - G loss: %.2f D loss real: %.2f Dloss fake: %.2f' %
                              (e, lossG, lossDreal, lossDfake))

            # variables = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
            # bmean = [v for v in variables if v.name == "generator/conv2D_1_1/BatchNorm/moving_mean:0"]
            # print sess.run(bmean)
            # raw_input()

        # Plot some generated images
        # output = sess.run([X_G_output, X_real_output])
        output = sess.run([X_G_output, X_real_output], feed_dict={X_real: X_batch})
        vu.save_image(output, FLAGS.data_format, e)

        # Save session
        saver.save(sess, os.path.join(FLAGS.model_dir, "model"), global_step=e)

    print('Finished training!')
