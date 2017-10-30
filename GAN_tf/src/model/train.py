import os
import sys
import models
import numpy as np
import tensorflow as tf
from tqdm import tqdm
sys.path.append("../utils")
import visualization_utils as vu
import training_utils as tu
import data_utils as du
import objectives

FLAGS = tf.app.flags.FLAGS


def train_model():

    # Setup session
    sess = tu.setup_session()

    # Setup async input queue of real images
    X_real = du.input_data(sess)

    #######################
    # Instantiate generator
    #######################
    list_filters = [256, 128, 64, 3]
    list_strides = [2] * len(list_filters)
    list_kernel_size = [3] * len(list_filters)
    list_padding = ["SAME"] * len(list_filters)
    output_shape = X_real.get_shape().as_list()[1:]
    G = models.Generator(list_filters, list_kernel_size, list_strides, list_padding, output_shape,
                         batch_size=FLAGS.batch_size, data_format=FLAGS.data_format)

    ###########################
    # Instantiate discriminator
    ###########################
    list_filters = [32, 64, 128, 256]
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

    # noise_input = tf.random_normal((FLAGS.batch_size, FLAGS.noise_dim,), stddev=0.1, name="noise_input")
    noise_input = tf.random_uniform((FLAGS.batch_size, FLAGS.noise_dim,), minval=-1, maxval=1, name="noise_input")
    X_fake = G(noise_input)

    # output images
    X_G_output = du.unnormalize_image(X_fake, name="X_G_output")
    X_real_output = du.unnormalize_image(X_real, name="X_real_output")

    D_real = D(X_real)
    D_fake = D(X_fake, reuse=True)

    ###########################
    # Instantiate losses
    ###########################

    G_loss = objectives.binary_cross_entropy_with_logits(D_fake, tf.ones_like(D_fake))
    D_loss_real = objectives.binary_cross_entropy_with_logits(D_real, tf.ones_like(D_real))
    D_loss_fake = objectives.binary_cross_entropy_with_logits(D_fake, tf.zeros_like(D_fake))

    # G_loss = objectives.wasserstein(D_fake, -tf.ones_like(D_fake))
    # D_loss_real = objectives.wasserstein(D_real, -tf.ones_like(D_real))
    # D_loss_fake = objectives.wasserstein(D_fake, tf.ones_like(D_fake))

    D_loss = D_loss_real + D_loss_fake

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

    # clip_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in D_vars]

    G_update = G_opt.minimize(G_loss, var_list=G_vars, name='G_loss_minimize')
    D_update = D_opt.minimize(D_loss, var_list=D_vars, name='D_loss_minimize')

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

    # Run checks on data dimensions
    list_data = [noise_input, X_real, X_fake, X_G_output, X_real_output]
    output = sess.run([noise_input, X_real, X_fake, X_G_output, X_real_output])
    tu.check_data(output, list_data)

    for e in tqdm(range(FLAGS.nb_epoch), desc="Training progress"):

        list_G_loss = []
        list_D_loss_real = []
        list_D_loss_fake = []

        t = tqdm(range(FLAGS.nb_batch_per_epoch), desc="Epoch %i" % e, mininterval=0.5)
        for batch_counter in t:

            o_D = sess.run([D_update, D_loss_real, D_loss_fake])
            sess.run([G_update, G_loss])
            o_G = sess.run([G_update, G_loss])
            output = sess.run([summary_op])

            list_G_loss.append(o_G[-1])
            list_D_loss_real.append(o_D[-2])
            list_D_loss_fake.append(o_D[-1])

            # output = sess.run(train_ops + loss_ops + [summary_op])
            # list_G_loss.append(output[2])
            # list_D_loss_real.append(output[4])
            # list_D_loss_fake.append(output[5])

            if batch_counter % (FLAGS.nb_batch_per_epoch // (int(0.5 * FLAGS.nb_batch_per_epoch))) == 0:
                writer.add_summary(output[-1], e * FLAGS.nb_batch_per_epoch + batch_counter)

        t.set_description('Epoch %i: - G loss: %.3f D loss real: %.3f Dloss fake: %.3f' %
                          (e, np.mean(list_G_loss), np.mean(list_D_loss_real), np.mean(list_D_loss_fake)))

        # Plot some generated images
        output = sess.run([X_G_output, X_real_output])
        vu.save_image(output, FLAGS.data_format, e)

        # Save session
        saver.save(sess, os.path.join(FLAGS.model_dir, "model"), global_step=e)

        if e == 0:
            print(len(list_data))
            output = sess.run([noise_input, X_real, X_fake, X_G_output, X_real_output])
            tu.check_data(output, list_data)

    print('Finished training!')
