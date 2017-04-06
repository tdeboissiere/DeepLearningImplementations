import os
import sys
import models
from tqdm import tqdm
import tensorflow as tf
sys.path.append("../utils")
import visualization_utils as vu
import training_utils as tu
import data_utils as du

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
    G_opt = tf.train.AdamOptimizer(learning_rate=1E-4, name='G_opt', beta1=0.5, beta2=0.9)
    D_opt = tf.train.AdamOptimizer(learning_rate=1E-4, name='D_opt', beta1=0.5, beta2=0.9)

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

    G_loss = -tf.reduce_mean(D_fake)
    D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)

    epsilon = tf.random_uniform(
        shape=[FLAGS.batch_size, 1, 1, 1],
        minval=0.,
        maxval=1.
    )
    X_hat = X_real + epsilon * (X_fake - X_real)
    D_X_hat = D(X_hat, reuse=True)
    grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
    if FLAGS.data_format == "NCHW":
        red_idx = [1]
    else:
        red_idx = [-1]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
    gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
    D_loss += 10 * gradient_penalty

    ###########################
    # Compute gradient updates
    ###########################

    dict_G_vars = G.get_trainable_variables()
    G_vars = [dict_G_vars[k] for k in dict_G_vars.keys()]

    dict_D_vars = D.get_trainable_variables()
    D_vars = [dict_D_vars[k] for k in dict_D_vars.keys()]

    G_gradvar = G_opt.compute_gradients(G_loss, var_list=G_vars, colocate_gradients_with_ops=True)
    G_update = G_opt.apply_gradients(G_gradvar, name='G_loss_minimize')

    D_gradvar = D_opt.compute_gradients(D_loss, var_list=D_vars, colocate_gradients_with_ops=True)
    D_update = D_opt.apply_gradients(D_gradvar, name='D_loss_minimize')

    ##########################
    # Group training ops
    ##########################
    loss_ops = [G_loss, D_loss]

    ##########################
    # Summary ops
    ##########################

    # Add summary for gradients
    tu.add_gradient_summary(G_gradvar)
    tu.add_gradient_summary(D_gradvar)

    # Add scalar symmaries
    tf.summary.scalar("G loss", G_loss)
    tf.summary.scalar("D loss", D_loss)
    tf.summary.scalar("gradient_penalty", gradient_penalty)

    summary_op = tf.summary.merge_all()

    ############################
    # Start training
    ############################

    # Initialize session
    saver = tu.initialize_session(sess)

    # Start queues
    tu.manage_queues(sess)

    # Summaries
    writer = tu.manage_summaries(sess)

    for e in tqdm(range(FLAGS.nb_epoch), desc="Training progress"):

        t = tqdm(range(FLAGS.nb_batch_per_epoch), desc="Epoch %i" % e, mininterval=0.5)
        for batch_counter in t:

            for di in range(5):
                sess.run([D_update])

            output = sess.run([G_update] + loss_ops + [summary_op])

            if batch_counter % (FLAGS.nb_batch_per_epoch // 20) == 0:
                writer.add_summary(output[-1], e * FLAGS.nb_batch_per_epoch + batch_counter)

            t.set_description('Epoch %i' % e)

        # Plot some generated images
        output = sess.run([X_G_output, X_real_output])
        vu.save_image(output, FLAGS.data_format, e)

        # Save session
        saver.save(sess, os.path.join(FLAGS.model_dir, "model"), global_step=e)

    print('Finished training!')
